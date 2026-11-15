"""Check the seven previously frozen response pairs in additional public human assays."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
RUN = ROOT / "results/idop_feasibility_20261109"
OUT = RUN / "independent_response_followup_20261110"
SCREEN = RUN / "transcriptome_readout/external_screen.csv"


def array_data(accession, platform):
    series_path = DATA / accession / f"{accession}_series_matrix.txt.gz"
    annot_path = DATA / ("GSE22795" if platform == "GPL6947" else accession) / f"{platform}.annot.gz"
    with gzip.open(series_path, "rt", encoding="utf-8", errors="replace") as stream:
        text = stream.read()
    series, matrix = text.split("!series_matrix_table_begin\n", maxsplit=1)
    fields = {}
    for line in series.splitlines():
        if line.startswith(("!Sample_title\t", "!Sample_geo_accession\t")):
            key, *values = next(csv.reader([line], delimiter="\t"))
            fields[key] = values
    samples = dict(zip(fields["!Sample_geo_accession"], fields["!Sample_title"], strict=True))
    values = pd.read_csv(io.StringIO(matrix.split("!series_matrix_table_end")[0]),
                         sep="\t").set_index("ID_REF")
    assert set(values.columns) == set(samples) and values.index.is_unique
    with gzip.open(annot_path, "rt", encoding="utf-8", errors="replace") as stream:
        annot = stream.read()
    annot = pd.read_csv(io.StringIO(annot[annot.index("ID\tGene title"):]), sep="\t",
                        usecols=["ID", "Gene symbol", "Gene ID"], comment="!", dtype=str)
    return values, samples, annot


def matrix_rows(accession, platform, source, markers, design):
    values, sample_titles, annot = array_data(accession, platform)
    for time, control_rule, treatment_rule in design:
        control = [key for key, title in sample_titles.items() if re.fullmatch(control_rule, title)]
        treated = [key for key, title in sample_titles.items() if re.fullmatch(treatment_rule, title)]
        assert control and len(control) == len(treated)
        for gene in [source, *markers]:
            mapped = annot.loc[annot["Gene symbol"].eq(gene), "ID"].drop_duplicates()
            probes = values.index.intersection(mapped)
            for probe in probes:
                yield (accession, source, time, gene, str(probe),
                       values.loc[probe, control], values.loc[probe, treated])


def region_rows(markers):
    rna = pd.read_csv(DATA / "GSE164786/GSE164786_PRDM14_RNA_seq.csv.gz", dtype={"Chromosome": str})
    control = rna.filter(regex=r"_No_IAA_Rep[1-3]_RNAseq$").columns.tolist()
    treated = rna.filter(regex=r"_Plus_IAA_Rep[1-3]_RNAseq$").columns.tolist()
    assert len(control) == len(treated) == 3
    for gene in ["PRDM14", *markers]:
        subset = rna.loc[rna.Probe.str.fullmatch(re.escape(gene) + r"-\d+", na=False)]
        # Coordinate duplicates have identical sample values in the deposited table.
        key = ["Chromosome", "Start", "End", "Probe.Strand"]
        for _, dup in subset.groupby(key, dropna=False):
            assert dup[control + treated].nunique().max() <= 1
        for _, region in subset.drop_duplicates(key).iterrows():
            label = f"{region['Chromosome']}:{region['Start']}-{region['End']}:{region['Probe.Strand']}"
            yield ("GSE164786", "PRDM14", "24h", gene, label, region[control], region[treated])


def summarize(rows, expected):
    detail = []
    for accession, source, time, gene, feature, control, treated in rows:
        control = control.astype(float)
        treated = treated.astype(float)
        assert np.isfinite(control).all() and np.isfinite(treated).all()
        direction = expected[source][gene]
        contrast = treated.mean() - control.mean()
        detail.append(dict(dataset=accession, source=source, time=time, gene=gene, feature=feature,
                           expected_direction=direction, delta=contrast,
                           aligned=bool(direction * contrast > 0),
                           complete_expected_separation=bool((direction * treated).min() > (direction * control).max()),
                           complete_opposite_separation=bool((direction * treated).max() < (direction * control).min()),
                           n_control=len(control), n_treated=len(treated),
                           control_values=json.dumps(control.to_dict()),
                           treated_values=json.dumps(treated.to_dict())))
    feature_table = pd.DataFrame(detail)
    summary = feature_table.groupby(["dataset", "source", "time", "gene"], sort=False).agg(
        direction=("expected_direction", "first"), features=("feature", "size"),
        expected_sign=("aligned", "sum"),
        expected_separated=("complete_expected_separation", "sum"),
        opposite_separated=("complete_opposite_separation", "sum"),
        median_contrast=("delta", "median"), n_control=("n_control", "first"),
        n_treated=("n_treated", "first"),
    ).reset_index()
    feature_table.to_csv(OUT / "feature_values.csv", index=False)
    summary.to_csv(OUT / "response_summary.csv", index=False)
    print(summary.to_string(index=False))


def main():
    OUT.mkdir(exist_ok=True)
    passing = pd.read_csv(SCREEN).query("external_status == 'pass'")
    assert len(passing) == 7
    expected = {source: {source: -1, **{
        row.marker: int(np.sign(row.day5_response)) for row in subset.itertuples()}}
        for source, subset in passing.groupby("source")}
    freeze = {
        "original_screen_sha256": hashlib.sha256(SCREEN.read_bytes()).hexdigest(),
        "expected_direction": expected,
        "GSE22795": "PRDM14 shRNA vs luciferase shRNA, day 3, 3 vs 3",
        "GSE164786": "PRDM14 auxin depletion vs no auxin, 24 h, 3 vs 3 libraries",
        "GSE21135": "OCT4 shRNA vs luciferase shRNA, days 2 and 3 separately, each 3 vs 3",
        "GSE34904": "H1 shOCT4 vs H1P by matched collection day 1–8, each 1 vs 1",
        "criteria": "Within each array probe or unique RNA-seq coordinate, compare treated and control means. "
                    "Report expected sign and complete expected or opposite separation. "
                    "Day contrasts with one sample per condition are descriptive. "
                    "Probes, regions and days are not treated as replicates.",
    }
    frozen = OUT / "followup_protocol.json"
    if frozen.exists():
        assert json.loads(frozen.read_text(encoding="utf-8")) == freeze
    else:
        frozen.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    prdm14 = [gene for gene in expected["PRDM14"] if gene != "PRDM14"]
    pou5f1 = [gene for gene in expected["POU5F1"] if gene != "POU5F1"]
    rows = [
        *matrix_rows("GSE22795", "GPL6947", "PRDM14", prdm14, [
            ("day3", r"Luciferase shRNA_3 Days knockdown_rep[1-3]",
             r"PRDM14 shRNA_3 Days knockdown_rep[1-3]")]),
        *region_rows(prdm14),
        *matrix_rows("GSE21135", "GPL6947", "POU5F1", pou5f1, [
            (f"day{day}", rf"Luciferase shRNA_{day} Days knockdown_rep[1-3]",
             rf"OCT4 shRNA_{day} Days knockdown_rep[1-3]") for day in (2, 3)]),
        *matrix_rows("GSE34904", "GPL10558", "POU5F1", pou5f1, [
            (f"day{day}", rf"H1P_{day}", rf"shOCT4_{day}") for day in range(1, 9)]),
    ]
    summarize(rows, expected)
    input_paths = [
        SCREEN, DATA / "GSE164786/GSE164786_PRDM14_RNA_seq.csv.gz",
        DATA / "GSE22795/GPL6947.annot.gz", DATA / "GSE34904/GPL10558.annot.gz",
        *(DATA / gse / f"{gse}_series_matrix.txt.gz" for gse in ("GSE22795", "GSE21135", "GSE34904")),
    ]
    (OUT / "input_sha256.json").write_text(json.dumps({
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

