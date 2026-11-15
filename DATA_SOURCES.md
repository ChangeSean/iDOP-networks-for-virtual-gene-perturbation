# Data sources and file map

All expression contrasts in the model use guide-by-day population means of log(1 + counts per 10,000) minus same-day pooled control means. The sample axis in `source_data/training_states.npz` contains the 16 targets in `protocol.json` for guide 1, followed by those same targets for guide 2. Its time axis is days 2, 3, 4 and 5. Its feature axis is the 23 core nodes, followed by the gene symbols in the stored `genes` array. The full ordering also appears in `nodes.npy`.

| Result | Input or source file | Origin |
|---|---|---|
| Figure 1 and Supplementary Table 1 | `transcriptome_readout/all_scored_pairs.csv`, `eligible_readouts.csv`, `frozen_candidates.csv`, `external_screen.csv` | [GSE213069](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213069) and [GSE303158](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE303158) |
| Figure 2 | `transcriptome_readout/nodes.npy`, `coupling.npz` and `manuscript_figures/Figure_S4_five_readout_decompositions.csv` | Fit to GSE213069. Matrix rows are receivers and columns are senders. |
| NANOG to POU5F1 | `source_data/NANOG_full_model.csv`, `source_data/NANOG_GSE303158_source.csv` | Same fitted matrix as Figure 2. External table contains guide and control means, counts and contrasts for primed and naïve cells from GSE303158. |
| Figure 3 | `manuscript_figures/Figure_3_DNMT3B_day5_comparison.csv`, `Figure_3_observed_holdout_trajectories.csv` and `Table_S2A_primed_CRISPR.csv` to `Table_S2C_naive_RNA_regions.csv` | GSE213069, GSE303158, [GSE22795](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE22795) and [GSE164786](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164786). Overlapping RNA regions come from the same libraries. |
| Figure 4 and Supplementary Table 2 | `manuscript_figures/Figure_3_case_ranks.csv`, `Figure_3_observed_holdout_trajectories.csv`, `independent_response_followup_20261110/feature_values.csv`, `feng_published_effects.csv` | GSE213069, GSE22795, GSE164786, [GSE21135](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE21135) and the [published CRISPRi effects](https://doi.org/10.6084/m9.figshare.26819743.v1). |
| Figure 5 | `source_data/Figure_5_Hallmark_screen.csv`, `Figure_5_Hallmark_protocol.json` | Model responses, published CRISPRi effects and the Enrichr MSigDB Hallmark 2020 library. |

The shortened paths for the transcriptome, figure and follow-up folders start at `results/idop_feasibility_20261109/`. The manuscript's figure numbers appear in the first column. Some source filenames reflect numbering from drafts.

Download URLs and hashes for the GSE213069 count archives, the primed GSE303158 object and the Feng et al. tables are stored as manifests under `source_data/`. The [naïve Seurat object](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303158/suppl/GSE303158_Seurat_after_mixscape_naive.RData.gz) has SHA-256 `0925fb3251d4fadec396e9b497ce2d621c2e756c2cab1c6ff10c7242ff00d5f7`. For the NANOG table, the RNA count assay was normalized by each cell's RNA total. Single-guide NANOG cells and non-targeting control cells were grouped by condition and batch, and contrasts were computed separately for each guide.

## Optional Hallmark recalculation

Save the [Enrichr Hallmark 2020 library](https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020) as `data/gene_sets/MSigDB_Hallmark_2020_Enrichr.gmt`. The expected SHA-256 is stored in `Figure_5_Hallmark_protocol.json`. Save the [Feng et al. published targeted effects](https://ndownloader.figshare.com/files/49291630) as `data/Feng_iPSC_CRISPRi_figshare/effects/TargetedScreen_LFC_byGene.tsv.gz`. From the archive root, run `python reproduce.py --hallmark`.


## Supplementary sensitivity analysis

The common-panel analysis uses `source_data/supplement_108_panel.npz` and the published case definitions in `results/idop_feasibility_20261109/renge_108_case_followup_20261110/case_protocol.json`. The five independently published targeted CRISPRi effects cited in Supplementary Table 2 are in `source_data/Supplementary_Table_2_targeted_CRISPRi.csv`.
