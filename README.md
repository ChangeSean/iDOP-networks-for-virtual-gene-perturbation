# iDOP review archive v0.1.0

This archive accompanies the manuscript *A weighted iDOP network decomposes virtual perturbation responses in human pluripotent cells*. It contains the fitted matrix, aggregate guide trajectories, model settings, figure source data and Python source used for the analyses.

## Reproduce the results

Use a fresh Python 3.13 environment. Run the commands from this directory.

```sh
python -m pip install -r requirements.txt
python reproduce.py
python reproduce.py --refit
python reproduce.py --figures
```

The default command checks the frozen nomination, the full network and the NANOG analysis. The refit command reconstructs the full fit, both whole-target holdouts and all 32 target-level bootstrap refits from the archived trajectories. The figure command writes five regenerated figures under `results/idop_feasibility_20261109/manuscript_figures` and `outputs/idop_followup_figures_20261112`. The final figure uses the archived Hallmark statistics. Install Arial for appearance closest to the submission figures.

For a separate recalculation of the Hallmark statistics from the original third-party files, follow `DATA_SOURCES.md` and run `python reproduce.py --hallmark`.

`CHECKSUMS.sha256` records the files in this review version. The vendored iDOP edge selector is covered by `idop_diffusion/_vendor/LICENSE.idopnetwork`. 


Run `python reproduce.py --supplement` to recalculate the 108-gene sensitivity analysis described in the Supplementary Information.
