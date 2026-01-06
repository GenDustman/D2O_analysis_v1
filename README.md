# D2O Analysis Pipeline 

This repository contains a Python-based analysis pipeline for the COHERENT D₂O detector. The pipeline processes *processed ROOT files* (PMT + SiPM waveform products) to extract key physics quantities such as total photoelectrons (P.E.), correlated time differences (Δt), and multiplicity. It also produces per-run QA plots, low-light SPE calibration fits, and global (merged) summary plots.

## Pipeline overview

The workflow has two stages:

1. **Parallel Processing (Map phase)**  
   The requested run range is split into SLURM sub-jobs. Each sub-job loops over runs, applies calibrations/cuts, and writes intermediate binary outputs (`.npy/.pkl/.json`) plus per-run debug plots.

2. **Aggregation (Reduce phase)**  
   A master job merges all sub-job outputs into global arrays/histograms and produces publication-ready plots in `MASTER_RESULTS/`.

## Main scripts

- `config.py`  
  Single source of truth for input paths, cut values, histogram binning, and optional modules (thin-veto / BRN).

- `submit_veto.sh`  
  End-to-end driver:
  - creates a timestamped `analysis_*` directory,
  - snapshots code into `analysis_*/code/`,
  - submits worker sub-jobs for the run range,
  - submits the master aggregation with SLURM dependency (`afterok`).

- `Read_Cut_Hist_D2O_multi_veto.py`  
  Worker (map step). Processes runs and produces:
  - per-run QA plots (correlation maps, cut histograms, low-light fits),
  - per-subjob aggregated intermediate files for the master reducer.

- `aggregate_master_veto.py`  
  Master (reduce step). Loads all `subjob_*` outputs and generates final plots + global bookkeeping.

- `submit_aggregation_only.sh`  
  Utility: re-run **only** the aggregation step on an existing `analysis_*` directory (useful for adjusting binning/fit windows/plot styling without reprocessing runs).


## Setup

### Prerequisites
- A SLURM environment (jobs are submitted via `sbatch`).
- Python 3 with typical dependencies (`numpy`, `pandas`, `matplotlib`, `scipy`, `uproot`, etc.).
  - On `ernest`, the default Python 3 is `/usr/bin/python3` (unless you activate a conda environment).
- Access to the processed ROOT files for the selected phase (M1 or M2).

### Data transfer (ORNL → MEG)
We recommend using **Globus** to transfer D₂O data from ORNL `phylogin1` to MEG.

- Becca’s LabArchives note:  
  [Transfer terabytes of data](https://mynotebook.labarchives.com/MTE1MjgwMy42fDg4Njc3Mi84ODY3NzIvTm90ZWJvb2svMjM4NzgwODQ4MnwyOTI2MzQ3LjU5OTk5OTk5OTY=/page/2052467-32)

- Globus authentication uses your ORNL Guest Portal credentials:  
  https://guest.ornl.gov/

- As of **Jan 5, 2026**, the D₂O data path on ORNL `phylogin1` is:
/data41/coherent/data/d2o/

## How to run

### Full processing + aggregation (recommended for new run ranges)
Use this when running a **new** run range, or when you changed logic affecting per-run derived data/cuts.

1. Edit `config.py` to point to your data directories (M1/M2) and adjust cuts/bins if needed.
2. Edit `submit_veto.sh`:
 - `SCRIPT_DIR` (code path)
 - `start_run`, `end_run`, `M1_or_M2`, `njobs`
 - output base directory
3. Submit:
 ```bash
 sh submit_veto.sh
```



## Appendix
For more details, the technote of this project: [edit technote](https://www.overleaf.com/8668822666tgwbrgxvmqhm#cc11df), **D2O Analysis Pipeline Documentation.pdf**.

The technote about data structure and variables from Tulasi Subedi:  
**D2O_Processed_Data_Variable_Description.pdf**

Note: the processed-data path referenced in that PDF may be obsolete; please use the current paths documented in the technote/README.

