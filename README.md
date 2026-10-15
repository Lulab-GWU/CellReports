# Cell Reports anxiety-ensemble analysis

Jupyter notebooks and processed source data for **“Context-dependent
recruitment of mouse prefrontal ensembles exhibits distinct vulnerabilities to
MeCP2 deficiency.”**

The notebooks in `code/figures/` have been run successfully with the data
included in this repository. They use repository-relative paths and save their
generated figures and tables under `results/`.

## Setup

Download or clone the complete repository, then run the following commands from
its root directory:

```bash
conda env create -f environment.yml
conda activate cellreports-anxiety-ensembles
jupyter lab
```

Open a notebook in `code/figures/` and select **Run All Cells**. The notebooks
locate the repository root through `pyproject.toml`, so no personal file paths
need to be edited.

## Figure notebooks

| Notebook | Main output |
|---|---|
| `code/figures/Fig_1H_OFT_spatialMap.ipynb` | `results/figures/Fig_1H.png` |
| `code/figures/Fig_3A_sampleTraces.ipynb` | `results/figures/Fig_3A.png` |
| `code/figures/Fig_4B_EPM_spatialMap.ipynb` | `results/figures/Fig_4B.png` |
| `code/figures/Fig_5A_EPM_sampleTraces.ipynb` | `results/figures/Fig_5A.png` |
| `code/figures/Fig_S1C.ipynb` | `results/figures/Fig_S1C.png` |
| `code/figures/Fig_S3.ipynb` | Figure S3 panels in `results/figures/` and statistics in `results/tables/` |
| `code/figures/Fig_S7.ipynb` | Figure S7 heatmaps in `results/figures/` and summary tables in `results/tables/` |

Each notebook identifies its specific input files and analysis choices in its
opening Markdown cells. Compact figure inputs are stored under
`data/source_data/`.

## Example analysis notebooks

The notebooks in `code/analysis/` demonstrate two core analysis steps using
the included example mouse `data/OFT_Data/NC128_WT/`:

- `code/analysis/1_ER_AM.ipynb` calculates calcium-event rate (transient rate) and amplitude
  for the whole OFT session and for center, corner, sniffing, and grooming
  periods. It saves `OFT_WT_All_Eventrate.xlsx` and
  `OFT_WT_All_Amplitude.xlsx` under `results/tables/`.
- `code/analysis/2_ONOFF.ipynb` demonstrates the center-ON/OFF and corner-ON/OFF
  classification procedure and writes `OFT_ONOFF.csv` to the NC128 example
  folder.

Both notebooks call the original analysis functions in `src/Functions.py`.
The figure notebooks use shared plotting and validation helpers from
`src/utilities.py`. The example analysis notebooks are illustrative and are not
required to run the figure notebooks.

## Data and outputs

```text
CellReports/
├── code/
│   ├── figures/       # runnable figure notebooks
│   └── analysis/      # ER/AM and ON/OFF example notebooks
├── data/
│   ├── source_data/   # compact inputs used by figure notebooks
│   └── OFT_Data/
│       └── NC128_WT/  # example input used by analysis notebooks
├── src/               # shared analysis and plotting functions
├── results/
│   ├── figures/       # generated PNG files
│   └── tables/        # generated CSV/XLSX files
├── environment.yml
└── pyproject.toml
```

Generated files can be recreated by rerunning the corresponding notebook.
