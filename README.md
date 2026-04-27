# synthetic_employment_income

`synthetic_employment_income` is a workflow-oriented research codebase for building and validating synthetic employment income data in China.

The repository contains the core pipeline for IPF-based spatial microsimulation, integerisation, income adjustment, and distributional comparison plotting.

## Repository Structure

```text
synthetic_employment_income/
├── README.md
├── requirements.txt
├── run_full_workflow.py
├── boxplot/
│   ├── config.py
│   ├── layout.py
│   ├── prepare_cfps.py
│   ├── prepare_common.py
│   ├── prepare_synthetic.py
│   ├── utils.py
│   └── violin_panel.py
├── income_adj/
│   ├── calibration.py
│   ├── config.py
│   ├── data_loader.py
│   ├── main.py
│   ├── plot_style.py
│   ├── validation_rural.py
│   ├── validation_urban.py
│   └── validation_utils.py
├── input_data/
│   ├── Employment_and_wage_data_by_city_2020.xlsx
│   ├── Per_capita_disposable_income_by_city_2018–2020.xlsx
│   ├── Per_capita_disposable_income_by_district_2020.xlsx
│   ├── Provincia_wage_data_for_private_and_non-private_sector_employment_2018–2020.xlsx
│   ├── cfps2020_employed_micro.parquet
│   ├── chip_employed.parquet
│   ├── geo_code_mapping_2020.json
│   ├── macro_master.parquet
│   ├── margin_total_pop.parquet
│   ├── original_chip2018.parquet
│   └── task_list.parquet
├── integerisation/
│   ├── config.py
│   ├── run_qisi_main.py
│   └── shared/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── pipeline.py
│       ├── province_outputs.py
│       ├── qisi_core.py
│       └── utils.py
└── ipf/
    ├── config.py
    ├── run_ipf_pipeline.py
    ├── run_ipf_quality.py
    └── shared/
        ├── core.py
        ├── pipeline.py
        └── quality.py
```

## Modules

### `input_data/`

This directory stores the input datasets used by the workflow.

### `ipf/`

Runs the iterative proportional fitting (IPF) stage that produces the spatially constrained weights used by the downstream synthetic population workflow.

### `integerisation/`

Transforms IPF weights into integerised synthetic records and exports intermediate outputs for downstream income adjustment.

### `income_adj/`

Applies income adjustment and validation steps to the synthetic income outputs, including separate urban and rural validation.

### `boxplot/`

Prepares comparison frames and produces boxplots for synthetic and validation microdata.

### `run_full_workflow.py`

Runs the full workflow across IPF, integerisation, income adjustment, validation, and boxplot generation.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
```

## Running the Workflow

Run the full pipeline from the repository root:

```bash
.venv/bin/python run_full_workflow.py
```

Run selected stage groups only:

```bash
.venv/bin/python run_full_workflow.py --stages ipf,integerisation,income_adj,validation,boxplot
```
