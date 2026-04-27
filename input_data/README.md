# Input Data

This directory contains the source data used by the synthetic employment income microdata workflow.

## Data Sources

| File | Usage |
| --- | --- |
| `city-level_minimum_wage_standards_2018.xlsx` | City-level 2018 minimum wage standards used to winsorize urban minimum wages in the microdata. |
| `macro_master.parquet` | 2020 census data from provincial statistical bureaus, used as small-area constraints. |
| `chip_employed.parquet` | 2018 employed micro-survey records; used with the census inputs as the core data for spatial microsimulation. |
| `Provincia_wage_data_for_private_and_non-private_sector_employment_2018–2020.xlsx` | Provincial wage and employment data used for income adjustment of urban samples. |
| `Per_capita_disposable_income_by_city_2018–2020.xlsx` | City-level disposable income data used for income adjustment of rural samples. |
| `Employment_and_wage_data_by_city_2020.xlsx` | City-level employment and wage data used for external validation of urban samples and scatter plots. |
| `Per_capita_disposable_income_by_district_2020.xlsx`, `margin_total_pop.parquet` | District-level disposable income data used for external validation of rural samples and scatter plots. |
| `cfps2020_employed_micro.parquet`, `original_chip2018.parquet` | Two microdata used for external validation against synthetic data and for plotting box plots. |
| `geo_code_mapping_2020.json` | Geographic code mapping used to align administrative units across workflow stages. |
| `task_list.parquet` | Workflow task list used by the IPF and integerisation stages. |
