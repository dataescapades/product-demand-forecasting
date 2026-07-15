# Data Attribution
The datasets located in the /data directory are sourced from Online Retail II dataset[^1] by Daqing Chen from the UCI Machine Learning Repository and are used under the terms of the [Creative Commons Attributions 4.0 International license](https://creativecommons.org/licenses/by/4.0/legalcode). The [Edited Dataset](data/online_retail_II%20-%20edited.csv) imported in the first notebook removes duplicates between sheets in Excel, concatenates the sheets, and saved the dataset as a CSV.

# Environment
This project was coded in VS Code using Jupyter and Data Wrangler extensions. An [environment YAML](environment.yaml) is included in the root directory.

# Project Structure
## Notebooks and Code
Notebooks and code for all steps of this project can be found in the the project root directory. They are numbered to indicate the order they were performed.
The cleaning, feature extraction, and feature engineering notebooks (steps 1-4) use Jupyter magic commands. The learning rate finder, model training, optimization, and testing are executable in the terminal.
- [1 - Cleaning](1-cleaning.py)
- [2 - Embeddings](2-feature_%20extraction_%20embeddings.py)
- [3 - Category Extraction](3-feature_extraction_category.py)
- [4 - Feature Engineering](4-feature_engineering.py)
- [5 - Learning Rate Finder](5-lr_finder.py)
- [6 - Model Trainer](6-model_trainer.py)
- [7 - Hyperparameter Optimizer](7-optuna.py)
- [8 - Model Tester](8-model_testing.py)

Executed notebooks for steps 1-4 can be found in the executed_notebooks directory.
- [1 - Cleaning (executed)](executed_notebooks/1-cleaning.html)
- [2 - Embeddings (executed)](executed_notebooks/2-feature_%20extraction_%20embeddings.html)
- [3 - Category Extraction (executed)](executed_notebooks/3-feature_extraction_category.html)
- [4 - Feature Engineering (executed)](executed_notebooks/4-feature_engineering.html)

## Additional Code
Helper code can be found in the src directory.
- [Cleaning Functions (steps 1-4)](src/functions.py)
- [Modeling Functions (steps 5-8)](src/tft_pipeline.py)
- [Pydantic Model Schema](src/model_schema.py)

Steps 5-8 use configurations defined in a YAML found in the root directory.
- [Model Configuration](model_config.yaml)

## Data
The [Dataset](data/online_retail_II%20-%20edited.csv) imported in the first notebook removes duplicates found between the sheets in the Excel file, concatenated the sheets, and was saved to CSV. Dictionaries were created to assist in cleaning the product descriptions in step 1 and saved in the following JSON files:
- [Mispelled Words](data/misspelled_words.json)
- [Word Variations](data/variations_dict.json)
- [Word Variations (regex)](data/variations_dict_regex.json)

Intermediate and final versions of the dataset are not included in the /data directory due to storage limits but can be recreated by the user by running notebooks 1-4 in order.

## Logs
Tensorboard, MLflow, and Optuna logs from steps 5-8 can be found in the logs directory.

## Model Checkpoints
Checkpoints of the best model from each run are saved in the models directory.

## Images
Screenshots and images embedded in reports can be found in the images directory.

## Reports
The [Technical Report](Technical_Report.md) is designed for a technical audience covers the details of the project reasoning, execution, and findings in depth. The [Executive Report](Executive_Report.md) is designed for a non-technical, executive audience, covering the key points and business application.

# Modeling Validation and Enforcement
Steps 5-8 use the [model configuration YAML](model_config.yaml) and [Pydantic schema](src/model_schema.py) to enforce all model parameter and data input. This is validated at the beginning of script execution, allowing for multiple runs to be specified and any errors caught prior to significant time investment. The scripts for steps 5-8 should not require alteration except to supply the name(s) of the desired run configuration(s) defined in the YAML.

## Model Configuration YAML
All variables that can be passed to the model are defined in the YAML.
- FILE PATHS - paths to any file used in the scripts
- GLOBAL DATA SCHEMA - all features in the dataset with associated data types cast during the run.
- DEFAULT HYPERPARAMETERS - all parameters and variables passed to the model
- EXPERIMENT CONFIGURATIONS - configuration for each run. Should call the default parameters then customize with desired changes, which ensures all required variables are passed to Pydantic schema for validation

## Pydantic
Configuration from the model configuration YAML is passed to Pydantic for validation and enforcement.

[^1]: Chen, D. (2019, September 20). Online Retail II [Dataset]. Retrieved from UCI Machine Learning Repository: https://doi.org/10.24432/C5CG6D