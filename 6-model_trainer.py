# import libraries
import os
import warnings
import logging
os.environ["TORCH_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PL_GLOBAL_SEED_LOGGING"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)

from src.tft_pipeline import (
    load_and_prep_data, create_datasets, create_dataloaders, init_loggers,
    validate_config_queue, train_model)
import yaml
import lightning.pytorch as pl

def main():
    with open('model_config.yaml', 'r') as f:
        master_yaml = yaml.safe_load(f)
    
    schema = master_yaml["FEATURE_SCHEMA"]
    run_configs = ["final_experiment"]

    print("🔍 Validating configuration queue...")

    validated_configs = validate_config_queue(
        run_queue=run_configs,
        yaml=master_yaml,
        schema=schema,
        model_defaults=master_yaml["MODEL_DEFAULTS"])
    
    print("✅ All configurations validated successfully. Starting sequential execution...\n")

    for run_name, config in validated_configs.items():
        print(f"\n🚀 Starting training for Run ID: {run_name}...")

        pl.seed_everything(42, workers=True)        

        # load and prepare data
        df = load_and_prep_data(
            config=config,
            schema=schema)
        
        # create datasets
        training, validation, testing = create_datasets(
            config=config,
            df=df)
        
        # create dataloaders
        train_dataloader, val_dataloader, _ = create_dataloaders(
            training=training,
            validation=validation,
            testing=testing,
            config=config)
        
        # initialize loggers
        loggers = init_loggers(
            run_id=run_name,
            config=config,
            log_model=False)
        
        # train model
        train_model(
            config=config,
            run_id=run_name, 
            training=training,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loggers=loggers,
            fast_dev_run=False)


if __name__ == "__main__":
    main()