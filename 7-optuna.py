import copy
import yaml
import optuna
import lightning.pytorch as pl
from optuna_integration import PyTorchLightningPruningCallback
from src.tft_pipeline import (
    validate_config_queue,
    load_and_prep_data,
    create_datasets,
    create_dataloaders,
    init_loggers,
    train_model)


def objective(trial, run_name, yaml, schema, model_defaults):
    trial_config = copy.deepcopy(yaml[run_name])
    suggested_hidden = trial.suggest_categorical("hidden_size", [64, 128])
    
    trial_config["hidden_size"] = suggested_hidden
    trial_config["dropout"] = trial.suggest_float("dropout", 0.05, 0.15)
    trial_config["max_epochs"] = 5
    
    mock_name = f"optuna_{run_name}_trial_{trial.number}"
    validated_output = validate_config_queue(
        run_queue=[mock_name],
        yaml={mock_name: trial_config},
        schema=schema,
        model_defaults=model_defaults)
    config = validated_output[mock_name]
    
    pl.seed_everything(42, workers=True)

    df = load_and_prep_data(
        config=config,
        schema=schema)

    training, validation, testing = create_datasets(
        config=config,
        df=df)
    
    train_dataloader, val_dataloader, _ = create_dataloaders(
        training=training,
        validation=validation,
        testing=testing,
        config=config)
    
    loggers = init_loggers(
        run_id=mock_name,
        config=config,
        log_model=False)
    
    pruning_callback = PyTorchLightningPruningCallback(
        trial,
        monitor="val_loss")
    
    trainer, _ = train_model(
        config=config,
        run_id=mock_name,
        training=training,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loggers=loggers,
        extra_callbacks=[pruning_callback])
    
    val_loss = trainer.callback_metrics["val_loss"].item()

    return val_loss

def main():
    with open('model_config.yaml', 'r') as f:
        master_yaml = yaml.safe_load(f)
        
    schema = master_yaml["FEATURE_SCHEMA"]
    model_defaults = master_yaml["MODEL_DEFAULTS"]
    run_name = "engineered_3_categories"

    study = optuna.create_study(
        study_name=f"{run_name}",
        storage=f"sqlite:///logs/optuna/{run_name}.db",
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner())
    
    print(f"🚀 Launching optimization loop for {run_name}...")

    study.optimize(
        lambda trial: objective(
            trial,
            run_name=run_name,
            yaml=master_yaml,
            schema=schema,
            model_defaults=model_defaults),
        n_trials=10)
    
    print(f"🏆 Best trial complete. Loss: {study.best_trial.value}")

if __name__ == "__main__":
    main()