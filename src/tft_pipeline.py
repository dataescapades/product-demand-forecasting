# import libraries
import os
import sys
sys.path.append('../src')

import pandas as pd
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MAE, SMAPE, RMSE
from src.model_schema import TFTConfig


def validate_config_queue(run_queue, yaml, schema, model_defaults):
    validated_configs = {}

    for run_name in run_queue:
        print(f"    [INTERNAL] {run_name}: Validating Configuration...")
        config = yaml.get(run_name)

        if config is None:
            raise KeyError(f"[ERROR] Run name '{run_name}' not found in model_config.yaml.")

        # Validate that the config keys match the model defaults keys
        required_params = set(model_defaults.keys())
        config_params = set(config.keys())
        missing_params = required_params - config_params
        
        if missing_params:
            raise KeyError(
                f"[ERROR] Missing required parameters in experiment config:\n"
                f"  ❌ {', '.join(missing_params)}\n"
                f"Ensure all MODEL_DEFAULTS keys are included in the experiment config.")

        invalid_params = config_params - required_params
        if invalid_params:
            raise KeyError(
                f"[ERROR] Invalid parameters in experiment config:\n"
                f"  ❌ {', '.join(invalid_params)}\n"
                f"Ensure only keys defined in MODEL_DEFAULTS are present in the experiment config.")

        # Validate data types and values using model schema
        try:
            validated_config =TFTConfig.model_validate(config, context={"schema": schema})
        except ValueError as e:
            raise ValueError(f"[CONFIGURATION ERROR]:\n {e}")
        
        validated_configs[run_name] = validated_config.model_dump()

        print(f"    [INTERNAL] {run_name}: Configuration validation passed.")

    return validated_configs


def load_and_prep_data(config, schema):
    print("    [INTERNAL] Loading data...")
    df = pd.read_csv(config["data_path"])

    print("    [INTERNAL] Setting data types...")

    for col in df.columns:
        df[col] = df[col].astype(schema[col])

    return df


def create_datasets(config, df):
    print("    [INTERNAL] Creating TimeSeriesDataSet...")
    training_cutoff = df["time_idx"].max() - config["max_prediction_length"]

    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx=config["time_idx"],
        target=config["target"],
        group_ids=config["group_ids"],
        max_encoder_length=config["max_encoder_length"],
        min_encoder_length=config["min_encoder_length"],
        max_prediction_length=config["max_prediction_length"],
        static_categoricals=config["static_categoricals"],
        static_reals=config["static_reals"],
        time_varying_known_categoricals=config["time_varying_known_categoricals"],
        time_varying_known_reals=config["time_varying_known_reals"],
        time_varying_unknown_categoricals=config["time_varying_unknown_categoricals"],
        time_varying_unknown_reals=config["time_varying_unknown_reals"],
        target_normalizer=GroupNormalizer(
            groups=config["group_ids"],
            method=config["normalizer_method"],
            transformation=config["normalizer_transformation"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True)

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[lambda x: x.time_idx <= training_cutoff],
        predict=True,
        stop_randomization=True)

    testing = TimeSeriesDataSet.from_dataset(
        training,
        df,
        predict=True,
        stop_randomization=True)

    return training, validation, testing


def create_dataloaders(training, validation, testing, config):
    num_workers = max(1, os.cpu_count() - 1)
    print(f"    [INTERNAL] Using {num_workers} workers and batch size {config['batch_size']}...")

    print("    [INTERNAL] Creating training dataloader...")
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=config["batch_size"],
        num_workers=num_workers,
        shuffle=True,
        multiprocessing_context="spawn",
        worker_init_fn=pl.seed_everything)
    
    print("    [INTERNAL] Creating validation dataloader...")
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=config["batch_size"],
        num_workers=0)
    
    print("    [INTERNAL] Creating testing dataloader...")
    test_dataloader = testing.to_dataloader(
        train=False,
        batch_size=config["batch_size"],
        num_workers=0)
    
    return train_dataloader, val_dataloader, test_dataloader

def init_loggers(run_id, config, log_model=True):
    
    print(f"    [INTERNAL] Initializing loggers for learning rate finder Run ID: {run_id}...")
    os.makedirs(config["mlflow_dir"], exist_ok=True)
    absolute_mlflow_dir = os.path.abspath(config["mlflow_dir"])
    mlflow_db_path = os.path.join(absolute_mlflow_dir, "mlflow.db")

    mlflow_logger = MLFlowLogger(
        experiment_name=config["experiment_name"],
        run_name=run_id,
        tracking_uri=f"sqlite:///{mlflow_db_path}",
        log_model=log_model)
    
    tensorboard_logger = TensorBoardLogger(
        save_dir=config["tensorboard_dir"],
        name=run_id)
    
    return [mlflow_logger, tensorboard_logger]

def find_optimal_lr(training_dataset, train_dataloader, val_dataloader, config, loggers):
    loss_dict = {
        "quantile": QuantileLoss(quantiles=config["quantiles"]),
        "mae": MAE(),
        "smape": SMAPE(),
        "rmse": RMSE()}

    chosen_loss = loss_dict[config["loss"]]

    trainer = pl.Trainer(
        gradient_clip_val=config["gradient_clip_val"],
        accelerator=config["accelerator"],
        devices=config["devices"],
        logger=loggers)

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config["learning_rate"],
        hidden_size=config["hidden_size"],
        attention_head_size=config["hidden_size"] // 16,
        hidden_continuous_size=config["hidden_size"] // 4,
        dropout=config["dropout"],
        output_size=config["output_size"],
        loss=chosen_loss,
        optimizer=config["optimizer"],
        reduce_on_plateau_patience=config["reduce_on_plateau_patience"])

    tuner = Tuner(trainer)

    print("Finding optimal learning rate...")

    lr_finder = tuner.lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
        num_training=100)

    suggested_lr = lr_finder.suggestion()
    print(f"Suggested learning rate: {suggested_lr}")

    return lr_finder


def plot_lr_finder(lr_finder, mlflow_logger):
    suggested_lr = lr_finder.suggestion()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lr_finder.results["lr"], lr_finder.results["loss"])
    ax.set_xscale("log")
    best_loss = min(lr_finder.results["loss"])
    ax.set_ylim(best_loss * 0.9, best_loss * 4)
    ax.axvline(suggested_lr, color="red", linestyle="--", label=f"Suggested LR: {suggested_lr:.2e}")
    ax.set_title("Learning Rate Finder")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Quantile Loss")
    ax.legend()

    mlflow_logger.experiment.log_figure(
        run_id=mlflow_logger.run_id,
        figure=fig,
        artifact_file="plots/lr_finder_plot.png")
    
    plt.close(fig)


def train_model(config, run_id, 
                training, train_dataloader, val_dataloader,
                loggers, fast_dev_run=False, extra_callbacks=None):
    loss_dict = {
        "quantile": QuantileLoss(quantiles=config["quantiles"]),
        "mae": MAE(),
        "smape": SMAPE(),
        "rmse": RMSE()}

    chosen_loss = loss_dict[config["loss"]]

    early_stop_callback = EarlyStopping(
        monitor=config["monitor"],
        min_delta=config["min_delta"],
        patience=config["patience"],
        verbose=False,
        mode=config["mode"])

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=config["model_checkpoint_dir"],
        monitor=config["monitor"],
        filename=f"Best_Model-{run_id}",
        save_top_k=1,
        mode=config["mode"])

    lr_monitor_callback = LearningRateMonitor()

    trainer = pl.Trainer(
        fast_dev_run=fast_dev_run,
        max_epochs=config["max_epochs"],
        gradient_clip_val=config["gradient_clip_val"],
        accelerator=config["accelerator"],
        devices=config["devices"],
        logger=loggers,
        callbacks=(
            [early_stop_callback, model_checkpoint_callback, lr_monitor_callback]
            + (extra_callbacks or [])))

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=config["learning_rate"],
        hidden_size=config["hidden_size"],
        attention_head_size=config["hidden_size"] // 16,
        hidden_continuous_size=config["hidden_size"] // 4,
        dropout=config["dropout"],
        output_size=config["output_size"],
        loss=chosen_loss,
        optimizer=config["optimizer"],
        reduce_on_plateau_patience=config["reduce_on_plateau_patience"])

    print("Starting model training...")

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    return trainer, tft