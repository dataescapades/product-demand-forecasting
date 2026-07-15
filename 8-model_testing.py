# import libraries
import yaml

import numpy as np
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer
from src.tft_pipeline import (
    validate_config_queue,
    load_and_prep_data,
    create_datasets, create_dataloaders)


def main():
    with open('model_config.yaml', 'r') as f:
        master_yaml = yaml.safe_load(f)

    schema = master_yaml["FEATURE_SCHEMA"]
    config = ["final_experiment"]
    model_ckpt_name = "Best_model-final_experiment.ckpt"

    validated_configs = validate_config_queue(
        run_queue=config,
        yaml=master_yaml,
        schema=schema,
        model_defaults=master_yaml["MODEL_DEFAULTS"])

    config = validated_configs["final_experiment"]

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
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        training=training,
        validation=validation,
        testing=testing,
        config=config)

    model = TemporalFusionTransformer.load_from_checkpoint(
        checkpoint_path=f"{config['model_checkpoint_dir']}/{model_ckpt_name}")

    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"])

    trainer.test(model, dataloaders=test_dataloader)

    outputs = model.predict(test_dataloader, mode="raw", return_x=True)
    predictions = outputs.output.prediction.detach().cpu().numpy()
    actuals = outputs.x["decoder_target"].detach().cpu().numpy()

    q90_idx = model.loss.quantiles.index(0.9)
    q90_predictions = predictions[:, :, q90_idx]

    coverage = np.mean((actuals <= q90_predictions))
    print(f"Coverage of 90th percentile predictions: {coverage:.2%}")

    insufficient_coverage = actuals - q90_predictions
    insufficient_coverage = insufficient_coverage[insufficient_coverage > 0]
    print(f"Mean insufficient coverage magnitude: {np.mean(insufficient_coverage):.2f}")
    print(f"Median insufficient coverage: {np.median(insufficient_coverage):.2f}")

if __name__ == "__main__":
    main()