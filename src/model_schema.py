from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Literal, Any, Dict


class TFTConfig(BaseModel):
    """
    Pydantic model for validating TFT experiment configurations. This model ensures
    that all required parameters are present, have the correct data types and values,
    and that all specified features are defined in the provided feature schema.
    It also includes validators for specific parameters to ensure they conform to
    available model options.
    """

    # file paths
    data_path: str = Field(description="Path to the input dataset CSV file.")
    mlflow_dir: str = Field(description="Directory for MLFlow logs.")
    tensorboard_dir: str = Field(description="Directory for TensorBoard logs.")
    model_checkpoint_dir: str = Field(description="Directory for model checkpoints.")

    # dataset and dataloader parameters
    max_encoder_length: int = Field(gt=0, description="Maximum length of the encoder sequence.")
    min_encoder_length: int = Field(gt=0, description="Minimum length of the encoder sequence.")
    max_prediction_length: int = Field(gt=0, description="Maximum length of the prediction sequence.")
    normalizer_method: str = Field(description="Method for target normalization (e.g., 'robust', 'standard').")
    normalizer_transformation: str = Field(description="Transformation for target normalization (e.g., 'log1p', 'none').")
    batch_size: int = Field(gt=0, description="Batch size for dataloaders.")

    # trainer parameters
    max_epochs: int = Field(gt=0, description="Maximum number of training epochs.")
    gradient_clip_val: float = Field(gt=0.0, description="Gradient clipping value.")
    accelerator: str = Field(description="Accelerator for training (e.g., 'cpu', 'gpu', 'auto').")
    devices: Literal[1] = Field(description="Number of devices to use for training.")
    fast_dev_run: bool = Field(description="Whether to run a fast development run.")

    # tft parameters
    learning_rate: float = Field(gt=0.0, lt=1.0, description="Learning rate for the optimizer.")
    hidden_size: int = Field(gt=0, description="Hidden size for the TFT model.")
    reduce_on_plateau_patience: int = Field(gt=0, description="Patience for ReduceLROnPlateau scheduler.")
    output_size: int = Field(gt=0, description="Output size (number of quantiles) for the TFT model.")
    dropout: float = Field(gt=0.0, lt=1.0, description="Dropout rate for the TFT model.")
    quantiles: list[float] = Field(description="List of quantiles to predict (e.g., [0.1, 0.25, 0.5, 0.75, 0.9]).")
    loss: str = Field(description="Loss function to use (e.g., 'quantile', 'mae', 'smape', 'rmse').")
    optimizer: str = Field(description="Optimizer to use (e.g., 'ranger').")

    # early stopping parameters
    monitor: str = Field(description="Metric to monitor for early stopping (e.g., 'val_loss').")
    min_delta: float = Field(gt=0.0, description="Minimum change in the monitored metric to qualify as an improvement.")
    patience: int = Field(gt=0, description="Number of epochs with no improvement after which training will be stopped.")
    mode: str = Field(description="Mode for early stopping (e.g., 'min', 'max').")

    # model variables
    target: Literal["quantity"] = Field(description="Name of the target variable in the dataset.")
    group_ids: list[Literal["stock_code"]] = Field(description="Name of the group identifier variable in the dataset.")
    time_idx: Literal["time_idx"] = Field(description="Name of the time index variable in the dataset.")
    static_categoricals: list[str] = Field(description="List of static categorical feature names.")
    static_reals: list[str] = Field(description="List of static real feature names.")
    time_varying_known_categoricals: list[str] = Field(description="List of time-varying known categorical feature names.")
    time_varying_known_reals: list[str] = Field(description="List of time-varying known real feature names.")
    time_varying_unknown_categoricals: list[str] = Field(description="List of time-varying unknown categorical feature names.")
    time_varying_unknown_reals: list[str] = Field(description="List of time-varying unknown real feature names.")

    # logger parameters
    experiment_name: str = Field(description="Name of the MLFlow experiment.")
    log_model: bool = Field(description="Whether to log the model in MLFlow.")

    @model_validator(mode="after")
    def verify_encoder_windows(self) -> "TFTConfig":
        """
        Validates that min_encoder_length is not greater than max_encoder_length.
        """
        if self.min_encoder_length > self.max_encoder_length:
            raise ValueError(f"min_encoder_length ({self.min_encoder_length}) cannot be greater than max_encoder_length ({self.max_encoder_length}).")
        return self

    @model_validator(mode="after")
    def verify_output_size(self) -> "TFTConfig":
        """
        Validates that output_size matches the length of the quantiles list when using QuantileLoss.
        """
        if self.loss.lower() == "quantile" and self.output_size != len(self.quantiles):
            raise ValueError(f"Output size ({self.output_size}) must match the number of quantiles ({len(self.quantiles)}) when using QuantileLoss.")
        return self

    @field_validator("loss")
    def validate_loss(cls, v: str) -> str:
        """
        Validates that the specified loss function is one of the accepted options.
        """
        valid_losses = {"quantile", "mae", "smape", "rmse"}
        if v.lower() not in valid_losses:
            raise ValueError(f"Invalid loss function: '{v}'. Choose from {', '.join(valid_losses)}.")
        return v.lower()

    @field_validator("optimizer")
    def validate_optimizer(cls, v: str) -> str:
        """
        Validates that the specified optimizer is one of the accepted options.
        """
        valid_optimizers = {"ranger"}
        if v.lower() not in valid_optimizers:
            raise ValueError(f"Invalid optimizer: '{v}'. Choose from {', '.join(valid_optimizers)}.")
        return v.lower()
    
    @field_validator("mode")
    def validate_mode(cls, v: str) -> str:
        """
        Validates that the specified early stopping mode is one of the accepted options.
        """
        valid_modes = {"min", "max"}
        if v.lower() not in valid_modes:
            raise ValueError(f"Invalid early stopping mode: '{v}'. Choose from {', '.join(valid_modes)}.")
        return v.lower()
    
    @field_validator("accelerator")
    def validate_accelerator(cls, v: str) -> str:
        """
        Validates that the specified accelerator is one of the accepted options.
        """
        valid_accelerators = {"cpu", "gpu", "auto"}
        if v.lower() not in valid_accelerators:
            raise ValueError(f"Invalid accelerator: '{v}'. Choose from {', '.join(valid_accelerators)}.")
        return v.lower()
    
    @field_validator("normalizer_method")
    def validate_normalizer_method(cls, v: str) -> str:
        """
        Validates that the specified normalizer method is one of the accepted options.
        """
        valid_methods = {"standard", "robust", "minmax"}
        if v.lower() not in valid_methods:
            raise ValueError(f"Invalid normalizer method: '{v}'. Choose from {', '.join(valid_methods)}.")
        return v.lower()
    
    @field_validator("normalizer_transformation")
    def validate_normalizer_transformation(cls, v: str) -> str:
        """
        Validates that the specified normalizer transformation is one of the accepted options.
        """
        valid_transformations = {"log1p", "none"}
        if v.lower() not in valid_transformations:
            raise ValueError(f"Invalid normalizer transformation: '{v}'. Choose from {', '.join(valid_transformations)}.")
        return v.lower()
    
    @field_validator("monitor")
    def validate_monitor(cls, v: str) -> str:
        """
        Validates that the specified loss monitor is one of the accepted options.
        """
        valid_monitors = {"val_loss", "train_loss"}
        if v.lower() not in valid_monitors:
            raise ValueError(f"Invalid monitor: '{v}'. Choose from {', '.join(valid_monitors)}.")
        return v.lower()
    
    @model_validator(mode="after")
    def validate_features_against_schema(self, info: Any) -> "TFTConfig":
        """
        Validates that all features specified in the config are in the feature schema
        """
        context: Dict[str, Any] = info.context if info and info.context else {}
        schema = context.get("schema")

        if schema is None:
            return self
        
        valid_features = set(schema.keys())
        config_features = set(
            self.static_categoricals +
            self.static_reals +
            self.time_varying_known_categoricals +
            self.time_varying_known_reals +
            self.time_varying_unknown_categoricals +
            self.time_varying_unknown_reals)

        invalid_features = config_features - valid_features
        
        if invalid_features:
            raise ValueError(
                f"[ERROR] Requested features missing from feature schema:\n"
                f"  ❌ {', '.join(invalid_features)}\n")
        
        return self
    
    @field_validator("static_categoricals")
    def enforce_static_categoricals(cls, v: list[str]) -> list[str]:
        """
        Ensures "stock_code" is included in static_categoricals.
        """
        static_categoricals = set(v)
        static_categoricals.add("stock_code")
        return list(static_categoricals)
    
    @field_validator("time_varying_known_reals")
    def enforce_time_varying_known_reals(cls, v: list[str]) -> list[str]:
        """
        Ensures "time_idx" and "price_scaled" are included in time_varying_known_reals.
        """
        time_varying_known_reals = set(v)
        time_varying_known_reals.update(["time_idx", "price_scaled"])
        return list(time_varying_known_reals)
    
    @field_validator("time_varying_unknown_reals")
    def enforce_time_varying_unknown_reals(cls, v: list[str]) -> list[str]:
        """
        Ensures "quantity" is included in time_varying_unknown_reals.
        """
        time_varying_unknown_reals = set(v)
        time_varying_unknown_reals.add("quantity")
        return list(time_varying_unknown_reals)
    
    @field_validator("static_categoricals",
                     "static_reals",
                     "time_varying_known_categoricals",
                     "time_varying_known_reals",
                     "time_varying_unknown_categoricals",
                     "time_varying_unknown_reals")
    def deduplicate_and_sort_features(cls, v: list[str]) -> list[str]:
        """
        Deduplicates and sorts feature lists for consistency.
        """
        return sorted(list(set(v)))