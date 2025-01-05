# Install tactis if you haven't installed yet
# !pip install tactis[research]

import torch
from tactis.gluon.estimator import TACTiSEstimator
from tactis.gluon.trainer import TACTISTrainer
from tactis.gluon.dataset import generate_hp_search_datasets, generate_prebacktesting_datasets, generate_backtesting_datasets
from tactis.gluon.metrics import compute_validation_metrics
from tactis.gluon.plots import plot_four_forecasts
from gluonts.evaluation.backtest import make_evaluation_predictions
import warnings
import random

warnings.filterwarnings("ignore", category=FutureWarning)

# %%
#######################
# CHECKPOINT 1: Loading the dataset for hyperparameter search
#######################

history_factor = 1
metadata, train_data, valid_data = generate_hp_search_datasets("fred_md", history_factor)

# %%
#######################
# CHECKPOINT 2: Creating the GluonTS Estimator object
#######################

estimator = TACTiSEstimator(
    model_parameters = {
        "flow_series_embedding_dim": 5,
        "copula_series_embedding_dim": 5,
        "flow_input_encoder_layers": 2,
        "copula_input_encoder_layers": 2,
        "input_encoding_normalization": True,
        "data_normalization": "standardization",
        "loss_normalization": "series",
        "bagging_size": 20,
        "positional_encoding":{
            "dropout": 0.0,
        },
        "flow_temporal_encoder":{
            "attention_layers": 2,
            "attention_heads": 1,
            "attention_dim": 16,
            "attention_feedforward_dim": 16,
            "dropout": 0.0,
        },
        "copula_temporal_encoder":{
            "attention_layers": 2,
            "attention_heads": 1,
            "attention_dim": 16,
            "attention_feedforward_dim": 16,
            "dropout": 0.0,
        },
        "copula_decoder":{
            "min_u": 0.05,
            "max_u": 0.95,
            "attentional_copula": {
                "attention_heads": 3,
                "attention_layers": 1,
                "attention_dim": 8,
                "mlp_layers": 2,
                "mlp_dim": 48,
                "resolution": 20,
                "activation_function": "relu"
            },
            "dsf_marginal": {
                "mlp_layers": 2,
                "mlp_dim": 48,
                "flow_layers": 2,
                "flow_hid_dim": 8,
            },
        },
    },
    num_series = train_data.list_data[0]["target"].shape[0],
    history_length = history_factor * metadata.prediction_length,
    prediction_length = metadata.prediction_length,
    freq = metadata.freq,
    trainer = TACTISTrainer(
        epochs_phase_1 = 20,
        epochs_phase_2 = 20,
        batch_size = 256,
        training_num_batches_per_epoch = 512,
        learning_rate = 1e-3,
        weight_decay = 1e-4,
        clip_gradient = 1e3,
        device = torch.device("cuda:0"),  # Modify device as needed
        checkpoint_dir = "checkpoints/fred_md_forecasting",
    ),
    cdf_normalization = False,
    num_parallel_samples = 100,
)
# %%
#######################
# CHECKPOINT 3: Training the model
#######################

model = estimator.train(
    train_data, 
    valid_data,
    num_workers=4,
    prefetch_factor=2
)

# %%
#######################
# CHECKPOINT 4: Loading and training with backtesting dataset
#######################

backtest_id = 3
metadata, backtest_train_data, backtest_valid_data = generate_prebacktesting_datasets("fred_md", backtest_id, history_factor)
_, _, backtest_test_data = generate_backtesting_datasets("fred_md", backtest_id, history_factor)

# Train on backtesting dataset
model = estimator.train(
    backtest_train_data, 
    backtest_valid_data,
    num_workers=4,
    prefetch_factor=2
)

# %%
#######################
# CHECKPOINT 5: Evaluating NLL on testing dataset
#######################

nll = estimator.validate(backtest_test_data, backtesting=True)
print("NLL:", nll.item())

# %%
#######################
# CHECKPOINT 6: Computing metrics for testing dataset
#######################

# Enable copula in model parameters
estimator.model_parameters["skip_copula"] = False

# Create predictor
transformation = estimator.create_transformation()
device = estimator.trainer.device
predictor = estimator.create_predictor(
    transformation=transformation,
    trained_network=model,
    device=device,
    experiment_mode="forecasting",
    history_length=history_factor * metadata.prediction_length,
)

# Compute metrics
predictor.batch_size = 16
metrics, ts_wise_metrics = compute_validation_metrics(
    predictor=predictor,
    dataset=backtest_test_data,
    window_length=estimator.history_length + estimator.prediction_length,
    prediction_length=estimator.prediction_length,
    num_samples=100,
    split=False,
)

print("Metrics:", metrics)

# %%
#######################
# CHECKPOINT 7: Generate forecasts (optional)
#######################

forecast_it, ts_it = make_evaluation_predictions(
    dataset=backtest_test_data, predictor=predictor, num_samples=100
)
forecasts = list(forecast_it)
targets = list(ts_it)

# Note: Plotting functionality removed since this is meant for HPC
# If you need plots, save the forecasts and targets to analyze later

# %%
#######################
# CHECKPOINT 8: Save the model
#######################

# Save the model
torch.save(model, "checkpoints/fred_md_forecasting/model.pth")

# %%
#######################
# CHECKPOINT 9: Analyse the forecast
#######################

def analyze_and_plot_forecasts(forecasts, targets, history_length, savefile_prefix):
    """
    Analyzes the forecasts and targets, and generates plots.

    Parameters:
    forecasts: List of forecasts generated by the model.
    targets: List of actual target values.
    history_length: Length of the historical data used for forecasting.
    savefile_prefix: Prefix for the saved plot files.
    """
    # Select four random series to plot
    selection = []
    for _ in range(4):
        selection.append((0, random.randint(0, len(forecasts) - 1)))

    # Generate and save the plots
    plot_four_forecasts(
        forecasts=forecasts,
        targets=targets,
        selection=selection,
        history_length=history_length,
        savefile=f"{savefile_prefix}_forecasts.png"
    )

analyze_and_plot_forecasts(forecasts, targets, estimator.history_length, "checkpoints/fred_md_forecasting/analysis")

# %%
