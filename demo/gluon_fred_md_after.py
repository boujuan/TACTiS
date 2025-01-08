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

history_factor = 3
metadata, train_data, valid_data = generate_hp_search_datasets("fred_md", history_factor)

# %%
#######################
# CHECKPOINT 2: Creating the GluonTS Estimator object
#######################

estimator = TACTiSEstimator(
    model_parameters={
        # Marginal CDF Encoder time series embedding dimensions
        "flow_series_embedding_dim": 5,
        # Attentional Copula Encoder time series embedding dimensions
        "copula_series_embedding_dim": 8,
        # Marginal CDF Encoder input encoder layers
        "flow_input_encoder_layers": 4,
        # Attentional Copula Encoder input encoder layers
        "copula_input_encoder_layers": 7,
        "input_encoding_normalization": True,
        # Data Normalization
        "data_normalization": "standardization",
        "loss_normalization": "series",
        "bagging_size": 20,
        "positional_encoding": {
            "dropout": 0.0,
        },
        # Marginal CDF Encoder parameters
        "flow_temporal_encoder": {
            # Marginal CDF Encoder number of transformer layers pairs
            "attention_layers": 1,
            # Marginal CDF Encoder transformer number of heads
            "attention_heads": 4,
            # Marginal CDF Encoder transformer embedding size (per head)
            "attention_dim": 512,
            "attention_feedforward_dim": 512,
            "dropout": 0.0,
        },
        # Attentional Copula Encoder parameters
        "copula_temporal_encoder": {
            # Attentional Copula Encoder number of transformer layers pairs
            "attention_layers": 5,
            # Attentional Copula Encoder transformer number of heads
            "attention_heads": 3,
            # Attentional Copula Encoder transformer embedding size (per head)
            "attention_dim": 8,
            "attention_feedforward_dim": 8,
            "dropout": 0.0,
        },
        "copula_decoder": {
            "min_u": 0.05,
            "max_u": 0.95,
            "attentional_copula": {
                # Decoder transformer number of heads
                "attention_heads": 4,
                # Decoder transformer number of layers
                "attention_layers": 7,
                # Decoder transformer embedding size (per head)
                "attention_dim": 64,
                # Decoder MLP number of layers
                "mlp_layers": 2,
                # Decoder MLP hidden dimensions
                "mlp_dim": 48,
                # Decoder number of bins in conditional distribution
                "resolution": 50,
                "activation_function": "relu"
            },
            "dsf_marginal": {
                # Decoder MLP number of layers
                "mlp_layers": 2,
                # Decoder MLP hidden dimensions
                "mlp_dim": 48,
                # Decoder DSF number of layers
                "flow_layers": 6,
                # Decoder DSF hidden dimensions
                "flow_hid_dim": 48,
            },
        },
    },
    num_series=train_data.list_data[0]["target"].shape[0],
    history_length=history_factor * metadata.prediction_length,
    prediction_length=metadata.prediction_length,
    freq=metadata.freq,
    trainer=TACTISTrainer(
        epochs_phase_1=20,
        epochs_phase_2=20,
        batch_size=256,
        training_num_batches_per_epoch=512,
        learning_rate=1e-3,
        weight_decay=1e-3,  # Phase 1 weight decay
        clip_gradient=1e3,  # Phase 1 gradient clipping
        device=torch.device("cuda:0"),
        checkpoint_dir="checkpoints/fred_md_forecasting",
    ),
    cdf_normalization=False,
    num_parallel_samples=100,
)

# %%
#######################
# CHECKPOINT 3: Training the model
#######################

# model = estimator.train(
#     train_data,
#     valid_data,
#     num_workers=4,
#     prefetch_factor=2
# )

# %%
#######################
# CHECKPOINT 4: Loading and training with backtesting dataset
#######################

backtest_id = 3
metadata, backtest_train_data, backtest_valid_data = generate_prebacktesting_datasets(
    "fred_md", backtest_id, history_factor
)
_, _, backtest_test_data = generate_backtesting_datasets("fred_md", backtest_id, history_factor)

# # Train on backtesting dataset
# model = estimator.train(
#     backtest_train_data,
#     backtest_valid_data,
#     num_workers=4,
#     prefetch_factor=2
# )

# %%
#######################
# CHECKPOINT 5: Evaluating NLL on testing dataset
#######################

# Load the best checkpoints for each stage
checkpoint_dir = "checkpoints/fred_md_forecasting"
device = torch.device("cuda:0")

# Create a dummy network instance
trained_net = estimator.create_training_network(device)

# Load the best stage 1 checkpoint
best_stage_1_ckpt_path = f"{checkpoint_dir}/best_stage_1.pth.tar"
checkpoint_stage_1 = torch.load(best_stage_1_ckpt_path, map_location=device)
trained_net.load_state_dict(checkpoint_stage_1["model"])
estimator.trainer.load_checkpoint = best_stage_1_ckpt_path
nll_stage_1 = estimator.validate(backtest_test_data, backtesting=True)
print(f"NLL (Stage 1): {nll_stage_1.item()}")

# Load the best stage 2 checkpoint
best_stage_2_ckpt_path = f"{checkpoint_dir}/best_stage_2.pth.tar"
checkpoint_stage_2 = torch.load(best_stage_2_ckpt_path, map_location=device)
trained_net.load_state_dict(checkpoint_stage_2["model"], strict=False)
estimator.trainer.load_checkpoint = best_stage_2_ckpt_path
nll_stage_2 = estimator.validate(backtest_test_data, backtesting=True)
print(f"NLL (Stage 2): {nll_stage_2.item()}")

# %%
#######################
# CHECKPOINT 6: Computing metrics for testing dataset
#######################

# Use the stage 2 checkpoint for metrics and plotting
model = trained_net

# Enable copula in model parameters
estimator.model_parameters["skip_copula"] = False

# Create predictor
transformation = estimator.create_transformation()
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

# %%
#######################
# CHECKPOINT 8: Save the model
#######################

# Save the model
# torch.save(model, "checkpoints/fred_md_forecasting/model.pth")

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
        selection=[(0, 10), (0, 20), (0, 30), (0, 40)],
        history_length=history_length,
        savefile=f"{savefile_prefix}_forecasts.png"
    )

analyze_and_plot_forecasts(forecasts, targets, estimator.history_length, "checkpoints/fred_md_forecasting/analysis")

# %%
