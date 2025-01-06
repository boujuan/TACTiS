import argparse
import math
import os
import random
import warnings
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import DataEntry, MetaData, Dataset, ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.transform import AddObservedValuesIndicator, InstanceSplitter, ExpectedNumInstanceSampler, Chain
from typing import List, Tuple, Optional, Iterator
from gluonts.dataset.common import DataEntry, MetaData, Dataset, ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository._tsf_datasets import Dataset as MonashDataset
from gluonts.dataset.repository._tsf_datasets import datasets as monash_datasets
from gluonts.dataset.repository.datasets import (
    dataset_recipes,
    default_dataset_path,
    generate_forecasting_dataset,
    get_dataset,
    partial,
)
from gluonts.transform import AddObservedValuesIndicator
from gluonts.transform import AddObservedValuesIndicator, CDFtoGaussianTransform, Chain, InstanceSampler, InstanceSplitter, RenameFields, ValidationSplitSampler, TestSplitSampler, cdf_to_gaussian_forward_transform
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import maybe_len
from gluonts.transform import SelectFields, Transformation
from gluonts.torch.batchify import batchify
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.util import copy_parameters
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.model.predictor import Predictor
from gluonts.evaluation import Evaluator, MultivariateEvaluator, backtest
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.env import env
from gluonts.model.forecast import SampleForecast, QuantileForecast
from gluonts.model.forecast_generator import DistributionForecastGenerator, SampleForecastGenerator
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.forecast import DistributionForecast
from gluonts.torch.distributions import DistributionOutput, DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import copy_parameters
from gluonts.transform import AddObservedValuesIndicator, ExpectedNumInstanceSampler, InstanceSplitter, Transformation
from gluonts.torch.batchify import batchify
from gluonts.torch.distributions import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import copy_parameters
from gluonts.transform import AddObservedValuesIndicator, ExpectedNumInstanceSampler, InstanceSplitter, Transformation
from tactis.gluon.estimator import TACTiSEstimator
from tactis.gluon.trainer import Trainer
from tactis.gluon.utils import get_module, set_seed
from tactis.model.utils import check_memory, occupy_memory
import itertools
import gc
import time
from tqdm import tqdm
from pathlib import Path
from tactis.gluon.dataset import ParquetDataset

warnings.simplefilter(action="ignore", category=FutureWarning)

# Constants for the dataset
DATA_FILE = "test_juan/SMARTEOLE_WakeSteering_SCADA_1minData_normalized.parquet"
FREQ = "1min"  # Adjust if your data has a different frequency
PREDICTION_LENGTH = 30  # How far ahead to predict
CONTEXT_LENGTH_MULTIPLIER = 2  # How much history to use relative to prediction length
CONTEXT_LENGTH = PREDICTION_LENGTH * CONTEXT_LENGTH_MULTIPLIER
# Define the names of the features in your dataset
CONTINUITY_GROUP = "continuity_group"
ND_COS_FEATURES = [f"nd_cos_{i:03d}" for i in range(1, 8)]
ND_SIN_FEATURES = [f"nd_sin_{i:03d}" for i in range(1, 8)]
WS_HORZ_FEATURES = [f"ws_horz_{i:03d}" for i in range(1, 8)]
WS_VERT_FEATURES = [f"ws_vert_{i:03d}" for i in range(1, 8)]
TARGET_FEATURES = WS_HORZ_FEATURES + WS_VERT_FEATURES
ALL_STATIC_FEATURES = (
    [CONTINUITY_GROUP] + ND_COS_FEATURES + ND_SIN_FEATURES
)  # Features that don't vary over time
ALL_DYNAMIC_FEATURES = WS_HORZ_FEATURES + WS_VERT_FEATURES  # Features that vary over time
ALL_FEATURES = ALL_STATIC_FEATURES + ALL_DYNAMIC_FEATURES
NUM_SERIES = len(TARGET_FEATURES)

# Add a helper function to create a mask for the target features
def create_mask(
    data: pd.DataFrame,
    static_features: List[str],
    dynamic_features: List[str],
    context_length: int,
    prediction_length: int,
) -> np.ndarray:
    """
    Create a mask for the target features.

    Args:
        data: The input data.
        static_features: The names of the static features.
        dynamic_features: The names of the dynamic features.
        context_length: The length of the context.
        prediction_length: The length of the prediction horizon.

    Returns:
        A mask for the target features.
    """
    mask = np.zeros((len(data), len(dynamic_features)), dtype=bool)
    mask[:context_length, :] = True  # Keep context_length timesteps for all features
    mask[context_length : context_length + prediction_length, :] = False
    return mask

# Add these constants at the top
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = os.cpu_count()
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-8
DEFAULT_CLIP_GRADIENT = 1e2
DEFAULT_EARLY_STOPPING_EPOCHS = 20
DEFAULT_LOSS_NORMALIZATION = "series"
DEFAULT_FLOW_SERIES_EMBEDDING_DIM = 16
DEFAULT_COPULA_SERIES_EMBEDDING_DIM = 16
DEFAULT_FLOW_INPUT_ENCODER_LAYERS = 1
DEFAULT_COPULA_INPUT_ENCODER_LAYERS = 1
DEFAULT_FLOW_ENCODER_NUM_LAYERS = 2
DEFAULT_FLOW_ENCODER_NUM_HEADS = 4
DEFAULT_FLOW_ENCODER_DIM = 64
DEFAULT_COPULA_ENCODER_NUM_LAYERS = 2
DEFAULT_COPULA_ENCODER_NUM_HEADS = 4
DEFAULT_COPULA_ENCODER_DIM = 64
DEFAULT_DECODER_NUM_HEADS = 4
DEFAULT_DECODER_NUM_LAYERS = 2
DEFAULT_DECODER_DIM = 64
DEFAULT_DECODER_MLP_LAYERS = 2
DEFAULT_DECODER_MLP_DIM = 64
DEFAULT_DECODER_RESOLUTION = 100
DEFAULT_DECODER_ATTENTION_MLP_CLASS = "gluon"
DEFAULT_DECODER_ACT = "relu"
DEFAULT_DSF_NUM_LAYERS = 2
DEFAULT_DSF_DIM = 64
DEFAULT_DSF_MLP_LAYERS = 2
DEFAULT_DSF_MLP_DIM = 64
DEFAULT_EXPERIMENT_MODE = "forecasting"
DEFAULT_DO_NOT_RESTRICT_TIME = False
DEFAULT_SKIP_BATCH_SIZE_SEARCH = False
DEFAULT_DO_NOT_RESTRICT_MEMORY = False
DEFAULT_DEVICE = "cpu"
DEFAULT_EVALUATE = False
DEFAULT_SEED = 42
DEFAULT_DATASET_PATH = "./SMARTEOLE_WakeSteering_SCADA_1minData_normalized.parquet"
DEFAULT_EPOCHS = 10
DEFAULT_NUM_SAMPLES_FOR_LOSS = 100
DEFAULT_FLOW_ENCODER_MLP_UNITS = 64
DEFAULT_FLOW_ENCODER_MLP_LAYERS = 2
DEFAULT_FLOW_ENCODER_NUM_LAYERS = 2
DEFAULT_FLOW_SERIES_EMBEDDING_DIM = 16
DEFAULT_COPULA_SERIES_EMBEDDING_DIM = 16
DEFAULT_FLOW_INPUT_ENCODER_LAYERS = 1
DEFAULT_COPULA_INPUT_ENCODER_LAYERS = 1
DEFAULT_FLOW_ENCODER_NUM_LAYERS = 1

def main(
    dataset_path: str,
    checkpoint_dir: str,
    load_checkpoint: Optional[str],
    batch_size: int,
    num_workers: int,
    training_num_batches_per_epoch: int,
    learning_rate: float,
    weight_decay: float,
    clip_gradient: float,
    early_stopping_epochs: int,
    loss_normalization: str,
    flow_series_embedding_dim: int,
    copula_series_embedding_dim: int,
    flow_input_encoder_layers: int,
    copula_input_encoder_layers: int,
    flow_encoder_num_layers: int,
    flow_encoder_num_heads: int,
    flow_encoder_dim: int,
    copula_encoder_num_layers: int,
    copula_encoder_num_heads: int,
    copula_encoder_dim: int,
    decoder_num_heads: int,
    decoder_num_layers: int,
    decoder_dim: int,
    decoder_mlp_layers: int,
    decoder_mlp_dim: int,
    decoder_resolution: int,
    decoder_attention_mlp_class: str,
    decoder_act: str,
    dsf_num_layers: int,
    dsf_dim: int,
    dsf_mlp_layers: int,
    dsf_mlp_dim: int,
    experiment_mode: str,
    do_not_restrict_time: bool,
    skip_batch_size_search: bool,
    do_not_restrict_memory: bool,
    device: str,
    evaluate: bool,
    seed: int,
    epochs: int,
    num_samples_for_loss: int,
    flow_encoder_mlp_units: int,
    flow_encoder_mlp_layers: int,
):
    # Set the seed for reproducibility
    set_seed(seed)

    # --- Dataset ---
    dataset_path = Path(dataset_path)
    assert dataset_path.exists(), f"Dataset path {dataset_path} does not exist."

    train_ds = ParquetDataset(file_path=dataset_path, freq=FREQ)
    test_ds = ParquetDataset(file_path=dataset_path, freq=FREQ)

    # --- Multivariate Grouper ---
    # Assuming MultivariateGrouper is adapted to handle the new dataset format
    metadata = MetaData(
        freq=FREQ,
        target=None,  # No specific target field in this context
        feat_static_cat=[],
        feat_static_real=[],
        feat_dynamic_real=ALL_DYNAMIC_FEATURES,
        feat_dynamic_cat=[],
        prediction_length=PREDICTION_LENGTH,
    )

    train_grouper = MultivariateGrouper(
        max_target_dim=min(NUM_SERIES, 100),  # Adjust as needed
        num_test_dates=1,  # Assuming single time series, adjust if needed
        train_sampler=InstanceSampler(
            is_train=True,
            past_length=CONTEXT_LENGTH,
            future_length=PREDICTION_LENGTH,
            time_series_fields=[FieldName.TARGET] + ALL_DYNAMIC_FEATURES,
        ),
    )

    test_grouper = MultivariateGrouper(
        max_target_dim=min(NUM_SERIES, 100),  # Adjust as needed
        num_test_dates=1,  # Assuming single time series, adjust if needed
        train_sampler=InstanceSampler(
            is_train=False,
            past_length=CONTEXT_LENGTH,
            future_length=PREDICTION_LENGTH,
            time_series_fields=[FieldName.TARGET] + ALL_DYNAMIC_FEATURES,
        ),
    )

    train_mv_list = list(train_grouper(iter(train_ds)))
    test_mv_list = list(test_grouper(iter(test_ds)))

    train_ds = ListDataset(train_mv_list, freq=FREQ)
    test_ds = ListDataset(test_mv_list, freq=FREQ)

    # --- Training parameters ---
    num_series = len(train_ds.data[0][FieldName.TARGET])
    # Adjust the following parameters according to your needs and computational resources
    epochs = epochs
    num_batches_per_epoch = training_num_batches_per_epoch
    batch_size = batch_size
    num_workers = num_workers
    checkpoint_dir = Path(checkpoint_dir)
    
    # --- TACTiS Estimator ---
    estimator = TACTiSEstimator(
        freq=FREQ,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        input_size=num_series,
        num_series=num_series,
        loss_normalization=loss_normalization,
        flow_series_embedding_dim=flow_series_embedding_dim,
        copula_series_embedding_dim=copula_series_embedding_dim,
        flow_input_encoder_layers=flow_input_encoder_layers,
        copula_input_encoder_layers=copula_input_encoder_layers,
        flow_encoder_num_layers=flow_encoder_num_layers,
        flow_encoder_num_heads=flow_encoder_num_heads,
        flow_encoder_dim=flow_encoder_dim,
        flow_encoder_mlp_units=flow_encoder_mlp_units,
        flow_encoder_mlp_layers=flow_encoder_mlp_layers,
        copula_encoder_num_layers=copula_encoder_num_layers,
        copula_encoder_num_heads=copula_encoder_num_heads,
        copula_encoder_dim=copula_encoder_dim,
        decoder_num_heads=decoder_num_heads,
        decoder_num_layers=decoder_num_layers,
        decoder_dim=decoder_dim,
        decoder_mlp_layers=decoder_mlp_layers,
        decoder_mlp_dim=decoder_mlp_dim,
        decoder_resolution=decoder_resolution,
        decoder_attention_mlp_class=decoder_attention_mlp_class,
        decoder_act=decoder_act,
        dsf_num_layers=dsf_num_layers,
        dsf_dim=dsf_dim,
        dsf_mlp_layers=dsf_mlp_layers,
        dsf_mlp_dim=dsf_mlp_dim,
        num_parallel_samples=num_samples_for_loss,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        trainer=Trainer(
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clip_gradient=clip_gradient,
            num_batches_per_epoch=num_batches_per_epoch,
            checkpoint_dir=checkpoint_dir,
            load_checkpoint=load_checkpoint,
            early_stopping_epochs=early_stopping_epochs,
            do_not_restrict_time=do_not_restrict_time,
            skip_batch_size_search=skip_batch_size_search,
        ),
    )

    # --- Train ---
    if not evaluate:
        predictor = estimator.train(train_ds, num_workers=num_workers, shuffle_buffer_length=1024)
    else:
        predictor = TACTiSPredictionNetwork.load(checkpoint_dir / "model")

    # --- Evaluation ---
    if evaluate:
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds,
            predictor=predictor,
            num_samples=num_samples_for_loss,
        )
        forecasts = list(tqdm(forecast_it, total=len(test_ds)))
        tss = list(ts_it)

        evaluator = MultivariateEvaluator(
            quantiles=[0.1, 0.5, 0.9],
            target_agg_funcs={},
        )
        agg_metrics, item_metrics = evaluator(tss, forecasts)

        print(f"CRPS: {agg_metrics['mean_wQuantileLoss']}")
        print(f"ND: {agg_metrics['ND']}")
        print(f"NRMSE: {agg_metrics['NRMSE']}")
        print(f"RMSE: {agg_metrics['RMSE']}")

        # --- Plotting ---
        for target, forecast in zip(tss, forecasts):
            plt.figure(figsize=(12, 5))
            target = target.to_frame() if isinstance(target, pd.Series) else target
            target[-4 * PREDICTION_LENGTH :].plot(label="target")
            forecast.plot(color="g", prediction_intervals=[50.0, 90.0])
            plt.legend()
            plt.show()

def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to load.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--training_num_batches_per_epoch",
        type=int,
        default=100,
        help="Number of batches per epoch during training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay for training.",
    )
    parser.add_argument(
        "--clip_gradient",
        type=float,
        default=DEFAULT_CLIP_GRADIENT,
        help="Gradient clipping value.",
    )
    parser.add_argument(
        "--early_stopping_epochs",
        type=int,
        default=DEFAULT_EARLY_STOPPING_EPOCHS,
        help="Number of epochs for early stopping.",
    )
    parser.add_argument(
        "--loss_normalization",
        type=str,
        default=DEFAULT_LOSS_NORMALIZATION,
        choices=["batch", "series", "none"],
        help="Normalization method for loss.",
    )
    parser.add_argument(
        "--flow_series_embedding_dim",
        type=int,
        default=DEFAULT_FLOW_SERIES_EMBEDDING_DIM,
        help="Embedding dimension for flow series.",
    )
    parser.add_argument(
        "--copula_series_embedding_dim",
        type=int,
        default=DEFAULT_COPULA_SERIES_EMBEDDING_DIM,
        help="Embedding dimension for copula series.",
    )
    parser.add_argument(
        "--flow_input_encoder_layers",
        type=int,
        default=DEFAULT_FLOW_INPUT_ENCODER_LAYERS,
        help="Number of layers for flow input encoder.",
    )
    parser.add_argument(
        "--copula_input_encoder_layers",
        type=int,
        default=DEFAULT_COPULA_INPUT_ENCODER_LAYERS,
        help="Number of layers for copula input encoder.",
    )
    parser.add_argument(
        "--flow_encoder_num_layers",
        type=int,
        default=DEFAULT_FLOW_ENCODER_NUM_LAYERS,
        help="Number of layers for flow encoder.",
    )
    parser.add_argument(
        "--flow_encoder_num_heads",
        type=int,
        default=DEFAULT_FLOW_ENCODER_NUM_HEADS,
        help="Number of heads for flow encoder.",
    )
    parser.add_argument(
        "--flow_encoder_dim",
        type=int,
        default=DEFAULT_FLOW_ENCODER_DIM,
        help="Dimension for flow encoder.",
    )
    parser.add_argument(
        "--copula_encoder_num_layers",
        type=int,
        default=DEFAULT_COPULA_ENCODER_NUM_LAYERS,
        help="Number of layers for copula encoder.",
    )
    parser.add_argument(
        "--copula_encoder_num_heads",
        type=int,
        default=DEFAULT_COPULA_ENCODER_NUM_HEADS,
        help="Number of heads for copula encoder.",
    )
    parser.add_argument(
        "--copula_encoder_dim",
        type=int,
        default=DEFAULT_COPULA_ENCODER_DIM,
        help="Dimension for copula encoder.",
    )
    parser.add_argument(
        "--decoder_num_heads",
        type=int,
        default=DEFAULT_DECODER_NUM_HEADS,
        help="Number of heads for decoder.",
    )
    parser.add_argument(
        "--decoder_num_layers",
        type=int,
        default=DEFAULT_DECODER_NUM_LAYERS,
        help="Number of layers for decoder.",
    )
    parser.add_argument(
        "--decoder_dim",
        type=int,
        default=DEFAULT_DECODER_DIM,
        help="Dimension for decoder.",
    )
    parser.add_argument(
        "--decoder_mlp_layers",
        type=int,
        default=DEFAULT_DECODER_MLP_LAYERS,
        help="Number of layers for decoder MLP.",
    )
    parser.add_argument(
        "--decoder_mlp_dim",
        type=int,
        default=DEFAULT_DECODER_MLP_DIM,
        help="Dimension for decoder MLP.",
    )
    parser.add_argument(
        "--decoder_resolution",
        type=int,
        default=DEFAULT_DECODER_RESOLUTION,
        help="Resolution for decoder.",
    )
    parser.add_argument(
        "--decoder_attention_mlp_class",
        type=str,
        default=DEFAULT_DECODER_ATTENTION_MLP_CLASS,
        choices=["gluon", "simple"],
        help="Class for decoder attention MLP.",
    )
    parser.add_argument(
        "--decoder_act",
        type=str,
        default=DEFAULT_DECODER_ACT,
        choices=["relu", "elu", "softplus", "tanh", "sigmoid", "leakyrelu"],
        help="Activation function for decoder.",
    )
    parser.add_argument(
        "--dsf_num_layers",
        type=int,
        default=DEFAULT_DSF_NUM_LAYERS,
        help="Number of layers for DSF.",
    )
    parser.add_argument(
        "--dsf_dim",
        type=int,
        default=DEFAULT_DSF_DIM,
        help="Dimension for DSF.",
    )
    parser.add_argument(
        "--dsf_mlp_layers",
        type=int,
        default=DEFAULT_DSF_MLP_LAYERS,
        help="Number of layers for DSF MLP.",
    )
    parser.add_argument(
        "--dsf_mlp_dim",
        type=int,
        default=DEFAULT_DSF_MLP_DIM,
        help="Number of units for DSF MLP.",
    )
    parser.add_argument(
        "--dsf_mlp_units",
        type=int,
        default=64,
        help="Number of units for DSF MLP.",
    )

    # Experiment mode
    parser.add_argument(
        "--experiment_mode",
        type=str,
        default="forecasting",
        choices=["forecasting", "interpolation"],
        help="Experiment mode.",
    )

    # Time restriction
    parser.add_argument(
        "--do_not_restrict_time",
        action="store_true",
        default=False,
        help="When enabled, training time is not restricted.",
    )

    # Batch size search
    parser.add_argument(
        "--skip_batch_size_search",
        action="store_true",
        default=False,
        help="When enabled, batch size search is skipped.",
    )

    # Memory restriction
    parser.add_argument(
        "--do_not_restrict_memory",
        action="store_true",
        default=False,
        help="When enabled, memory is not restricted.",
    )

    # CPU/GPU
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for training (cpu or cuda).",
    )

    # Flag for evaluation (either NLL or sampling and metrics)
    # A checkpoint must be provided for evaluation
    # Note evaluation is only supported after training the model in both phases.
    parser.add_argument("--evaluate", action="store_true", help="Evaluate for NLL and metrics.")

    # Add seed argument
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # Add dataset path argument
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./SMARTEOLE_WakeSteering_SCADA_1minData_normalized.parquet",  # Update with your default path
        help="Path to the Parquet dataset.",
    )

    # Add epochs argument
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for training.",
    )

    # Add num_samples_for_loss argument
    parser.add_argument(
        "--num_samples_for_loss",
        type=int,
        default=100,
        help="Number of samples for loss calculation.",
    )

    parser.add_argument(
        "--flow_encoder_mlp_units",
        type=int,
        default=64,
        help="Number of units for flow encoder MLP.",
    )
    parser.add_argument(
        "--flow_encoder_mlp_layers",
        type=int,
        default=2,
        help="Number of layers for flow encoder MLP.",
    )

    args = parser.parse_args()

    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA version:", torch.version.cuda)

    # Occupy GPU memory to avoid முடியவில்லை: CUDA error: out of memory
    if args.device != "cpu":
        occupy_memory(args.device)

    main(**vars(args))
