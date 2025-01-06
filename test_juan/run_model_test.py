import pandas as pd
import numpy as np
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
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.env import env
from pts import Trainer
from pts.model import PyTorchEstimator
from pts.model.estimator import TrainOutput
from pts.model.utils import get_module_forward_input_names
from pts.dataset.loader import TransformedDataset, TransformedIterableDataset
from tactis.gluon.network import (
    TACTiSPredictionNetwork,
    TACTiSTrainingNetwork,
    TACTiSPredictionNetworkInterpolation,
)
from tactis.gluon.metrics import compute_validation_metrics, SplitValidationTransform
from tactis.gluon.trainer import TACTISTrainer
from tactis.gluon.estimator import TACTiSEstimator
from tactis.gluon.utils import (
    save_checkpoint,
    load_checkpoint,
    set_seed,
)
import random
import argparse
import warnings
from gluonts.evaluation.backtest import make_evaluation_predictions
from tactis.gluon.plots import plot_four_forecasts
from tactis.gluon.metrics import compute_validation_metrics, compute_validation_metrics_interpolation
from tactis.model.utils import check_memory
import itertools
import gc
import time
from tqdm import tqdm

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
DEFAULT_ACCUMULATION_STEPS = 8

class ParquetDataset(Dataset):
    def __init__(self, parquet_file: str, freq: str):
        # Read the parquet file
        self.data = pd.read_parquet(parquet_file)
        
        # Ensure the index is datetime
        if not isinstance(self.data.index, pd.DatetimeIndex):
            if 'time' in self.data.columns:
                self.data.set_index('time', inplace=True)
            self.data.index = pd.to_datetime(self.data.index)
        
        self.freq = freq

    def __iter__(self) -> Iterator[DataEntry]:
        for i in range(len(self.data)):
            start_time_data = self.data.index[i]

            data_entry = {
                FieldName.START: pd.Timestamp(start_time_data, freq=self.freq),
                FieldName.TARGET: self.data[TARGET_FEATURES].iloc[i].values,
                FieldName.FEAT_STATIC_REAL: self.data[ALL_STATIC_FEATURES].iloc[i].values,
                FieldName.FEAT_DYNAMIC_REAL: self.data[ALL_DYNAMIC_FEATURES].iloc[i].values,
            }
            yield data_entry

    def __len__(self):
        return len(self.data)

def create_data_transformation(
    context_length: int,
    prediction_length: int,
) -> Transformation:
    return Chain(
        [
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=ExpectedNumInstanceSampler(
                    num_instances=1,
                    min_future=prediction_length,
                ),
                past_length=context_length,
                future_length=prediction_length,
                time_series_fields=[
                    FieldName.FEAT_DYNAMIC_REAL,
                    FieldName.OBSERVED_VALUES,
                ],
            ),
        ]
    )

def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    try:
        # Print initial GPU memory usage
        if torch.cuda.is_available() and not args.use_cpu:
            print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # Create dataset
        print("Creating dataset...")
        dataset = ParquetDataset(
            parquet_file=DATA_FILE,
            freq=FREQ,
        )
        print("Dataset created.")

        # Adjust batch size based on available memory
        if torch.cuda.is_available() and not args.use_cpu:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            free_memory = total_memory - torch.cuda.memory_allocated()
            # Adjust batch size based on available memory (this is a simple heuristic)
            adjusted_batch_size = min(args.batch_size, max(1, int(free_memory / (1024**3) * 32)))
            print(f"Adjusted batch size: {adjusted_batch_size}")
            args.batch_size = adjusted_batch_size

        # Create data transformation
        transformation = create_data_transformation(
            context_length=CONTEXT_LENGTH,
            prediction_length=PREDICTION_LENGTH,
        )

        # Split into train and validation sets (80-20 split)
        total_length = len(dataset)
        train_length = int(0.8 * total_length)
        print("Splitting dataset into train and validation sets...")
        print(f"Total length: {total_length}")
        print(f"Train length: {train_length}")

        train_data = ListDataset(
            list(dataset)[:train_length],
            freq=FREQ
        )

        valid_data = ListDataset(
            list(dataset)[train_length:],
            freq=FREQ
        )

        # Create the model parameters
        model_parameters = {
            "flow_series_embedding_dim": args.flow_series_embedding_dim,
            "copula_series_embedding_dim": args.copula_series_embedding_dim,
            "flow_input_encoder_layers": args.flow_input_encoder_layers,
            "copula_input_encoder_layers": args.copula_input_encoder_layers,
            "input_encoding_normalization": True,
            "data_normalization": "standardization",
            "loss_normalization": args.loss_normalization,
            "positional_encoding": {
                "dropout": 0.0,
            },
            "flow_temporal_encoder": {
                "attention_layers": args.flow_encoder_num_layers,
                "attention_heads": args.flow_encoder_num_heads,
                "attention_dim": args.flow_encoder_dim,
                "attention_feedforward_dim": args.flow_encoder_dim,
                "dropout": 0.0,
            },
            "copula_temporal_encoder": {
                "attention_layers": args.copula_encoder_num_layers,
                "attention_heads": args.copula_encoder_num_heads,
                "attention_dim": args.copula_encoder_dim,
                "attention_feedforward_dim": args.copula_encoder_dim,
                "dropout": 0.0,
            },
            "copula_decoder": {
                "min_u": 0.05,
                "max_u": 0.95,
                "attentional_copula": {
                    "attention_heads": args.decoder_num_heads,
                    "attention_layers": args.decoder_num_layers,
                    "attention_dim": args.decoder_dim,
                    "mlp_layers": args.decoder_mlp_layers,
                    "mlp_dim": args.decoder_mlp_dim,
                    "resolution": args.decoder_resolution,
                    "activation_function": args.decoder_act,
                },
                "dsf_marginal": {
                    "mlp_layers": args.dsf_mlp_layers,
                    "mlp_dim": args.dsf_mlp_dim,
                    "flow_layers": args.dsf_num_layers,
                    "flow_hid_dim": args.dsf_dim,
                },
            },
        }

        # Create the estimator with adjusted parameters
        print("Creating estimator...")
        estimator = TACTiSEstimator(
            model_parameters=model_parameters,
            num_series=NUM_SERIES,
            history_length=CONTEXT_LENGTH,
            prediction_length=PREDICTION_LENGTH,
            freq=FREQ,
            trainer=TACTISTrainer(
                epochs_phase_1=20,
                epochs_phase_2=20,
                batch_size=args.batch_size,
                training_num_batches_per_epoch=512,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                clip_gradient=args.clip_gradient,
                device=torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu"),
                checkpoint_dir=args.checkpoint_dir,
                seed=args.seed,
                load_checkpoint=args.load_checkpoint if hasattr(args, 'load_checkpoint') else None,
                early_stopping_epochs=args.early_stopping_epochs,
                do_not_restrict_time=args.do_not_restrict_time,
                skip_batch_size_search=args.skip_batch_size_search,
            ),
            cdf_normalization=False,
            num_parallel_samples=100,
        )
        print("Estimator created.")

        if not args.evaluate:
            # Train with memory management
            print("Starting training...")
            trained_net = estimator.train(
                training_data=train_data,
                validation_data=valid_data,
                num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
                prefetch_factor=2
            )
            print("Training finished.")
        else:
            # Evaluate the model
            predictor = estimator.create_predictor(
                transformation=estimator.create_transformation(),
                trained_network=estimator.create_training_network(estimator.trainer.device),
                device=estimator.trainer.device,
            )
            
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=valid_data,
                predictor=predictor,
                num_samples=100
            )
            
            forecasts = list(forecast_it)
            tss = list(ts_it)
            
            metrics, item_metrics = compute_validation_metrics(
                predictor=predictor,
                dataset=valid_data,
                window_length=CONTEXT_LENGTH + PREDICTION_LENGTH,
                prediction_length=PREDICTION_LENGTH,
                num_samples=100,
            )
            
            print("Evaluation metrics:", metrics)

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("WARNING: out of memory error occurred. Trying to recover...")
            torch.cuda.empty_cache()
            gc.collect()
            # Could implement retry logic here
            raise e
        else:
            raise e
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {str(e)}")
        raise e
    finally:
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints.")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Checkpoint to load.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--training_num_batches_per_epoch",
        type=int,
        default=512,
        help="Number of batches per epoch for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for training.",
    )
    parser.add_argument(
        "--clip_gradient",
        type=float,
        default=10.0,
        help="Gradient clipping value for training.",
    )
    parser.add_argument(
        "--early_stopping_epochs",
        type=int,
        default=10,
        help="Number of epochs to wait for improvement before early stopping.",
    )
    parser.add_argument(
        "--loss_normalization",
        type=str,
        default="batch",
        choices=["batch", "series", "none"],
        help="Normalization mode for the loss.",
    )

    # Encoders
    parser.add_argument(
        "--flow_series_embedding_dim",
        type=int,
        default=16,
        help="Dimension for the flow series embedding.",
    )
    parser.add_argument(
        "--copula_series_embedding_dim",
        type=int,
        default=16,
        help="Dimension for the copula series embedding.",
    )
    parser.add_argument(
        "--flow_input_encoder_layers",
        type=int,
        default=1,
        help="Number of layers for the flow input encoder.",
    )
    parser.add_argument(
        "--copula_input_encoder_layers",
        type=int,
        default=1,
        help="Number of layers for the copula input encoder.",
    )
    parser.add_argument(
        "--flow_encoder_num_layers",
        type=int,
        default=1,
        help="Number of layers for the flow encoder.",
    )
    parser.add_argument(
        "--flow_encoder_num_heads",
        type=int,
        default=1,
        help="Number of heads for the flow encoder.",
    )
    parser.add_argument(
        "--flow_encoder_dim",
        type=int,
        default=8,
        help="Dimension for the flow encoder.",
    )
    parser.add_argument(
        "--copula_encoder_num_layers",
        type=int,
        default=1,
        help="Number of layers for the copula encoder.",
    )
    parser.add_argument(
        "--copula_encoder_num_heads",
        type=int,
        default=1,
        help="Number of heads for the copula encoder.",
    )
    parser.add_argument(
        "--copula_encoder_dim",
        type=int,
        default=8,
        help="Dimension for the copula encoder.",
    )

    # Decoder
    parser.add_argument(
        "--decoder_num_heads",
        type=int,
        default=3,
        help="Number of heads for the decoder.",
    )
    parser.add_argument(
        "--decoder_num_layers",
        type=int,
        default=1,
        help="Number of layers for the decoder.",
    )
    parser.add_argument(
        "--decoder_dim",
        type=int,
        default=4,
        help="Dimension for the decoder.",
    )
    parser.add_argument(
        "--decoder_mlp_layers",
        type=int,
        default=2,
        help="Number of layers for the decoder MLP.",
    )
    parser.add_argument(
        "--decoder_mlp_dim",
        type=int,
        default=24,
        help="Dimension for the decoder MLP.",
    )
    parser.add_argument(
        "--decoder_resolution",
        type=int,
        default=20,
        help="Resolution for the decoder.",
    )
    parser.add_argument(
        "--decoder_attention_mlp_class",
        type=str,
        default="gluon",
        choices=["gluon", "simple"],
        help="Class for the decoder attention MLP.",
    )
    parser.add_argument(
        "--decoder_act",
        type=str,
        default="relu",
        choices=["relu", "elu", "softplus", "tanh", "sigmoid", "leakyrelu"],
        help="Activation function for the decoder.",
    )

    # DSF
    parser.add_argument(
        "--dsf_num_layers",
        type=int,
        default=2,
        help="Number of layers for the DSF.",
    )
    parser.add_argument(
        "--dsf_dim",
        type=int,
        default=8,
        help="Dimension for the DSF.",
    )
    parser.add_argument(
        "--dsf_mlp_layers",
        type=int,
        default=2,
        help="Number of layers for the DSF MLP.",
    )
    parser.add_argument(
        "--dsf_mlp_dim",
        type=int,
        default=24,
        help="Dimension for the DSF MLP.",
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

    # CPU
    parser.add_argument(
        "--use_cpu", action="store_true", default=False, help="When enabled, CPU is used instead of GPU"
    )

    # Flag for evaluation (either NLL or sampling and metrics)
    # A checkpoint must be provided for evaluation
    # Note evaluation is only supported after training the model in both phases.
    parser.add_argument("--evaluate", action="store_true", help="Evaluate for NLL and metrics.")

    # Add seed argument
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA version:", torch.version.cuda)

    main(args=args)
