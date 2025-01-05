import pandas as pd
import numpy as np
import torch
from gluonts.dataset.common import DataEntry, MetaData, Dataset, ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.transform import AddObservedValuesIndicator
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
DEFAULT_BATCH_SIZE = 32  # Reduced from 256
DEFAULT_ACCUMULATION_STEPS = 8  # To simulate larger batch size
MAX_SAMPLES_IN_MEMORY = 1000  # Limit number of samples loaded at once

class ParquetDataset(Dataset):
    def __init__(self, parquet_file: str, freq: str, context_length: int, prediction_length: int, max_samples=MAX_SAMPLES_IN_MEMORY):
        # Read the parquet file
        self.data = pd.read_parquet(parquet_file)
        
        # Ensure the index is datetime
        if not isinstance(self.data.index, pd.DatetimeIndex):
            if 'time' in self.data.columns:
                self.data.set_index('time', inplace=True)
            self.data.index = pd.to_datetime(self.data.index)
        
        self.freq = freq
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.max_samples = max_samples
        self.metadata = MetaData(
            freq=self.freq,
            target=None,
            feat_static_cat=[],
            feat_static_real=[],
            feat_dynamic_real=[],
            feat_dynamic_cat=[],
            prediction_length=self.prediction_length,
        )

    def __iter__(self) -> Iterator[DataEntry]:
        # Process data in chunks to save memory
        chunk_size = self.max_samples
        total_samples = len(self.data) - self.context_length - self.prediction_length + 1

        for chunk_start in tqdm(range(0, total_samples, chunk_size), desc="Loading data chunks"):
            start_time = time.time()
            chunk_end = min(chunk_start + chunk_size, total_samples)

            for i in range(chunk_start, chunk_end):
                mask = create_mask(
                    self.data,
                    ALL_STATIC_FEATURES,
                    ALL_DYNAMIC_FEATURES,
                    self.context_length,
                    self.prediction_length,
                )

                # Select only the necessary rows for this data entry
                targets = self.data[TARGET_FEATURES].iloc[i : i + self.context_length + self.prediction_length].values.T
                start_time_data = self.data.index[i]

                data_entry = {
                    FieldName.START: pd.Timestamp(start_time_data),
                    FieldName.TARGET: targets,
                    FieldName.FEAT_STATIC_REAL: self.data[ALL_STATIC_FEATURES]
                    .iloc[i : i + self.context_length + self.prediction_length]
                    .values.T,
                    FieldName.FEAT_DYNAMIC_REAL: self.data[ALL_DYNAMIC_FEATURES]
                    .iloc[i : i + self.context_length + self.prediction_length]
                    .values.T,
                    FieldName.OBSERVED_VALUES: mask[
                        i : i + self.context_length + self.prediction_length
                    ].T,
                }
                yield data_entry

            # Clear memory after processing each chunk
            torch.cuda.empty_cache()
            gc.collect()
            end_time = time.time()
            print(f"Chunk loading time: {end_time - start_time:.4f} seconds")

    def __len__(self):
        return len(self.data) - self.context_length - self.prediction_length + 1

def create_data_transformation(
    dataset: Dataset,
    context_length: int,
    prediction_length: int,
    mode: str,
    cdf_normalization: bool = False,
) -> Transformation:
    """
    Create the data transformation for the TACTiS model.

    Args:
        dataset: The dataset to transform.
        context_length: The length of the context.
        prediction_length: The length of the prediction horizon.
        mode: The mode of the transformation ("train", "validation", or "test").
        cdf_normalization: Whether to apply CDF normalization to the target features.

    Returns:
        The data transformation.
    """
    if mode == "train":
        instance_sampler = ValidationSplitSampler(
            min_future=prediction_length,
            past_length=context_length,
            min_past=context_length,
        )
    elif mode == "validation":
        instance_sampler = ValidationSplitSampler(
            min_future=prediction_length,
            past_length=context_length,
            min_past=context_length,
        )
    elif mode == "test":
        instance_sampler = TestSplitSampler()
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Create the transformation
    transformation = Chain(
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
                instance_sampler=instance_sampler,
                past_length=context_length,
                future_length=prediction_length,
                time_series_fields=[FieldName.FEAT_DYNAMIC_REAL, FieldName.OBSERVED_VALUES],
            ),
        ]
    )

    return transformation

def create_datasets(
    parquet_file: str,
    freq: str,
    context_length: int,
    prediction_length: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[MetaData, Dataset, Dataset, Dataset]:
    """
    Create the training, validation, and test datasets.

    Args:
        parquet_file: The path to the Parquet file.
        freq: The frequency of the data.
        context_length: The length of the context.
        prediction_length: The length of the prediction horizon.
        train_frac: The fraction of the data to use for training.
        val_frac: The fraction of the data to use for validation.

    Returns:
        A tuple containing the metadata, training dataset, validation dataset, and test dataset.
    """
    dataset = ParquetDataset(parquet_file, freq, context_length, prediction_length)
    num_data = len(dataset)
    num_train = int(num_data * train_frac)
    num_val = int(num_data * val_frac)

    train_data = ListDataset(
        list(dataset)[:num_train],
        freq=dataset.freq,
    )
    val_data = ListDataset(
        list(dataset)[num_train : num_train + num_val],
        freq=dataset.freq,
    )
    test_data = ListDataset(
        list(dataset)[num_train + num_val :],
        freq=dataset.freq,
    )

    return dataset.metadata, train_data, val_data, test_data

def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    try:
        # Print initial GPU memory usage
        if torch.cuda.is_available() and not args.use_cpu:
            print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # Create datasets with memory management
        print("Creating dataset...")
        dataset = ParquetDataset(
            parquet_file=DATA_FILE,
            freq=FREQ,
            context_length=CONTEXT_LENGTH,
            prediction_length=PREDICTION_LENGTH,
            max_samples=MAX_SAMPLES_IN_MEMORY
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

        # Split into train and validation sets (80-20 split)
        total_length = len(dataset)
        train_length = int(0.8 * total_length)
        print("Splitting dataset into train and validation sets...")
        print(f"Total length: {total_length}")
        print(f"Train length: {train_length}")
        
        # Load data in chunks
        train_data = []
        valid_data = []
        
        for i, data_entry in enumerate(dataset):
            print(f"Processing data entry {i} of {total_length}")
            if i < train_length:
                train_data.append(data_entry)
            else:
                valid_data.append(data_entry)
                
            # Clear memory periodically
            if i % 1000 == 0:
                print(f"Clearing memory after {i} data entries...")
                torch.cuda.empty_cache()
                gc.collect()

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
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of multiprocessing workers.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size.")
    parser.add_argument("--epochs", type=int, default=1000, help="Epochs.")

    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["rmsprop", "adam"], help="Optimizer to be used."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Folder to store all checkpoints in. This folder will be created automatically if it does not exist.",
    )
    parser.add_argument(
        "--load_checkpoint", type=str, help="Checkpoint to start training from or a checkpoint to evaluate."
    )
    parser.add_argument(
        "--training_num_batches_per_epoch",
        type=int,
        default=512,
        help="Number of batches in a single epoch of training.",
    )
    parser.add_argument(
        "--log_subparams_every",
        type=int,
        default=10000,
        help="Frequency of logging the epoch number and iteration number during training.",
    )
    parser.add_argument("--bagging_size", type=int, default=20, help="Bagging Size")

    # Early stopping epochs based on total validation loss. -1 indicates no early stopping.
    parser.add_argument("--early_stopping_epochs", type=int, default=50, help="Early stopping patience")

    # HPARAMS
    # General ones
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight Decay")
    parser.add_argument("--clip_gradient", type=float, default=1e3, help="Clip Gradient")
    parser.add_argument(
        "--flow_series_embedding_dim",
        type=int,
        default=5,
        help="Embedding dimension for the series for the flow.",
    )
    parser.add_argument(
        "--copula_series_embedding_dim",
        type=int,
        default=5,
        help="Embedding dimension for the series for the copula.",
    )
    parser.add_argument(
        "--flow_input_encoder_layers",
        type=int,
        default=2,
        help="Number of layers for the input encoder for the flow.",
    )
    parser.add_argument(
        "--copula_input_encoder_layers",
        type=int,
        default=2,
        help="Number of layers for the input encoder for the copula.",
    )
    parser.add_argument(
        "--loss_normalization",
        type=str,
        default="series",
        choices=["none", "series", "batch"],
        help="Normalization for the loss.",
    )
    parser.add_argument(
        "--history_factor",
        type=float,
        default=2,
        help="Factor to determine the history length w.r.t. prediction length.",
    )

    # Encoder
    parser.add_argument(
        "--flow_encoder_num_layers",
        type=int,
        default=2,
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
        default=16,
        help="Dimension for the flow encoder.",
    )
    parser.add_argument(
        "--copula_encoder_num_layers",
        type=int,
        default=2,
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
        default=16,
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
        default=8,
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
        default=48,
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
        default=48,
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

    args = parser.parse_args()

    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA version:", torch.version.cuda)

    main(args=args)
