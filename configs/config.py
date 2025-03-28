import os
import pickle
from dataclasses import dataclass,field
import torch
import numpy as np
from torch.utils.data import DataLoader

from models.cnn import CnnModel
from models.resnet import res_net18
from models.mlp import MLP
from models.mlp_riga import MLP_RIGA
from models.HufuNet import HufuNet
from models.attackerae import AttackerAE
from configs.dataconfig import Database

# set the device to GPU if available else CPU
devices: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Base directory for saving all results.
BASE_DIR = os.path.join("results", "trained_models", "Hufu")

@dataclass
class Config:
    """Configuration class for setting up model training parameters and directories."""
    benign_seed: int =2020  #42  # 2020  2050 # Seeds experiments for benign model training
    attacker_seed: int =64  # seed for attacker model training
    watermark_enabled: bool = True  # Flag Enable or disable watermarking
    architecture_name: str = "CNN" # Options: ["MLP", "MLP_RIGA", "CNN", "ResNet18"]
    total_rounds: int = 100  # Total rounds for training the main model
    attacker_round: int = 50 # Total rounds for training the attacked model
    embed_interval: int =5   # Interval for embedding the watermark
    model_dir: str = os.path.join(BASE_DIR, "Trained_Dummy_Model") # Directory for saving the pre-trained models(dummy model for watermarking)
    benign_encoder_path: str = os.path.join(BASE_DIR, "benign_checkpoints", "model_encoder_final.pth") # Path to save the HufuNet encoder model
    benign_decoder_path: str = os.path.join(BASE_DIR, "benign_checkpoints", "model_decoder_final.pth") # Path to save the HufuNet decoder model
    lr: float = 5e-3 # Learning rate for the main model and HufuNet autoencoder
    momentum: float = 0.9 # Momentum for the main model dummy model
    weight_decay: float = 5e-4 # Weight decay for the main model
    betas = (0.9, 0.99) # Betas for Adam optimizer
    eta_min: float = 1e-10 # Minimum learning rate for the main model

    auto_lr = 3e-4#  Attacker Autoencoder learning rate


    attacker_decoder_path: str = os.path.join(BASE_DIR, "attacker_checkpoints", "model_decoder_final.pth") # Path to save the AttackerAE decoder model
    attack_encoder_path: str = os.path.join(BASE_DIR, "attacker_checkpoints", "model_encoder_final.pth") # Path to save the AttackerAE encoder model
    attacker_lr: float = 5e-2 # divide lr /(1/10) # Learning rate for the attacked model
    attacker_weight_decay: float = 5e-4 # Weight decay for the attacked model
    fine_tuning_epochs: int = 50 # Fine-tuning epochs for the attacker

    hufunet_batch_size: int = 500 # Batch size for HufuNet autoencoder

    attacker_ae_batch_size: int = 64 # Batch size for AttackerAE autoencoder
    main_model_batch_size: int = 64 # Batch size for the main model
    attacker_main_model_batch_size: int = 64

    hufunet_dataset: str = "mnist" # Dataset for HufuNet autoencoder
    attacker_ae_dataset: str = "mnist" # Dataset for AttackerAE autoencoder
    # Map each architecture to a dataset for main model
    architecture_dataset_map: dict = field(default_factory=lambda: {
        "MLP": "mnist",
        "MLP_RIGA": "mnist",
        "CNN": "cifar10",
        "ResNet18": "cifar10"
    })
    # Directory for saving the results
    plot_dir: str = os.path.join("results", "plots")
    autoencoder_plot_dir: str = os.path.join("results", "autoencoder_plots")

    @property
    def result_dir(self) -> str:
        """Returns the directory for saving results based on watermarking status and architecture."""
        base_result_dir = os.path.join(BASE_DIR, "TrainedModel")
        watermark_subdir = "Watermarked" if self.watermark_enabled else "NonWatermarked"
        return os.path.join(base_result_dir, watermark_subdir, self.architecture_name)

    @property
    def log_file_path(self) -> str:
        """Returns the path to the log file for the current architecture."""
        return os.path.join(self.result_dir, f"{self.architecture_name}_log.txt")

    @property
    def full_plot_dir(self) -> str:
        """ Returns the directory for saving autoencoder plots and , creates the directory if it does not exist."""
        plot_subdir = os.path.join(self.autoencoder_plot_dir, self.architecture_name)
        os.makedirs(plot_subdir, exist_ok=True)
        return plot_subdir



class SeedManager:
    """Manages setting seeds for reproducibility."""
    @staticmethod
    def set_benign_seed(seed: int) -> None:
        """Sets the seed for benign model training."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def set_attacker_seed(seed: int) -> None:
        """Sets the seed for attacker model training."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class DataLoaderFactory:
    """
    Factory class for creating data loaders for different models and datasets.
    """
    @staticmethod
    def get_model_datasets(identifier: str, config: Config, mode: str = "benign"):
        """

        Returns train_loader and test_loader for the main model.

        Args:
            identifier (str): Model architecture identifier.
            config (Config): Configuration object.
            mode (str): Mode of operation, either 'benign' or 'attacker'.

        Returns:
            tuple: train_loader, test_loader
        """
        # Get dataset from config
        if identifier not in config.architecture_dataset_map:
            raise ValueError(f"Unknown architecture '{identifier}' not in config.architecture_dataset_map.")

        database = config.architecture_dataset_map[identifier]

        # Get batch size from config
        if mode == "benign":
            batch_size = config.main_model_batch_size
        elif mode == "attacker":
            batch_size = config.attacker_main_model_batch_size
        else:
            batch_size = 64  # fallback if needed

        # Load the dataset
        train_set, test_set = Database.get_datasets(database)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    @staticmethod
    def get_autoencoder_datasets(identifier: str, config: Config):
        """
        Returns train_loader and test_loader for autoencoders.

        Args:
            identifier (str): Autoencoder model identifier.
            config (Config): Configuration object.

        Returns:
            tuple: train_loader, test_loader
        """
        if identifier == "HufuNet":
            database   = config.hufunet_dataset
            batch_size = config.hufunet_batch_size
        elif identifier == "AttackerAE":
            database   = config.attacker_ae_dataset
            batch_size = config.attacker_ae_batch_size
        else:
            raise ValueError(f"Unknown identifier for autoencoder models: {identifier}")

        train_set, test_set = Database.get_datasets(database)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)
        return train_loader, test_loader


class ModelFactory:
    """ Factory class for creating models instances on the architecture name."""
    @staticmethod
    def get_model(architecture_name: str):
        """
        Returns an instance of the model based on the architecture name.

        Args:
            architecture_name (str): Name of the model architecture.

        Returns:
            torch.nn.Module: Model instance.

        """
        if architecture_name == "CNN":
            return CnnModel()
        elif architecture_name == "MLP":
            return MLP()
        elif architecture_name == "MLP_RIGA":
            return MLP_RIGA()
        elif architecture_name == "ResNet18":
            return res_net18()
        elif architecture_name == "HufuNet":
            return HufuNet()
        elif architecture_name == "AttackerAE":
            return AttackerAE()
        else:
            raise ValueError(f"Unknown architecture: {architecture_name}")

class CheckpointLoader:
    """Utility class for loading model checkpoints."""
    @staticmethod
    def load_checkpoint(file_path):
        """
        Loads a checkpoint from the given file path.

        Args:
            file_path (str): Path to the checkpoint file.

        Returns:
            dict: Loaded checkpoint.
        """
        checkpoint = torch.load(file_path, map_location=devices, pickle_module=pickle)
        for key in ("net", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
        return checkpoint

    @staticmethod
    def load_prepared_model_weights(model_dir, architecture_name):
        """
        Loads pre-trained model weights from the specified directory.

        Args:
            model_dir (str): Directory containing the model weights.
            architecture_name (str): Name of the model architecture.

        Returns:
            dict: Loaded model weights.
        """
        expected_filename = f"{architecture_name}_dummy.pth"
        path = os.path.join(model_dir, expected_filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected pre-trained model file '{expected_filename}' not found in '{model_dir}'."
            )
        return CheckpointLoader.load_checkpoint(path)


class DirectoryManager:
    """Utility class for managing directories."""
    @staticmethod
    def save_attacker_dir(architecture_name: str) -> str:
        """
                Returns the directory path for saving attacker results.

                Args:
                    architecture_name (str): Name of the model architecture.

                Returns:
                    str: Directory path for saving attacker results.
                """
        return os.path.join(BASE_DIR, "TrainedModel", "attacker_results", architecture_name)

    @staticmethod
    def main_model_dir(architecture_name: str) -> str:
        """
                Returns the directory path for saving main model results.

                Args:
                    architecture_name (str): Name of the model architecture.

                Returns:
                    str: Directory path for saving main model results.
                """
        return os.path.join(BASE_DIR, "TrainedModel", architecture_name)

    @staticmethod
    def save_autoencoder_dir(architecture_name: str) -> str:
        """
                Returns the directory path for saving autoencoder results.

                Args:
                    architecture_name (str): Name of the autoencoder architecture.

                Returns:
                    str: Directory path for saving autoencoder results.
                """
        return os.path.join(BASE_DIR, "Autoencoder", architecture_name)
