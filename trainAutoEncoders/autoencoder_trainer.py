#!/usr/bin/env python

import os
import argparse
import gc
import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import devices, Config, SeedManager, DataLoaderFactory
from configs.helper import validate_autoencoder
from configs.plotsconfig import PlotConfig
from trainAutoEncoders.trainepoch import train_autos

# Models
from models.HufuNet import HufuNet
from models.attackerae import AttackerAE


class AutoencoderTrainer:
    """
    A trainer class for either HufuNet or AttackerAE autoencoders, depending on
    config.architecture_name.
    """
    def __init__(self, config: Config):
        self.config = config

        # Check the user's chosen model_name
        if config.architecture_name == "HufuNet":
            # Benign autoencoder
            SeedManager.set_benign_seed(config.benign_seed)
            self.autoencoder = HufuNet().to(devices)

            # Ensure directory for benign checkpoints
            os.makedirs(os.path.dirname(config.benign_decoder_path), exist_ok=True)
            self.encoder_save_path = config.benign_encoder_path
            self.decoder_save_path = config.benign_decoder_path

            # We'll read dataset name from config.hufunet_dataset
            self.auto_dataset_name = config.hufunet_dataset
            identifier = "HufuNet"

        elif config.architecture_name == "AttackerAE":
            # Attacker AE
            SeedManager.set_attacker_seed(config.attacker_seed)
            self.autoencoder = AttackerAE().to(devices)

            # Ensure directory for attacker checkpoints
            os.makedirs(os.path.dirname(config.attacker_decoder_path), exist_ok=True)
            self.encoder_save_path = config.attack_encoder_path
            self.decoder_save_path = config.attacker_decoder_path

            # We'll read dataset name from config.attacker_ae_dataset
            self.auto_dataset_name = config.attacker_ae_dataset
            identifier = "AttackerAE"

        else:
            raise ValueError(
                f"AutoencoderTrainer only supports 'HufuNet' or 'AttackerAE'. "
                f"You provided architecture_name='{config.architecture_name}'."
            )

        # Load the autoencoder data from config
        self.train_loader, self.test_loader = DataLoaderFactory.get_autoencoder_datasets(identifier, config)
        self.total_rounds = config.total_rounds

        # Print info about autoencoder training
        auto_batch_size = self.train_loader.batch_size
        print(f"Training Autoencoder: {config.architecture_name} "
              f"using dataset: {self.auto_dataset_name} "
              f"with batch size: {auto_batch_size}")

        # Print total trainable parameters (encoder + decoder)
        total_params = sum(p.numel() for p in self.autoencoder.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_params}")

        # Checkpoint/logging directories
        self.checkpoint_dir = os.path.dirname(self.encoder_save_path)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Log file path
        self.log_file = os.path.join(self.checkpoint_dir, f"train_log_{config.architecture_name}_ae.txt")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # log header
        header_line = f"{'Epoch':<10}{'Train Loss':<15}{'Validation Loss':<15}\n"
        with open(self.log_file, "w") as f:
            f.write(header_line)

        # Loss, optimizer, scheduler
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses   = []

        # Different LR for attacker's vs. benign(HufuNet autoencoder)
        if config.architecture_name == "AttackerAE":
            self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=config.auto_lr)
        else:
            self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=config.lr)


    def train(self):
        """
        Main training loop for the autoencoder. Uses the shared train_autos function
        and logs both train & validation losses.
        """
        for epoch in range(self.total_rounds):
            train_loss_epoch = train_autos(
                self.autoencoder,
                self.train_loader,
                self.optimizer,
                self.criterion,
                devices,
                epoch,
                self.total_rounds
            )
            self.train_losses.append(train_loss_epoch)

            # Validation step
            val_loss = validate_autoencoder(self.autoencoder, self.test_loader)
            self.val_losses.append(val_loss)

            # Always print to console
            print(f"Epoch [{epoch+1}/{self.total_rounds}] - "
                  f"Training Loss: {train_loss_epoch:.4f} - "
                  f"Validation Loss: {val_loss:.4f}")

            # log to file first or every 10th epoch
            if epoch == 0 or ((epoch + 1) % 10 == 0):
                line = f"{epoch+1:<10}{train_loss_epoch:<15.4f}{val_loss:<15.4f}\n"
                with open(self.log_file, "a") as f:
                    f.write(line)


        print("Training completed.")

        # Save the final encoder & decoder
        torch.save(self.autoencoder.encoder.state_dict(), self.encoder_save_path)
        torch.save(self.autoencoder.decoder.state_dict(), self.decoder_save_path)
        print("Encoder and Decoder saved at:")
        print("  ", self.encoder_save_path)
        print("  ", self.decoder_save_path)

        # Log final details
        model_name = self.config.architecture_name
        dataset_used = self.auto_dataset_name
        batch_size = getattr(self.train_loader, "batch_size", "Unknown")
        encoder_params = sum(p.numel() for p in self.autoencoder.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.autoencoder.decoder.parameters() if p.requires_grad)

        details = (
            "\n==== Final Model Details ====\n"
            f"Model Name: {model_name}\n"
            f"Dataset Used: {dataset_used}\n"
            f"Batch Size: {batch_size}\n"
            f"Encoder Trained Parameters: {encoder_params}\n"
            f"Decoder Trained Parameters: {decoder_params}\n"
        )
        print(details)
        with open(self.log_file, "a") as f:
            f.write(details)

        # Use the PlotConfig class for visualizations
        plot_dir = self.config.full_plot_dir
        PlotConfig.visualize_reconstructions(self.autoencoder, self.test_loader, devices, model_name, plot_dir)
        PlotConfig.plot_auto_losses(self.train_losses, self.val_losses, self.total_rounds, plot_dir, model_name)

        # Clean up
        del self.autoencoder, self.train_loader, self.test_loader, self.optimizer
        torch.cuda.empty_cache()
        gc.collect()
        print("Memory cleaned up.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an Autoencoder (HufuNet or AttackerAE).")
    parser.add_argument("--model_name", type=str, default="HufuNet",
                        help="Which model to train: 'HufuNet' or 'AttackerAE'.")
    parser.add_argument("--total_rounds", type=int, default=2,
                        help="Number of epochs (default=2).")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    cfg.architecture_name = args.model_name
    cfg.total_rounds = args.total_rounds

    trainer = AutoencoderTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
