import os
import gc
import argparse
import torch
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler

from configs.config import devices, Config, SeedManager, DataLoaderFactory, ModelFactory, CheckpointLoader
from train_epoch import train_model, validate_model
from models.HufuNet import HufuNet
from configs.helper import (
    SingleLoss,
    extract_dummy_encoder,
    evaluate_autoencoder_loss,
    combine_encoder_decoder
)
from configs.plotsconfig import PlotConfig


class NonWatermarkTrainer:
    """
    Trainer for models trained without watermark embedding.

    This class handles training, evaluation, logging, and saving of a non-watermarked model,
    along with evaluation using the HufuNet autoencoder to monitor reconstruction consistency.
    """
    def __init__(self, config: Config):
        """
        Initialize the trainer, set up directories, models, dataloaders, and optimizers.

        Args:
            config (Config): Experiment configuration object.
        """
        self.config = config
        SeedManager.set_benign_seed(config.benign_seed)

        # Directories and logging
        self.result_dir = config.result_dir
        self.log_file_path = config.log_file_path
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        # create a header line for the log file
        header_line = f"{'Epoch':<10}{'TrainAcc':<15}{'TrainLoss':<15}{'ValAcc':<15}{'ValLoss':<15}\n"
        with open(self.log_file_path, "w") as lf:
            lf.write(header_line)

        # Data loaders (Main Model)
        self.main_train_loader, self.main_val_loader = DataLoaderFactory.get_model_datasets(
            config.architecture_name,
            config=config,
            mode="benign"
        )

        # Retrieve the dataset name and batch size they are using
        main_dataset_name = config.architecture_dataset_map[config.architecture_name]
        main_batch_size   = self.main_train_loader.batch_size

        # Print the main model info
        print(f"Training MainModel: {config.architecture_name} "
              f"using dataset: {main_dataset_name} "
              f"with batch size: {main_batch_size}")

        # For HufuNet autoencoder
        auto_train_loader, self.auto_val_loader = DataLoaderFactory.get_autoencoder_datasets("HufuNet", config=config)
        auto_dataset_name = config.hufunet_dataset
        auto_batch_size   = auto_train_loader.batch_size

        print(f"Training Autoencoder: HufuNet using dataset: {auto_dataset_name} "
              f"with batch size: {auto_batch_size}")

        #  Print trainable parameter for HufuNet
        temp_auto = HufuNet().to(devices)
        encoder_total_params = sum(p.numel() for p in temp_auto.encoder.parameters() if p.requires_grad)
        print(f"Trainable parameters for HufuNet encoder: {encoder_total_params}")

        # Load Pretrained dummy Weights
        prepared_weights = CheckpointLoader.load_prepared_model_weights(config.model_dir, config.architecture_name)

        # load an autoencoder template to get encoder_weights
        self.encoder_weights = temp_auto.encoder.state_dict()
        self.decoder_weights = CheckpointLoader.load_checkpoint(config.benign_decoder_path)

        # Initialize Main Model
        self.main_model = self.load_main_model(prepared_weights, config.architecture_name)
        self.main_model.to(devices)

        # Print trainable parameter count for the main model
        total_params = sum(p.numel() for p in self.main_model.parameters() if p.requires_grad)
        print(f"Trainable parameters for MainModel: {total_params}")

        # Optimizer and Scheduler
        self.optimizer = optim.Adam(
            self.main_model.parameters(),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay
        )
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.total_rounds,
            eta_min=config.eta_min
        )

        # Initialize the autoencoder for evaluating the reconstruction loss in the last round of training
        self.eval_autoencoder = HufuNet().to(devices)
        self.eval_autoencoder.decoder.load_state_dict(self.decoder_weights, strict=True)

        # Metrics storage
        self.train_losses = []
        self.train_accs   = []
        self.val_losses   = []
        self.val_accs     = []

    def load_main_model(self, weights_dict, arch_name):
        """

        Load a main model and initialize with pretrained weights.

        Args:
            weights_dict (dict): Dictionary of pretrained weights.
            arch_name (str): Model architecture name.

        Returns:
            nn.Module: Initialized main model.
        """
        model = ModelFactory.get_model(arch_name)
        base_model_sd = model.state_dict()
        for key in base_model_sd:
            if key in weights_dict and base_model_sd[key].shape == weights_dict[key].shape:
                base_model_sd[key] = weights_dict[key].clone()
        model.load_state_dict(base_model_sd, strict=True)
        return model

    def run(self):
        for epoch in range(self.config.total_rounds):
            print(f"Epoch {epoch+1}/{self.config.total_rounds}")

            train_loss, train_acc = train_model(
                self.main_model,
                self.main_train_loader,
                SingleLoss(),
                self.optimizer,
                devices,
                desc="Training"
            )
            val_loss, val_acc = validate_model(
                self.main_model,
                self.main_val_loader,
                SingleLoss(),
                devices,
                desc=f"Validation Epoch {epoch+1}"
            )

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # print Acc and Loss for each epoch
            print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Train Loss {train_loss:.4f}, "
                  f"Val Acc {val_acc:.2f}%, Val Loss {val_loss:.4f}")

            # Log line
            line = f"{epoch+1:<10}{train_acc:<15.2f}{train_loss:<15.4f}{val_acc:<15.2f}{val_loss:<15.4f}\n"
            # Log only to txt file for first epoch or every 10th epoch
            if epoch == 0 or ((epoch + 1) % 10 == 0):
                with open(self.log_file_path, "a") as lf:
                    lf.write(line)

            self.scheduler.step()

        self.save_final_results()
        self.plot_main_model_metrics()

        print("\nAll training steps completed.")
        torch.cuda.empty_cache()
        gc.collect()
        print("Memory cleaned.")

        return self.main_model.state_dict()

    def save_final_results(self):
        """
        Save the final main model weights and compute AE reconstruction loss.
        Logs final stats to file.
        """
        arch = self.config.architecture_name
        final_main_state = self.main_model.state_dict()
        final_main_path  = os.path.join(self.result_dir, f"{arch}_final_main_model.pth")
        os.makedirs(os.path.dirname(final_main_path), exist_ok=True)
        torch.save(final_main_state, final_main_path)
        print(f"\n==> Final main model saved at {final_main_path}")

        # Evaluate autoencoder reconstruction
        temp_auto = HufuNet().to(devices)
        dummy_encoder_template = temp_auto.encoder.state_dict()
        extracted_encoder_state = extract_dummy_encoder(
            final_main_state, dummy_encoder_template, str(self.config.benign_seed)
        )
        combine_encoder_decoder(extracted_encoder_state, self.decoder_weights)
        final_ae_loss = evaluate_autoencoder_loss(extracted_encoder_state, self.decoder_weights, self.auto_val_loader)

        num_params = sum(p.numel() for p in self.main_model.parameters() if p.requires_grad)


        # append final details to the log file
        details = (
            "\n==== Final Model Details ====\n"
            f"Trainable parameters for MainModel: {num_params}\n"
            f"Final Reformulated AE Reconstruction Loss = {final_ae_loss:.4f}\n"
        )
        print(details)
        with open(self.log_file_path, "a") as lf:
            lf.write(details)

    def plot_main_model_metrics(self):
        """Plot training and validation losses for the main model."""
        PlotConfig.plot_non_watermark(
            self.train_losses,
            self.val_losses,
            self.config.total_rounds,
            self.result_dir,
            self.config.architecture_name
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train Non-Watermark Model with minimal arguments.")
    parser.add_argument("--model_name", type=str, default="CNN",
                        help="Architecture name (e.g. CNN, MLP, ResNet18).")
    parser.add_argument("--total_rounds", type=int, default=1,
                        help="Number of training epochs.")
    return parser.parse_args()

def main():
    args = parse_args()

    cfg = Config()
    # by default, watermark is enabled. We need to disable it for this task.
    cfg.watermark_enabled = False

    # Override the following two fields
    cfg.architecture_name = args.model_name
    cfg.total_rounds      = args.total_rounds

    trainer = NonWatermarkTrainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()
