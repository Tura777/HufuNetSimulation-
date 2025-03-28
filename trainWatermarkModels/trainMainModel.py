import os
import gc
import argparse
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from tqdm import tqdm

from models.HufuNet import HufuNet
from configs.config import Config, SeedManager, DataLoaderFactory, ModelFactory, CheckpointLoader, devices
from configs.helper import (
    MultiLoss,
    SingleLoss,
    validate_main_model,
    AverageMeter,
    embed_encoder_to_model,
    extract_encoder_from_model,
    evaluate_autoencoder_loss,
    combine_encoder_decoder,
    report_embedding_extraction_misplacements,
    verify_extraction_reinsertion
)
from attackers.attacker import AttackSimulator
from configs.plotsconfig import PlotConfig

torch.backends.cudnn.enabled = False


class WatermarkTrainer:
    def __init__(self, config: Config):
        """
        Trainer for models with embedded watermarking using HufuNet-style autoencoders.

            Responsibilities:
            - Load pretrained weights and benign watermark templates.
            - Train a model while iteratively embedding encoder weights.
            - Maintain gradient-based regulation via MultiLoss.
            - Periodically re-embed updated encoder.
            - Log performance and reconstruction loss.
            - Perform extraction verification and report mismatches.
        """
        self.config = config
        SeedManager.set_benign_seed(config.benign_seed)

        # Directories & Logging
        self.result_dir = config.result_dir
        self.log_file_path = config.log_file_path
        os.makedirs(self.result_dir, exist_ok=True)

        # Create main log file with header that includes watermarking info
        with open(self.log_file_path, "w") as lf:
            lf.write(f"=== Watermarking {self.config.architecture_name} ===\n")
            lf.write(f"{'Epoch':<6}{'MainTrainAcc':>14}{'MainTrainLoss':>14}"
                     f"{'MainValAcc':>12}{'MainValLoss':>12}{'ExtractedAE':>12}\n")

        # AE training loss log
        self.ae_loss_file_path = os.path.join(self.result_dir, "AElossWM.txt")
        with open(self.ae_loss_file_path, "w") as lf:
            lf.write("Epoch   AE_Train_Loss\n")

        # DataLoaders
        self.main_train_loader = None
        self.main_val_loader   = None
        self.auto_train_loader = None
        self.auto_val_loader   = None

        # Models
        self.main_model       = None
        self.eval_autoencoder = None
        self.combined_autoencoder = None  # for iterative training

        # Watermark states
        self.encoder_weights  = None
        self.decoder_weights  = None
        self.current_encoder  = None

        # Optimizer and scheduler
        self.optimizer = None
        self.scheduler = None

        # Grad tracking (for MultiLoss)
        self.prev_encoder_grad = torch.tensor(0.0, device=devices)
        self.prev_main_grad    = torch.tensor(0.0, device=devices)
        self.prev_ratio        = torch.tensor(1.0, device=devices)

        # Metrics
        self.main_train_losses = []
        self.main_val_losses   = []
        self.ae_train_losses   = []
        self.ae_val_losses     = []
        self.loss_correlation_val   = []
        self.loss_correlation_train = []
        self.last_ae_train_loss     = 0.0

        # Load data
        self.load_data()

        #  For Initializing models
        self.initialize_models()

    def load_data(self):
        """
        Load main model and their dataset (train/val).
        If watermark is enabled, also load autoencoder data (train/val).
        """
        # Main model data
        self.main_train_loader, self.main_val_loader = DataLoaderFactory.get_model_datasets(
            self.config.architecture_name,
            config=self.config,
            mode="benign"
        )
        main_dataset = self.config.architecture_dataset_map[self.config.architecture_name]
        main_bs      = self.main_train_loader.batch_size
        print(f"Training MainModel: {self.config.architecture_name} using dataset: {main_dataset}, batch size: {main_bs}")

        # If watermark is enabled, load autoencoder data
        if self.config.watermark_enabled:
            self.auto_train_loader, self.auto_val_loader = DataLoaderFactory.get_autoencoder_datasets(
                "HufuNet", config=self.config
            )
            print(f"Training Autoencoder (HufuNet) using dataset: {self.config.hufunet_dataset}, batch size: {self.auto_train_loader.batch_size}")
        else:
            self.auto_train_loader = None
            self.auto_val_loader   = None

    def initialize_models(self):
        """
        Load pre-trained dummy weights, load benign watermark,
        create main model + optimizer/scheduler,
        and  an eval autoencoder for reference.
        """
        #  Dummy weights
        prepared_weights = CheckpointLoader.load_prepared_model_weights(
            self.config.model_dir,
            self.config.architecture_name
        )
        # Benign watermark
        self.encoder_weights = CheckpointLoader.load_checkpoint(self.config.benign_encoder_path)
        self.decoder_weights = CheckpointLoader.load_checkpoint(self.config.benign_decoder_path)
        self.current_encoder = deepcopy(self.encoder_weights)

        # Create main model and load prepared weights
        self.main_model = self.load_main_model_weights(prepared_weights)
        self.main_model.to(devices)

        # Count trainable parameters for the main model
        total_params = sum(p.numel() for p in self.main_model.parameters() if p.requires_grad)
        print(f"Trainable Main model parameters: {total_params}")

        #  set  Optimizer + scheduler
        self.optimizer = optim.Adam(
            self.main_model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.total_rounds,
            eta_min=self.config.eta_min
        )

        #  Autoencoder for evaluation
        self.eval_autoencoder = HufuNet().to(devices)
        self.eval_autoencoder.decoder.load_state_dict(self.decoder_weights, strict=True)
        # Print trainable parameter counts for the autoencoder and its encoder
        auto_total_params = sum(p.numel() for p in self.eval_autoencoder.parameters() if p.requires_grad)
        encoder_total_params = sum(p.numel() for p in self.eval_autoencoder.encoder.parameters() if p.requires_grad)
        print(f"Trainable Autoencoder parameters: {auto_total_params}")
        print(f"Trainable parameters for Encoder: {encoder_total_params}")

        # Combined autoencoder initially set to None
        self.combined_autoencoder = None

    def load_main_model_weights(self, prepared_weights):
        """
        Create main model, load 'prepared_weights' into it.
        """
        model = ModelFactory.get_model(self.config.architecture_name)
        base_model_sd = model.state_dict()
        for key in base_model_sd:
            if key in prepared_weights and base_model_sd[key].shape == prepared_weights[key].shape:
                base_model_sd[key] = prepared_weights[key].clone()
        model.load_state_dict(base_model_sd, strict=True)
        return model


    # TRAINING METHODS
    def train_main_model(self, epoch):
        """
        Train main model. Return (main_train_acc, main_train_loss, alpha_val, criterion).
        """
        criterion = SingleLoss() if epoch == 0 else MultiLoss()
        alpha_val = 0.00001 if (epoch > self.config.total_rounds * 0.3) else 0.000005  # as per original code by HufuNet

        threshold = 1e-5  # this also constant set by original code in HufuNet
        wm_grad_sum   = 0.0
        wm_count      = 0.0
        main_grad_sum = 0.0
        main_count    = 0.0

        losses = AverageMeter()
        correct = 0
        total   = 0

        self.main_model.train()
        for inputs, targets in tqdm(self.main_train_loader, desc=f"Main Epoch {epoch + 1}", leave=True):
            inputs, targets = inputs.to(devices), targets.to(devices)
            outputs = self.main_model(inputs)

            # SingleLoss or MultiLoss
            if isinstance(criterion, SingleLoss):
                loss = criterion(outputs, targets)
            else:
                loss = criterion(
                    outputs,
                    targets,
                    prev_grad_h=self.prev_encoder_grad,
                    prev_grad_m=self.prev_main_grad,
                    alpha=alpha_val,
                    gamma=10.0,
                    prev_ratio=self.prev_ratio
                )

            losses.update(loss.item(), inputs.size(0))
            self.optimizer.zero_grad()
            loss.backward()

            # Watermark vs main gradient separation
            for param in self.main_model.parameters():
                if param.grad is not None:
                    wm_mask = (param.abs() < threshold).float()
                    main_mask = 1.0 - wm_mask
                    wm_grad_sum   += (param.grad.abs() * wm_mask).sum().item()
                    wm_count      += wm_mask.sum().item()
                    main_grad_sum += (param.grad.abs() * main_mask).sum().item()
                    main_count    += main_mask.sum().item()

            self.optimizer.step()

            # Accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total   += targets.size(0)

        main_train_acc = 100.0 * correct / total if total > 0 else 0.0

        # Update gradient stats
        grad_h = wm_grad_sum / wm_count if wm_count > 0 else 0.0
        grad_m = main_grad_sum / main_count if main_count > 0 else 0.0
        ratio  = (grad_m / grad_h) if (wm_count > 0 and grad_h > 1e-9) else 0.0

        self.prev_encoder_grad = torch.tensor(grad_h, device=devices)
        self.prev_main_grad    = torch.tensor(grad_m, device=devices)
        self.prev_ratio        = torch.tensor(ratio, device=devices)

        self.scheduler.step()

        return main_train_acc, losses.avg, alpha_val, criterion

    def validate_main_model(self, criterion, alpha_val):
        """
        Validate the main model => returns (main_val_acc, main_val_loss).
        """
        val_loss, main_acc = validate_main_model(
            model=self.main_model,
            val_loader=self.main_val_loader,
            criterion=criterion,
            device=devices,
            prev_grad_h=self.prev_encoder_grad,
            prev_grad_m=self.prev_main_grad,
            alpha=alpha_val,
            gamma=10.0,
            prev_ratio=self.prev_ratio
        )
        return main_acc, val_loss

    def train_autoencoder(self, epoch):
        """
        Train autoencoder for combined_autoencoder on auto_train_loader.
        """
        self.combined_autoencoder.train()

        # Freeze the decoder
        for param in self.combined_autoencoder.decoder.parameters():
            param.requires_grad = False

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.combined_autoencoder.encoder.parameters(), lr=self.config.lr)

        epoch_loss = 0.0
        loop = tqdm(self.auto_train_loader, desc=f"AE Epoch {epoch + 1}", leave=True)
        for images, _ in loop:
            images = images.to(devices)
            optimizer.zero_grad()
            _, decoded = self.combined_autoencoder(images)
            loss = criterion(decoded, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(self.auto_train_loader)
        return avg_loss

    def run(self):
        """
        Run watermark training.

        Returns:
            main_model state_dict, auto_val_loader

        """
        print(f"=== Watermarking {self.config.architecture_name} ===\n")
        print("Running watermark training from scratch...")

        # Initial embed
        fused_sd = embed_encoder_to_model(
            self.main_model.state_dict(),
            self.current_encoder,
            str(self.config.benign_seed)
        )
        self.main_model.load_state_dict(fused_sd, strict=True)
        print("Initial watermark embedding done. Starting iterative training...\n")

        for epoch in range(self.config.total_rounds):
            print(f"\n=== [ Round {epoch + 1}/{self.config.total_rounds} ] ===")

            # Train and validate main model
            main_train_acc, main_train_loss, alpha_val, criterion = self.train_main_model(epoch)
            main_val_acc, main_val_loss = self.validate_main_model(criterion, alpha_val)

            # Extract encoder and update combined autoencoder
            extracted_encoder = extract_encoder_from_model(
                self.main_model.state_dict(),
                self.current_encoder,
                str(self.config.benign_seed)
            )
            self.combined_autoencoder = combine_encoder_decoder(extracted_encoder, self.decoder_weights)
            extracted_ae_loss = evaluate_autoencoder_loss(
                self.combined_autoencoder.encoder.state_dict(),
                self.decoder_weights,
                self.auto_val_loader
            )

            # For non-final epochs, train the autoencoder.
            if epoch != self.config.total_rounds - 1:
                ae_train_loss = self.train_autoencoder(epoch)
                self.last_ae_train_loss = ae_train_loss
            else:
                ae_train_loss = extracted_ae_loss

            # Regardless of the epoch, re-embed the main model if the embedding interval is met.
            if (epoch + 1) % self.config.embed_interval == 0:
                self.current_encoder = deepcopy(self.combined_autoencoder.encoder.state_dict())
                fused_sd = embed_encoder_to_model(
                    self.main_model.state_dict(),
                    self.current_encoder,
                    str(self.config.benign_seed)
                )
                self.main_model.load_state_dict(fused_sd, strict=True)

            # Track log metrics
            self.main_train_losses.append(main_train_loss)
            self.main_val_losses.append(main_val_loss)
            self.ae_train_losses.append(ae_train_loss)
            self.ae_val_losses.append(extracted_ae_loss)
            # Debugging: compute loss correlations (can be omitted if not needed)
            corr_val = self.compute_pearson_correlation(self.main_val_losses, self.ae_val_losses)
            corr_train = self.compute_pearson_correlation(self.main_train_losses, self.ae_train_losses)
            self.loss_correlation_val.append(corr_val)
            self.loss_correlation_train.append(corr_train)

            print(f"Epoch {epoch + 1}: Main Train Acc {main_train_acc:.2f}%, Main Loss {main_train_loss:.3f}, "
                  f"Main Val Acc {main_val_acc:.2f}%, Main Val Loss {main_val_loss:.3f}, "
                  f"Extracted AE Recon({extracted_ae_loss:.4f})")
            print(f"Epoch {epoch + 1}: AE Recon Loss {ae_train_loss:.4f}")

            log_line = (f"{epoch + 1:<6}"
                        f"{main_train_acc:>14.2f}"
                        f"{main_train_loss:>14.3f}"
                        f"{main_val_acc:>12.2f}"
                        f"{main_val_loss:>12.3f}"
                        f"{extracted_ae_loss:>12.4f}\n")
            if epoch == 0 or ((epoch + 1) % 10 == 0):
                with open(self.log_file_path, "a") as lf:
                    lf.write(log_line)
                with open(self.ae_loss_file_path, "a") as lf:
                    lf.write(f"{epoch + 1:<6}{ae_train_loss:.4f}\n")

        # After training, perform final extraction and evaluation
        final_extracted_encoder = extract_encoder_from_model(
            self.main_model.state_dict(),
            self.current_encoder,
            str(self.config.benign_seed)
        )
        final_ae_loss = evaluate_autoencoder_loss(
            final_extracted_encoder,
            self.decoder_weights,
            self.auto_val_loader
        )

        self.save_final_results(final_extracted_encoder, final_ae_loss)

        PlotConfig.plot_watermark_metrics(
            main_train_losses=self.main_train_losses,
            main_val_losses=self.main_val_losses,
            ae_train_losses=self.ae_train_losses,
            ae_val_losses=self.ae_val_losses,
            loss_correlation_train=self.loss_correlation_train,
            loss_correlation_val=self.loss_correlation_val,
            result_dir=self.result_dir
        )

        return self.main_model.state_dict(), self.auto_val_loader

    def compute_pearson_correlation(self, main_losses, ae_losses):
        if len(main_losses) < 2 or len(ae_losses) < 2:
            return 0.0
        x = np.array(main_losses)
        y = np.array(ae_losses)
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            return 0.0
        corr_matrix = np.corrcoef(x, y)
        if np.isnan(corr_matrix).any():
            return 0.0
        return corr_matrix[0, 1]

    def save_final_results(self, final_extracted_encoder, final_ae_loss):
        arch = self.config.architecture_name
        final_main_path = os.path.join(self.result_dir, f"{arch}_final_main_model.pth")
        torch.save(self.main_model.state_dict(), final_main_path)
        print(f"\n==> Final main model saved at {final_main_path}")

        with open(self.log_file_path, "a") as lf:
            lf.write(f"\n==> Final main model saved.\n")

        print(f"Final extracted encoder AE loss: {final_ae_loss:.4f}")
        final_encoder_path = os.path.join(self.result_dir, f"{arch}_extracted_encoder.pth")
        torch.save(final_extracted_encoder, final_encoder_path)
        print(f"Final extracted encoder saved => {final_encoder_path}")
        print(f"Final AE Reconstruction Loss = {final_ae_loss:.4f}")
        with open(self.log_file_path, "a") as lf:
            lf.write(f"Final AE Reconstruction Loss = {final_ae_loss:.4f}\n")

        num_params = sum(p.numel() for p in self.main_model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {num_params}")
        with open(self.log_file_path, "a") as lf:
            lf.write(f"Trainable parameters: {num_params}\n")
        # The following codes are only useful for debugging otherwise can be omitted
        print("\n==== Embedding Extraction Report ====")
        report = report_embedding_extraction_misplacements(
            self.main_model.state_dict(),
            self.current_encoder,
            str(self.config.benign_seed)
        )
        extraction_report_path = os.path.join(self.result_dir, "embedding_extraction_report.txt")
        with open(extraction_report_path, "w") as f:
            f.write("==== Embedding Extraction Report ====\n\n")
            f.write("Fixed positions:\n")
            for pos in report["fixed_positions"]:
                f.write(f"{pos}\n")
            f.write("\n")
            if report["mismatches"]:
                f.write("Mismatched positions:\n")
                for mismatch in report["mismatches"]:
                    f.write(f"{mismatch}\n")
            else:
                f.write("All embedding positions match the extraction positions.\n")

        print(f"Extraction embedding report saved at {extraction_report_path}")

        if report["mismatches"]:
            print(f"WARNING: Found {len(report['mismatches'])} mismatched positions. See file for details.")
        else:
            print("All embedding positions match the extraction positions.")

        diff = verify_extraction_reinsertion(final_extracted_encoder, device=devices)
        print("\n==== Extraction-Reinsertion Verification ====")
        print("Printing vector difference (should be all zeros):", diff)

        details_path = os.path.join(self.result_dir, "extraction_reinsertion_details.txt")
        with open(details_path, "w") as df:
            df.write("Detailed element-by-element differences (index: difference):\n")
            for idx, value in enumerate(diff.flatten().tolist()):
                df.write(f"Index {idx}: {value}\n")
        print(f"Detailed extraction-reinsertion differences saved at {details_path}")

        if torch.sum(diff) == 0:
            print("Extraction and reinsertion verification successful: difference is zero.")
        else:
            print("Extraction and reinsertion verification FAILED.")
        # end of debugging

class SimulationAttackerManager:
    def __init__(self, trainer, config):
        self.watermarked_model = trainer.main_model
        self.decoder_weights   = trainer.decoder_weights
        self.seed              = config.benign_seed
        self.architecture_name = config.architecture_name
        self.eval_autoencoder  = trainer.eval_autoencoder
        self.config            = config

    def run_attacks(self):
        simulator = AttackSimulator(
            self.config,
            self.watermarked_model,
            self.decoder_weights,
            self.eval_autoencoder
        )
        simulator.simulate_attacks()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Watermark Model or run attacks.")
    parser.add_argument("--model_name", type=str, default="CNN",
                        help="Architecture name (e.g. CNN, MLP, ResNet18).")
    parser.add_argument("--total_rounds", type=int, default=4,
                        help="Number of training epochs.")
    parser.add_argument("--attacks_only", action="store_true",
                        help="If set, skip watermark training and run attacks only (must have a trained model).")
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config()
    config.watermark_enabled = True

    # Override from command line
    config.architecture_name = args.model_name
    config.total_rounds      = args.total_rounds

    trainer = WatermarkTrainer(config)

    if args.attacks_only:
        final_model_path = os.path.join(config.result_dir, f"{config.architecture_name}_final_main_model.pth")
        if not os.path.exists(final_model_path):
            print(f"No final main model found at '{final_model_path}'. Attack-only mode aborted.")
            return
        model_state = CheckpointLoader.load_checkpoint(final_model_path)
        trainer.main_model.load_state_dict(model_state)
    else:
        print("Running watermark training from scratch...")
        trainer.run()

    # Attacker simulation
    attacker_manager = SimulationAttackerManager(trainer, config)
    attacker_manager.run_attacks()

    print("\nAll steps completed.")
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleaned.")


if __name__ == "__main__":
    main()
