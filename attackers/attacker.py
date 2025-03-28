import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import prune
from configs.config import (
    devices, Config, DataLoaderFactory, DirectoryManager,
    CheckpointLoader, SeedManager
)
from configs.helper import evaluate_autoencoder_loss, extract_encoder_from_model
from attackers.aehelper import (
    embed_encoder_to_attacked_model,
    extract_encoder_from_attacked_model,
    combined_attacker_autoencoder
)

class AttackSimulator:
    """Simulates watermark-removal attacks on deep neural networks.

       Supports:
               - Pruning attack
                - Fine-tuning attack
                - Watermark overwriting attack
    It logs model performance, autoencoder losses, and manages re-embedding logic.
    """
    def __init__(self, config: Config, watermarked_model, benign_decoder, eval_autoencoder):
        """ Initialize the attack simulator with necessary components.

        Args:
            config (Config): Experimental configuration.
            watermarked_model (nn.Module): Watermarked model to attack.
            benign_decoder (nn.Module): Decoder from the benign autoencoder.
            eval_autoencoder (nn.Module): Full autoencoder used for evaluation.
        """

        self.config = config
        self.watermarked_model = watermarked_model
        self.benign_decoder = benign_decoder
        self.eval_autoencoder = eval_autoencoder
        self.device = devices

        # Set attacker seed for reproducibility
        SeedManager.set_attacker_seed(self.config.attacker_seed)

        self._load_attacker_components()
        self._setup_results_directory()
        #self._init_loss_log()

        # Main loss for classification tasks
        self.criterion_main = nn.CrossEntropyLoss()

    def _load_attacker_components(self):
        """Load attacker autoencoder components from checkpoints."""
        self.attacker_encoder_template = CheckpointLoader.load_checkpoint(self.config.attack_encoder_path)
        self.attacker_decoder = CheckpointLoader.load_checkpoint(self.config.attacker_decoder_path)

    def _setup_results_directory(self):
        """Set up results directory and log file paths."""
        self.results_dir = DirectoryManager.save_attacker_dir(self.config.architecture_name)
        os.makedirs(self.results_dir, exist_ok=True)
        self.overwriting_log_file = os.path.join(self.results_dir, "AttackerOverwritingLog.txt")
        self.fine_tuning_log_file = os.path.join(self.results_dir, "fine_tuning_attack_report.txt")
        self.pruning_log_file = os.path.join(self.results_dir, "pruning_attack_report.txt")


    def _train_epoch(self, model, data_loader, optimizer, loss_fn):
        """ Train a model for one epoch
        Args:
            model (nn.Module): Model to train.
            data_loader (DataLoader): Training data loader.
            optimizer (Optimizer): Model optimizer.
            loss_fn (Loss): Loss function to use.
        Returns:
            float: Training accuracy(%).
            float: Training loss(average).
        """
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            bs = inputs.size(0)
            total_loss += loss.item() * bs
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += bs

        train_acc = 100.0 * correct / total if total else 0.0
        train_loss = total_loss / total if total else 0.0
        return train_acc, train_loss

    def _evaluate(self, model, data_loader, loss_fn=None):
        """ Evaluate a model on a data loader.

        Args:
            model (nn.Module): Model to evaluate.
            data_loader (DataLoader): Data loader for evaluation.
            loss_fn (Loss): Loss function to using cross-entropy loss by default.

        Returns:
            float: Evaluation accuracy(%).
            float: average loss

        """
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets) if loss_fn else self.criterion_main(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += inputs.size(0)
        acc = 100.0 * correct / total if total else 0.0
        avg_loss = total_loss / total if total else 0.0
        return acc, avg_loss

    def _train_autoencoder_epoch(self, ae_model, data_loader, optimizer):
        """Train an autoencoder model for one epoch.
        Args:
            ae_model (nn.Module): Autoencoder model to train.
            data_loader (DataLoader): Training data loader.
            optimizer (Optimizer): Autoencoder optimizer.
        Returns:
            float: Average reconstruction loss(MSE)
        """
        ae_model.train()
        total_loss, total_samples = 0.0, 0
        criterion = nn.MSELoss()
        for images, _ in data_loader:
            images = images.to(self.device)
            optimizer.zero_grad()
            _, decoded = ae_model(images)
            loss = criterion(decoded, images)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
        return total_loss / total_samples if total_samples else 0.0

    def _eval_autoencoder_epoch(self, ae_model, data_loader):
        """Evaluate an autoencoder model on a data loader.
        Args:
            ae_model (nn.Module): Autoencoder model to evaluate.
            data_loader (DataLoader): Data loader for evaluation.
        Returns:
            float: Average MSE reconstruction loss
        """
        ae_model.eval()
        total_loss, total_samples = 0.0, 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(self.device)
                _, decoded = ae_model(images)
                loss = criterion(decoded, images)
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
        return total_loss / total_samples if total_samples else 0.0
    # Ref : this part of code is taken from DICTION project https://github.com/Bellafqira/DICTION
    def prune_model(self, model, prune_percent):
        """ Prune the model's weights and biases  using L1 unstructured pruning
         Args:
            model (nn.Module): Model to prune.
            prune_percent (float): Percentage of weights to prune (0–100).

        Returns:
            nn.Module: The pruned model.
        """
        amount = prune_percent / 100.0
        for _, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                prune.l1_unstructured(module, name="weight", amount=amount)
                prune.remove(module, "weight")
            if hasattr(module, "bias") and module.bias is not None:
                prune.l1_unstructured(module, name="bias", amount=amount)
                prune.remove(module, "bias")
        sum_zeros, sum_elements = 0, 0
        for _, param in model.named_parameters():
            sum_zeros += float(torch.sum(torch.eq(param, 0)))
            sum_elements += float(param.nelement())
        sparsity = 100.0 * sum_zeros / sum_elements
        print(f"Pruning {prune_percent}% => Sparsity: {sparsity:.4f}%")
        return model

    def _extract_encoder(self, model, encoder, seed):
        """Extract encoder state using a helper function."""
        return extract_encoder_from_model(model.state_dict(), encoder, str(seed), self.device)

    def pruning_attack(self, target_model, benign_val_loader, benign_auto_val_loader, benign_encoder):
        """
        Execute a pruning attack and log both terminal outputs and results to a file.
        """
        ae_loss_zero = 0.0
        prune_rates = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100]

        log_lines = []
        arch = self.config.architecture_name
        log_lines.append("=== Pruning Attack Report ===")
        log_lines.append(f"Architecture: {arch}")
        log_lines.append(f"{'Prune %':<15}{'val_acc':<15}{'val_loss':<15}{'ae_loss':<15}")

        for rate in prune_rates:
            model_copy = deepcopy(target_model)
            self.prune_model(model_copy, rate)
            val_acc, val_loss = self._evaluate(model_copy, benign_val_loader)
            extracted_enc = self._extract_encoder(model_copy, benign_encoder, self.config.benign_seed)

            if rate == 100:
                ae_loss_normal = evaluate_autoencoder_loss(extracted_enc, self.benign_decoder, benign_auto_val_loader)
                zeroed_enc = {k: torch.zeros_like(v) for k, v in extracted_enc.items()}
                ae_loss_zero = evaluate_autoencoder_loss(zeroed_enc, self.benign_decoder, benign_auto_val_loader)
                terminal_line = (f"Prune={rate}% => ValAcc={val_acc:.2f}%, ValLoss={val_loss:.4f}, "
                                 f"AE(normal)={ae_loss_normal:.4f}, AE(zeroed)={ae_loss_zero:.4f}")
                log_row = f"{rate:<15}{val_acc:<15.2f}{val_loss:<15.4f}{ae_loss_normal:<15.4f}"
            else:
                ae_loss = evaluate_autoencoder_loss(extracted_enc, self.benign_decoder, benign_auto_val_loader)
                terminal_line = f"Prune={rate}% => ValAcc={val_acc:.2f}%, ValLoss={val_loss:.4f}, AE={ae_loss:.4f}"
                log_row = f"{rate:<15}{val_acc:<15.2f}{val_loss:<15.4f}{ae_loss:<15.4f}"

            print(terminal_line)
            log_lines.append(log_row)
            if rate == 100:
                log_lines.append(f"==> 100% manually zeroed AE  {ae_loss_zero:<15.4f}")

        with open(self.pruning_log_file, "w") as f:
            f.write("\n".join(log_lines))
        print("\nPruning attack report saved =>", self.pruning_log_file)

    def fine_tuning_attack(self, target_model, attacker_train_loader, benign_val_loader,
                           benign_auto_val_loader, benign_encoder):
        """
        Execute a fine-tuning attack while logging epoch-by-epoch performance.
        """
        pre_acc, _ = self._evaluate(target_model, benign_val_loader)
        extracted_enc = self._extract_encoder(target_model, benign_encoder, self.config.benign_seed)
        pre_ae_loss = evaluate_autoencoder_loss(extracted_enc, self.benign_decoder, benign_auto_val_loader)
        print(f"Pre-Fine-Tuning => Acc={pre_acc:.2f}%, AE={pre_ae_loss:.4f}")

        optimizer = optim.Adam(
            target_model.parameters(),
            lr=self.config.attacker_lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.attacker_round,
            eta_min=self.config.eta_min
        )
        log_epoch_lines = []
        for epoch in range(self.config.attacker_round):
            train_acc, train_loss = self._train_epoch(target_model, attacker_train_loader, optimizer, self.criterion_main)
            scheduler.step()
            val_acc, val_loss = self._evaluate(target_model, benign_val_loader)
            extracted_enc = self._extract_encoder(target_model, benign_encoder, self.config.benign_seed)
            ae_loss = evaluate_autoencoder_loss(extracted_enc, self.benign_decoder, benign_auto_val_loader)
            print(f"Epoch {epoch + 1}: TrainAcc={train_acc:.2f}%, TrainLoss={train_loss:.4f}, "
                  f"ValAcc={val_acc:.2f}%, ValLoss={val_loss:.4f}, AE={ae_loss:.4f}")
            if epoch == 0 or ((epoch + 1) % 10 == 0):
                line = f"{epoch+1:<10}{train_acc:>10.2f}{train_loss:>12.4f}{val_acc:>10.2f}{val_loss:>12.4f}{ae_loss:>12.4f}"
                log_epoch_lines.append(line)

        post_acc, _ = self._evaluate(target_model, benign_val_loader)
        extracted_enc = self._extract_encoder(target_model, benign_encoder, self.config.benign_seed)
        post_ae_loss = evaluate_autoencoder_loss(extracted_enc, self.benign_decoder, benign_auto_val_loader)
        print(f"Post-Fine-Tuning => Acc={post_acc:.2f}%, AE={post_ae_loss:.4f}")

        log_lines = []
        arch = self.config.architecture_name
        log_lines.append("=== Fine-Tuning Attack Report ===")
        log_lines.append(f"Architecture: {arch}")
        log_lines.append(f"Pre-Fine-Tuning => Acc={pre_acc:.2f}%, AE={pre_ae_loss:.4f}")
        header = f"{'Epoch':<10}{'TrainAcc':>10}{'TrainLoss':>12}{'ValAcc':>10}{'ValLoss':>12}{'ReconLoss':>12}"
        log_lines.append(header)
        log_lines.extend(log_epoch_lines)
        log_lines.append(f"Post-Fine-Tuning => Acc={post_acc:.2f}%, AE={post_ae_loss:.4f}")
        with open(self.fine_tuning_log_file, "w") as f:
            f.write("\n".join(log_lines))
        print("\nFine-tuning attack report saved =>", self.fine_tuning_log_file)

    def watermark_overwriting_attack(self, target_model, attacker_main_train_loader, attacker_main_val_loader,
                                     attacker_ae_train_loader, attacker_ae_val_loader,
                                     benign_auto_val_loader, benign_encoder, benign_seed):
        """
        Execute a watermark overwriting attack, including periodic re-embedding, and log details.
        """
        pre_main_acc, pre_main_loss = self._evaluate(target_model, attacker_main_val_loader)
        print(f"[Pre-Attack] => MainValAcc={pre_main_acc:.2f}%, MainValLoss={pre_main_loss:.4f}")
        extracted_enc = self._extract_encoder(target_model, benign_encoder, benign_seed)
        pre_benign_ae_loss = evaluate_autoencoder_loss(extracted_enc, self.benign_decoder, benign_auto_val_loader)
        print(f"[Pre-Attack] => HufuNet AE Recon={pre_benign_ae_loss:.4f}")

        log_lines = []
        log_lines.append("=== Watermark Overwriting Attack Report ===")
        log_lines.append(f"[Pre-Attack] => MainValAcc={pre_main_acc:.2f}%, MainValLoss={pre_main_loss:.4f}")
        log_lines.append(f"[Pre-Attack] => HufuNet AE Recon={pre_benign_ae_loss:.4f}")
        header = f"{'Round':<10}{'TrainAcc':>10}{'TrainLoss':>12}{'ValAcc':>10}{'ValLoss':>12}{'AE_Train':>12}{'AE_Val':>10}"
        log_lines.append(header)

        optimizer = optim.Adam(
            target_model.parameters(),
            lr=self.config.attacker_lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.attacker_round,
            eta_min=self.config.eta_min
        )
        total_rounds = self.config.attacker_round
        round_log_lines = []
        for round_num in range(1, total_rounds + 1):
            print(f"\n[Round {round_num}/{total_rounds}]")
            print("Embedding attacker watermark (attacker seed) into main model ...")
            fused_sd = embed_encoder_to_attacked_model(
                target_model.state_dict(),
                self.attacker_encoder_template,
                str(self.config.attacker_seed),
                self.device
            )
            target_model.load_state_dict(fused_sd, strict=True)
            train_acc, train_loss = self._train_epoch(target_model, attacker_main_train_loader, optimizer, self.criterion_main)
            scheduler.step()
            val_acc, val_loss = self._evaluate(target_model, attacker_main_val_loader)
            print(f"[Round {round_num}] => MainTrainAcc={train_acc:.2f}%, TrainLoss={train_loss:.4f}, "
                  f"MainValAcc={val_acc:.2f}%, MainValLoss={val_loss:.4f}")

            extracted_attacker_enc = extract_encoder_from_attacked_model(
                target_model.state_dict(),
                self.attacker_encoder_template,
                str(self.config.attacker_seed),
                self.device
            )
            attacker_ae = combined_attacker_autoencoder(extracted_attacker_enc, self.attacker_decoder).to(self.device)
            for param in attacker_ae.decoder.parameters():
                param.requires_grad = False
            ae_optimizer = optim.Adam(attacker_ae.encoder.parameters(), lr=self.config.attacker_lr)
            train_loss_ae = self._train_autoencoder_epoch(attacker_ae, attacker_ae_train_loader, ae_optimizer)
            val_loss_ae = self._eval_autoencoder_epoch(attacker_ae, attacker_ae_val_loader)
            print(f"=> AttackerAE TrainLoss={train_loss_ae:.4f}, ValLoss={val_loss_ae:.4f}")

            if round_num != total_rounds and (round_num % self.config.embed_interval == 0):
                print("Re-embedding updated attacker AE’s encoder into main model ...")
                updated_ae_encoder = attacker_ae.encoder.state_dict()
                fused_sd2 = embed_encoder_to_attacked_model(
                    target_model.state_dict(),
                    updated_ae_encoder,
                    str(self.config.attacker_seed),
                    self.device
                )
                target_model.load_state_dict(fused_sd2, strict=True)
                self.attacker_encoder_template = deepcopy(updated_ae_encoder)
            else:
                print("escaping re-embedding as embedding conditions not met.")

            if round_num == 1 or (round_num % 10 == 0):
                row = f"{round_num:<10}{train_acc:>10.2f}{train_loss:>12.4f}{val_acc:>10.2f}{val_loss:>12.4f}{train_loss_ae:>12.4f}{val_loss_ae:>10.4f}"
                round_log_lines.append(row)

        log_lines.extend(round_log_lines)
        post_main_acc, post_main_loss = self._evaluate(target_model, attacker_main_val_loader)
        print(f"Post-Attack => MainValAcc={post_main_acc:.2f}%, AERecon={post_main_loss:.4f}")
        final_extracted_enc = self._extract_encoder(target_model, benign_encoder, benign_seed)
        final_benign_ae_loss = evaluate_autoencoder_loss(final_extracted_enc, self.benign_decoder, benign_auto_val_loader)
        print(f"AE evaluated on  HufuNet  Recon: {final_benign_ae_loss:.4f}")
        log_lines.append(f"Post-Attack => MainValAcc={post_main_acc:.2f}%, MainValLoss={post_main_loss:.4f}")
        log_lines.append(f"AE evaluated on  HufuNet  Recon: {final_benign_ae_loss:.4f}")

        with open(self.overwriting_log_file, "w") as lf:
            lf.write("\n".join(log_lines))
        print(f"\nWatermark Overwriting Attack logs saved => {self.overwriting_log_file}")

    def simulate_attacks(self):
        """
        Orchestrate all attacks: Pruning, Fine-Tuning, and Watermark Overwriting.
        """
        arch = self.config.architecture_name
        attacker_train_loader, attacker_val_loader = DataLoaderFactory.get_model_datasets(
            arch, self.config, mode="attacker"
        )
        attacker_ae_train_loader, attacker_ae_val_loader = DataLoaderFactory.get_autoencoder_datasets(
            "AttackerAE", self.config
        )
        _, benign_val_loader = DataLoaderFactory.get_model_datasets(
            arch, self.config, mode="benign"
        )
        _, benign_auto_val_loader = DataLoaderFactory.get_autoencoder_datasets("HufuNet", self.config)

        target_model = deepcopy(self.watermarked_model)
        benign_encoder_template = self.eval_autoencoder.encoder.state_dict()
        benign_seed_str = str(self.config.benign_seed)

        print("\n=== Starting Pruning Attack ===\n")
        self.pruning_attack(
            deepcopy(target_model),
            benign_val_loader,
            benign_auto_val_loader,
            benign_encoder_template
        )

        print("\n=== Starting Fine-Tuning Attack ===\n")
        self.fine_tuning_attack(
            deepcopy(target_model),
            attacker_train_loader,
            benign_val_loader,
            benign_auto_val_loader,
            benign_encoder_template
        )

        print("\n=== Starting Watermark Overwriting Attack ===\n")
        self.watermark_overwriting_attack(
            deepcopy(target_model),
            attacker_train_loader,
            attacker_val_loader,
            attacker_ae_train_loader,
            attacker_ae_val_loader,
            benign_auto_val_loader,
            benign_encoder_template,
            benign_seed_str
        )
