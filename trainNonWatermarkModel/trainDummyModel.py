
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from configs.config import devices, Config, DataLoaderFactory, ModelFactory, SeedManager
from train_epoch import train_model, validate_model



class DummyModelTrainer:
    """
    This script trains a dummy model on benign data for a one epoch unless specified otherwise.
    """
    def __init__(self, config: Config):
        self.config = config

        # Set random seed for reproducibility.
        SeedManager.set_benign_seed(config.benign_seed)

       # Initialize model and data loaders.
        self.model = ModelFactory.get_model(config.architecture_name).to(devices)
        self.train_loader, self.test_loader = DataLoaderFactory.get_model_datasets(
            config.architecture_name,
            config=config,
            mode="benign"
        )

        # printing model name, dataset name and batch size and total parameters
        main_dataset_name = config.architecture_dataset_map[config.architecture_name]
        main_batch_size   = self.train_loader.batch_size
        print(f"Training MainModel: {config.architecture_name} "
              f"using dataset: {main_dataset_name} "
              f"with batch size: {main_batch_size}")

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_params}")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.lr,   # default from Config
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.total_rounds,
            eta_min=config.eta_min
        )

        # Ensure model directory exists.
        os.makedirs(self.config.model_dir, exist_ok=True)

    def run(self):
        for epoch in range(1, self.config.total_rounds + 1):
            print(f"Epoch {epoch}/{self.config.total_rounds}")
            train_loss, train_acc = train_model(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                devices,
                desc="Training"
            )
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

            val_loss, val_acc = validate_model(
                self.model,
                self.test_loader,
                self.criterion,
                devices,
                desc=f"Validation Epoch {epoch}"
            )
            print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%")

            self.scheduler.step()

        self.save_model()
        self.cleanup()

    def save_model(self):
        model_path = os.path.join(self.config.model_dir, f"{self.config.architecture_name}_dummy.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def cleanup(self):
        del self.model, self.train_loader, self.test_loader, self.optimizer, self.scheduler
        torch.cuda.empty_cache()
        gc.collect()
        print("Memory cleaned up.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Dummy Model with minimal arguments.")
    parser.add_argument("--model_name", type=str, default="CNN",
                        help="Architecture name (e.g. CNN, MLP, ResNet18).")
    parser.add_argument("--total_rounds", type=int, default=1,
                        help="Number of training epochs.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load default config.
    cfg = Config()

    # Override only these two fields from the command line.
    cfg.architecture_name = args.model_name
    cfg.total_rounds = args.total_rounds

    trainer = DummyModelTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
