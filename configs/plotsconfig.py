import os
import matplotlib.pyplot as plt
import numpy as np
import torch


class PlotConfig:
    """
    A class that provides static methods for plotting in both autoencoder
    and watermark trainer contexts.
    """

    @staticmethod
    def visualize_reconstructions(autoencoder, test_loader, devices, model_name, plot_dir, num_images=3):
        """
        Visualizes and saves a reconstruction plot for an autoencoder.
        """
        autoencoder.eval()
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(devices)
                _, decoded = autoencoder(images)
                break  # Only visualize the first batch

        images = images.cpu().numpy()
        decoded = decoded.cpu().numpy()

        plt.figure(figsize=(20, 4))
        for i in range(num_images):
            # Original
            plt.subplot(2, num_images, i + 1)
            plt.imshow(images[i].squeeze(), cmap='gray')
            plt.title("Original")
            plt.axis('off')

            # Reconstructed
            plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(decoded[i].squeeze(), cmap='gray')
            plt.title("Reconstructed")
            plt.axis('off')

        # Save the visualization with a unique name.
        recon_save_path = os.path.join(plot_dir, f"{model_name}_reconstructions.png")
        plt.savefig(recon_save_path, dpi=400)
        plt.close()
        print(f"Reconstruction plot saved to: {recon_save_path}")

    @staticmethod
    def plot_auto_losses(train_losses, val_losses, total_rounds, plot_dir, model_name):
        """
        Plots and saves the training and validation losses over epochs for an autoencoder.
        """
        epochs = range(1, total_rounds + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_losses, label="Train Loss", marker='o')
        plt.plot(epochs, val_losses, label="Validation Loss", marker='s')
        plt.title(f"{model_name} Reconstruction Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()

        png_save_path = os.path.join(plot_dir, f"{model_name}_loss_plot.png")
        pdf_save_path = os.path.join(plot_dir, f"{model_name}_loss_plot.pdf")
        plt.savefig(png_save_path, dpi=400)
        plt.savefig(pdf_save_path, dpi=400)
        plt.close()
        print(f"Loss plots saved to:\n  {png_save_path}\n  {pdf_save_path}")

    @staticmethod
    def plot_non_watermark(main_train_losses, main_val_losses, total_rounds, plot_dir, model_name):
        """
        Plots and saves the training and validation losses over epochs for a non-watermark trainer.
        This method is tailored for the non-watermark training context.
        """
        epochs = range(1, total_rounds + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, main_train_losses, label="Main Train Loss", marker='o')
        plt.plot(epochs, main_val_losses, label="Main Val Loss", marker='s')
        plt.title(f"{model_name} Non-Watermark Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        png_save_path = os.path.join(plot_dir, f"{model_name}_nonwatermark_loss.png")
        pdf_save_path = os.path.join(plot_dir, f"{model_name}_nonwatermark_loss.pdf")
        plt.savefig(png_save_path, dpi=400)
        plt.savefig(pdf_save_path, format='pdf', dpi=400)
        plt.close()
        print(f"Non-watermark loss plots saved to:\n  {png_save_path}\n  {pdf_save_path}")
    @staticmethod
    def plot_watermark_metrics(
        main_train_losses,
        main_val_losses,
        ae_train_losses,
        ae_val_losses,
        loss_correlation_train,
        loss_correlation_val,
        result_dir
    ):
        """
        Refactored version of 'plot_all_metrics' from WatermarkTrainer.
        Creates and saves:
          1) AE loss (train vs. val)
          2) Main model loss (train vs. val)
          3) Correlation (train & val) between AE and Main model losses
          4) Combined plot of AE + Main model losses
        """
        num_epochs = len(main_train_losses)
        if num_epochs < 1:
            print("No training epochs recorded; skipping plots.")
            return

        epochs = np.arange(1, num_epochs + 1)


        #  AE Loss Over Epochs (Train vs. Val)
        fig1 = plt.figure(figsize=(7, 5))
        plt.plot(epochs, ae_train_losses, label="AE Train Loss", marker='o')
        plt.plot(epochs, ae_val_losses, label="AE Val Loss", marker='s')
        plt.title("AE Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()

        ae_loss_png = os.path.join(result_dir, "ae_loss.png")
        ae_loss_pdf = os.path.join(result_dir, "ae_loss.pdf")
        #plt.savefig(ae_loss_png, dpi=400, bbox_inches='tight')
        plt.savefig(ae_loss_pdf, format='pdf', dpi=400)
        plt.close(fig1)


        #  Main Model Loss Over Epochs (Train vs. Val)

        fig2 = plt.figure(figsize=(7, 5))
        plt.plot(epochs, main_train_losses, label="Main Train Loss", marker='o')
        plt.plot(epochs, main_val_losses, label="Main Val Loss", marker='s')
        plt.title("Main Model Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        main_loss_png = os.path.join(result_dir, "main_model_loss.png")
        main_loss_pdf = os.path.join(result_dir, "main_model_loss.pdf")
        plt.savefig(main_loss_png, dpi=400, bbox_inches='tight')
        plt.savefig(main_loss_pdf, format='pdf', dpi=400, bbox_inches='tight')
        plt.close(fig2)

        # 3) Correlation (Train + Val)

        fig3 = plt.figure(figsize=(7, 5))
        plt.plot(epochs, loss_correlation_train, label="Train Corr", marker='o')
        plt.plot(epochs, loss_correlation_val, label="Val Corr", marker='s')
        plt.title("Correlation (Main Loss vs. AE Loss)")
        plt.xlabel("Epoch")
        plt.ylabel("Pearson Corr")
        plt.legend()

        corr_png = os.path.join(result_dir, "loss_correlation.png")
        corr_pdf = os.path.join(result_dir, "loss_correlation.pdf")
        plt.savefig(corr_png, dpi=400, bbox_inches='tight')
        plt.savefig(corr_pdf, format='pdf', dpi=400, bbox_inches='tight')
        plt.close(fig3)


        #  Combined Plot
        fig4 = plt.figure(figsize=(7, 5))
        plt.plot(epochs, ae_train_losses, label="AE Train Loss", marker='o')
        plt.plot(epochs, ae_val_losses, label="AE Val Loss", marker='s')
        plt.plot(epochs, main_train_losses, label="Main Train Loss", marker='^')
        plt.plot(epochs, main_val_losses, label="Main Val Loss", marker='v')
        plt.title("Combined AE & Main Model Loss (Train & Val)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        combined_png = os.path.join(result_dir, "combined_loss.png")
        combined_pdf = os.path.join(result_dir, "combined_loss.pdf")
        plt.savefig(combined_png, dpi=400, bbox_inches='tight')
        plt.savefig(combined_pdf, format='pdf', dpi=400, bbox_inches='tight')
        plt.close(fig4)
        print(f"Plots saved as separate PNG + PDF in under results directory\n")


