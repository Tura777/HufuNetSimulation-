#!/usr/bin/env python3

import sys
import subprocess

# Valid models for watermark, nonwatermark, dummy
VALID_WM_MODELS = ["MLP", "MLP_RIGA", "CNN", "ResNet18"]
# Valid models for autoencoder
VALID_AE_MODELS = ["HufuNet", "AttackerAE"]


def get_model_name(args):
    """
    Helper to extract the value of --model_name <VALUE> from args.
    Returns None if not found.
    """
    for i, arg in enumerate(args):
        if arg == "--model_name" and i + 1 < len(args):
            return args[i + 1]
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [watermark | nonwatermark | autoencoder | dummy] [options...]")
        print("Examples:")
        print("  python run.py watermark --model_name CNN --total_rounds 5")
        print("  python run.py watermark --model_name MLP_RIGA --attacks_only")
        print("  python run.py nonwatermark --model_name MLP --total_rounds 3")
        print("  python run.py autoencoder --model_name HufuNet --total_rounds 10")
        print("  python run.py dummy --model_name CNN --total_rounds 4")
        sys.exit(1)

    command = sys.argv[1]  # e.g. "watermark", "nonwatermark", "autoencoder", or "dummy"
    args = sys.argv[2:]  # everything after the first argument

    # Extract --model_name if provided
    model_name = get_model_name(args)

    # Decide what to do based on the command:
    if command == "watermark":
        # Validate model_name (if provided)
        if model_name and model_name not in VALID_WM_MODELS:
            print(f"[Error] For 'watermark', model_name must be one of: {VALID_WM_MODELS}")
            sys.exit(1)

        # Always call trainMainModel.py for watermark
        # That script itself checks if --attacks_only is present
        subprocess.run(["python", "trainWatermarkModels/trainMainModel.py"] + args)

    elif command == "nonwatermark":
        if model_name and model_name not in VALID_WM_MODELS:
            print(f"[Error] For 'nonwatermark', model_name must be one of: {VALID_WM_MODELS}")
            sys.exit(1)

        subprocess.run(["python", "trainNonWatermarkModel/trainNonWMMainModel.py"] + args)

    elif command == "autoencoder":
        if model_name and model_name not in VALID_AE_MODELS:
            print(f"[Error] For 'autoencoder', model_name must be one of: {VALID_AE_MODELS}")
            sys.exit(1)

        subprocess.run(["python", "trainAutoEncoders/autoencoder_trainer.py"] + args)

    elif command == "dummy":
        if model_name and model_name not in VALID_WM_MODELS:
            print(f"[Error] For 'dummy', model_name must be one of: {VALID_WM_MODELS}")
            sys.exit(1)

        subprocess.run(["python", "trainNonWatermarkModel/trainDummyModel.py"] + args)

    else:
        print(f"[Error] Unknown command: {command}")
        print("Valid commands: watermark, nonwatermark, autoencoder, dummy")
        sys.exit(1)


if __name__ == "__main__":
    main()
