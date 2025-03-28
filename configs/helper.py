import hashlib
import hmac

import torch
import torch.nn as nn
from configs.config import devices
from models.HufuNet import HufuNet

# Disable CuDNN for deterministic behavior
torch.backends.cudnn.enabled = False

"""
Credits: Loss functions and AverageMeter utility adapted from:
https://github.com/lvpeizhuo/HufuNet
"""
class MultiLoss(nn.Module):
    """
    Custom multi-term loss used for dynamic watermarking.
    Adds regularization based on prior gradients and adaptive coefficients.
    """
    def __init__(self, init_a_constant=0.6523, init_b=0.0000800375825259):
        super().__init__()
        self.init_a_constant = init_a_constant
        self.init_b = init_b

    def forward(self, outputs, targets, prev_grad_h, prev_grad_m, alpha, gamma, prev_ratio):
        """
        Computes the multi-component loss.

        Args:
            outputs (Tensor): Model predictions.
            targets (Tensor): Ground truth labels.
            prev_grad_h (float): Previous gradient from host.
            prev_grad_m (float): Previous gradient from model.
            alpha (float): Regularization coefficient.
            gamma (float): Not used in this function but likely relevant elsewhere.
            prev_ratio (float): Previous loss ratio metric.

        Returns:
            Tensor: The combined loss value.
        """
        single_loss = nn.CrossEntropyLoss().to(devices)
        if prev_grad_h <= 0 or prev_grad_m <= 0:
            raise ValueError("prev_grad_h and prev_grad_m must be > 0.")
        loss = single_loss(outputs, targets)

        ratio_beta1 = torch.abs(torch.as_tensor(prev_grad_m, device=devices) / torch.as_tensor(prev_grad_h, device=devices))
        ratio_beta2 = prev_grad_m * torch.abs(1.0 / (self.init_a_constant - loss)) / self.init_b

        custom_loss = (loss +
                       alpha * (3 - ratio_beta1) ** 2 +
                       alpha * (1.5 - prev_ratio) ** 2 +
                       torch.exp(-ratio_beta2) * ratio_beta2)
        return custom_loss


class SingleLoss(nn.Module):
    """
    Standard cross-entropy loss.
    """
    def __init__(self):
        super().__init__()
        self.single_loss = nn.CrossEntropyLoss().to(devices)

    def forward(self, outputs, targets):
        return self.single_loss(outputs, targets)


class AverageMeter:
    """
    Tracks and updates average statistics (e.g., loss or accuracy) over time.
    """
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def validate_main_model(model, val_loader, criterion, device,
             prev_grad_h=None, prev_grad_m=None, alpha=None, gamma=None, prev_ratio=None):
    """
     Validate the main model on a given validation loader.

    Returns:
        tuple: (avg_loss, accuracy) across the validation set.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if isinstance(criterion, MultiLoss):
                loss = criterion(outputs, targets, prev_grad_h, prev_grad_m, alpha, gamma, prev_ratio)
            else:
                loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate_autoencoder(model, val_loader):
    """
    Evaluate a HufuNet autoencoder’s reconstruction MSE loss.

    Returns:
        float: Average reconstruction loss.
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(devices)
            _, decoded = model(images)
            loss = criterion(decoded, images)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def count_params(state_dict):
    """
    Count total number of parameters in a model state_dict.

    Returns:
        int: Total parameter count.
    """
    return sum(param.numel() for param in state_dict.values())


# --- HMAC-based Unique Position Computation ---
def compute_hmac_position(i, total_dim, seed, counter=0):
    """
    Compute a single position using HMAC-based hashing.

    Returns:
        int: Position index in flattened host model weights.
    """
    msg = f"{i}_{counter}"
    mac = hmac.new(seed.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256)
    return int(mac.hexdigest(), 16) % total_dim


def compute_unique_positions_hmac(L, total_dim, seed):
    """
    Compute unique positions using HMAC, avoiding collisions.

    Returns:
        List[int]: Unique indices in the host model.
    """
    positions = []
    for i in range(L):
        counter = 0
        pos = compute_hmac_position(i, total_dim, seed, counter)
        while pos in positions:
            counter += 1
            pos = compute_hmac_position(i, total_dim, seed, counter)
        positions.append(pos)
    return positions


def embed_encoder_to_model(model_weights, encoder, seed, device=devices):
    """
    Embed the encoder parameters into the main model’s weights using HMAC.

    Returns:
        dict: Modified model state_dict with embedded encoder.
    """
    # Use a consistent ordering: sort the keys for the host region.
    weight_keys = sorted([k for k in model_weights if "weight" in k and model_weights[k].numel() > 50])
    total_main = sum(model_weights[k].numel() for k in weight_keys)
    print(f"Total host parameters (>50 only): {total_main}")

    main_flat = torch.empty(total_main, device=device)
    meta = []
    start_idx = 0
    for k in weight_keys:
        tensor = model_weights[k].to(device)
        n = tensor.numel()
        main_flat[start_idx:start_idx + n] = tensor.view(-1)
        meta.append((k, start_idx, n, tensor.shape))
        start_idx += n

    # Flatten the encoder using sorted keys for consistency.
    enc_keys = sorted(encoder.keys())
    encoder_flat = torch.cat([encoder[k].to(device).view(-1) for k in enc_keys])
    L = encoder_flat.numel()
    #print(f"Total encoder parameters: {L}")

    if L > total_main:
        raise ValueError("Not enough host capacity to embed the entire encoder!")
    positions = compute_unique_positions_hmac(L, total_main, seed)
    for i in range(L):
        pos = positions[i]
        main_flat[pos] = encoder_flat[i]

    # Rebuild the new state_dict for main model weights.
    new_model = {}
    for (k, start, n, shape) in meta:
        new_model[k] = main_flat[start:start + n].view(shape)
    for k in model_weights:
        if k not in new_model:
            new_model[k] = model_weights[k]
    return new_model


def extract_encoder_from_model(model_state, encoder_template, seed, device=devices):
    """
    Extract encoder weights from the host model using HMAC positions.

    Returns:
        dict: Reconstructed encoder state_dict.
    """
    weight_keys = sorted([k for k in model_state if "weight" in k and model_state[k].numel() > 50])
    total_main = sum(model_state[k].numel() for k in weight_keys)
    main_flat = torch.empty(total_main, device=device)
    start_idx = 0
    for k in weight_keys:
        tensor = model_state[k].to(device)
        n = tensor.numel()
        main_flat[start_idx:start_idx + n] = tensor.view(-1)
        start_idx += n

    # Flatten the encoder template using sorted keys.
    enc_keys = sorted(encoder_template.keys())
    enc_shapes = []
    total_enc = 0
    for k in enc_keys:
        tensor = encoder_template[k].to(device)
        n = tensor.numel()
        enc_shapes.append((k, n, tensor.shape))
        total_enc += n
    L = total_enc
    # For debugging:
    #print(f"Total encoder parameters to extract: {L}")

    positions = compute_unique_positions_hmac(L, total_main, seed)
    extracted_flat = torch.empty(L, device=device)
    for i in range(L):
        pos = positions[i]
        extracted_flat[i] = main_flat[pos]

    # Reconstruct the encoder state_dict using the same sorted order.
    extracted_encoder = {}
    start_idx = 0
    for (k, n, shape) in enc_shapes:
        extracted_encoder[k] = extracted_flat[start_idx:start_idx + n].view(shape).cpu()
        start_idx += n
    return extracted_encoder

def extract_dummy_encoder(host_state, dummy_encoder_template, seed, device=devices):
    """
    Extract encoder from a non-watermarked model using a dummy encoder template.

    Returns:
        dict: Extracted encoder.
    """
    return extract_encoder_from_model(host_state, dummy_encoder_template, seed, device=device)

def combine_encoder_decoder(extracted_encoder, decoder, device=devices):
    """
    Combine an extracted encoder and provided decoder into a HufuNet instance.

    Returns:
        HufuNet: Autoencoder model.
    """
    autoencoder = HufuNet().to(device)
    autoencoder.encoder.load_state_dict(extracted_encoder, strict=True)
    autoencoder.decoder.load_state_dict(decoder, strict=True)
    return autoencoder


def evaluate_autoencoder_loss(extracted_encoder_state, decoder_weights, auto_val_loader):
    """
    Evaluate loss of an autoencoder built from extracted encoder and given decoder.

    Returns:
        float: Reconstruction loss
    """
    autoencoder = combine_encoder_decoder(extracted_encoder_state, decoder_weights)
    autoencoder.eval()
    for param in autoencoder.decoder.parameters():
        param.requires_grad = False
    loss = validate_autoencoder(autoencoder, auto_val_loader)
    return loss

# For debugging
def check_embed_watermark(model_weights, encoder, decoder, seed, device=devices):
    """
    Debug utility to verify embedding consistency.

    Returns:
        dict: New model with encoder embedded.
    """
    main_param_count = count_params(model_weights)
    encoder_param_count = count_params(encoder)
    decoder_param_count = count_params(decoder)
    print(f"[CHECK BEFORE EMBEDDING] Main model parameters: {main_param_count}, "
          f"Encoder parameters: {encoder_param_count}, Decoder parameters: {decoder_param_count}")
    new_model = embed_encoder_to_model(model_weights, encoder, seed, device)
    new_main_param_count = count_params(new_model)
    print(f"[CHECK AFTER EMBEDDING] Main model parameters: {new_main_param_count}")
    if new_main_param_count != main_param_count:
        print("Warning: The parameter count changed after embedding!")
    else:
        print("Embedding successful: parameter count remains consistent.")
    return new_model


def verify_extraction_reinsertion(extracted_encoder, device=devices):
    """
    Sanity check: verify that reloading extracted encoder produces same tensor values.

    Returns:
        Tensor: Absolute difference between original and reloaded encoder vectors.

    """
    keys = sorted(extracted_encoder.keys())
    x1 = torch.cat([extracted_encoder[k].view(-1) for k in keys]).to(device)
    model = HufuNet().to(device)
    model.encoder.load_state_dict(extracted_encoder, strict=True)
    encoder_state = model.encoder.state_dict()
    x2 = torch.cat([encoder_state[k].view(-1) for k in sorted(encoder_state.keys())]).to(device)
    diff = torch.abs(x2 - x1)
    return diff


def report_embedding_extraction_misplacements(model_weights, encoder, seed, device=devices):
    """
    Report mismatches between embedding and extraction positions, if any.

    Returns:
        dict: Contains position list and mismatches (if any).
    """
    weight_keys = sorted([k for k in model_weights if "weight" in k and model_weights[k].numel() > 50])
    total_main = sum(model_weights[k].numel() for k in weight_keys)
    enc_keys = sorted(encoder.keys())
    encoder_flat = torch.cat([encoder[k].to(device).view(-1) for k in enc_keys])
    L = encoder_flat.numel()
    positions_embedding = compute_unique_positions_hmac(L, total_main, seed)
    positions_extraction = compute_unique_positions_hmac(L, total_main, seed)
    mismatches = []
    for i, (p_emb, p_ext) in enumerate(zip(positions_embedding, positions_extraction)):
        if p_emb != p_ext:
            mismatches.append((i, p_emb, p_ext))
    report = {
        "fixed_positions": positions_embedding,
        "mismatches": mismatches
    }
    return report
