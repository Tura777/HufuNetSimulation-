import hashlib
import hmac
import torch
from configs.config import devices
from models.attackerae import AttackerAE

# Disable CuDNN for deterministic results
torch.backends.cudnn.enabled = False

# Cache for computed HMAC positions to avoid redundant computation
_positions_cache = {}

def count_params(state_dict):
    """
    Count the total number of parameters in a given model state dictionary.

    Args:
        state_dict (dict): Model state_dict.

    Returns:
        int: Total number of parameters (elements).
    """
    return sum(param.numel() for param in state_dict.values())

def compute_hmac_position(i, total_dim, seed, counter=0):
    """
    Compute a single deterministic position using HMAC hashing.

    Args:
        i (int): Index of the parameter.
        total_dim (int): Total number of host parameters.
        seed (str): Secret seed for HMAC.
        counter (int): Collision counter for uniqueness.

    Returns:
        int: Position index in the host model's flattened weights.
    """
    msg = f"{i}_{counter}"
    mac = hmac.new(seed.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256)
    return int(mac.hexdigest(), 16) % total_dim

def compute_unique_positions_hmac(loc, total_dim, seed):
    """
    Compute `loc` unique HMAC-based positions in a tensor of length `total_dim`.
    Uses caching to avoid recomputation.

    Args:
        loc (int): Number of positions to compute.
        total_dim (int): Length of host parameter vector.
        seed (str): Secret seed for HMAC.

    Returns:
        List[int]: Unique HMAC-based embedding positions.
    """
    key = (loc, total_dim, seed)
    if key in _positions_cache:
        return _positions_cache[key]
    positions = []
    positions_set = set()
    for i in range(loc):
        counter = 0
        pos = compute_hmac_position(i, total_dim, seed, counter)
        while pos in positions_set:
            counter += 1
            pos = compute_hmac_position(i, total_dim, seed, counter)
        positions.append(pos)
        positions_set.add(pos)
    _positions_cache[key] = positions
    return positions

def embed_encoder_to_attacked_model(model_weights, encoder, seed, device=devices):
    """
    Embed the encoder parameters into a host model (attacked model).
    Only host weights with >25 parameters are considered for embedding.

    Args:
        model_weights (dict): Host model's state_dict.
        encoder (dict): Encoder model's state_dict to embed.
        seed (str): Secret seed for HMAC position encoding.
        device (torch.device): Target device for computation.

    Returns:
        dict: Modified host model weights with embedded encoder.
    """
    weight_keys = sorted([k for k in model_weights if "weight" in k and model_weights[k].numel() > 25])
    total_main = sum(model_weights[k].numel() for k in weight_keys)
    print(f"Total host parameters (>25 only): {total_main}")
    main_flat = torch.empty(total_main, device=device)
    meta = []
    start_idx = 0
    for k in weight_keys:
        tensor = model_weights[k].to(device)
        n = tensor.numel()
        main_flat[start_idx:start_idx+n] = tensor.view(-1)
        meta.append((k, start_idx, n, tensor.shape))
        start_idx += n
    # Flatten the full encoder using sorted keys for consistency.
    enc_keys = sorted(encoder.keys())
    encoder_flat = torch.cat([encoder[k].to(device).view(-1) for k in enc_keys])
    loc = encoder_flat.numel()
    print(f"Total encoder parameters: {loc}")
    if loc > total_main:
        raise ValueError("Not enough host capacity to embed the entire encoder!")
    positions = compute_unique_positions_hmac(loc, total_main, seed)
    for i in range(loc):
        pos = positions[i]
        main_flat[pos] = encoder_flat[i]
    new_model = {}
    for (k, start, n, shape) in meta:
        new_model[k] = main_flat[start:start+n].view(shape)
    for k in model_weights:
        if k not in new_model:
            new_model[k] = model_weights[k]
    return new_model

def extract_encoder_from_attacked_model(model_state, encoder_template, seed, device=devices):
    """
    Extract the encoder parameters from a host model using HMAC positions.
    Args:
        model_state (dict): Host model state_dict.
        encoder_template (dict): Template encoder with the same structure as the original.
        seed (str): HMAC seed used during embedding.
        device (torch.device): Target device.

    Returns:
        dict: Extracted encoder state_dict.
    """
    weight_keys = sorted([k for k in model_state if "weight" in k and model_state[k].numel() > 25])
    total_main = sum(model_state[k].numel() for k in weight_keys)
    main_flat = torch.empty(total_main, device=device)
    start_idx = 0
    for k in weight_keys:
        tensor = model_state[k].to(device)
        n = tensor.numel()
        main_flat[start_idx:start_idx+n] = tensor.view(-1)
        start_idx += n
    # Flatten encoder template using sorted keys.
    enc_keys = sorted(encoder_template.keys())
    enc_shapes = []
    total_enc = 0
    for k in enc_keys:
        tensor = encoder_template[k].to(device)
        n = tensor.numel()
        enc_shapes.append((k, n, tensor.shape))
        total_enc += n
    temp_loc = total_enc
    print(f"Total encoder parameters to extract: {temp_loc}")
    positions = compute_unique_positions_hmac(temp_loc, total_main, seed)
    extracted_flat = torch.empty(temp_loc, device=device)
    for i in range(temp_loc):
        pos = positions[i]
        extracted_flat[i] = main_flat[pos]
    extracted_encoder = {}
    start_idx = 0
    for (k, n, shape) in enc_shapes:
        extracted_encoder[k] = extracted_flat[start_idx:start_idx+n].view(shape).cpu()
        start_idx += n
    return extracted_encoder

def combined_attacker_autoencoder(extracted_encoder, decoder, device=devices):
    """
    Combine an extracted encoder with a provided decoder to reconstruct
    an autoencoder (AttackerAE).

    Args:
        extracted_encoder (dict): State_dict of extracted encoder.
        decoder (dict): State_dict of decoder.
        device (torch.device): Target device.

    Returns:
        nn.Module: Complete attacker autoencoder.
    """
    autoencoder = AttackerAE().to(device)
    autoencoder.encoder.load_state_dict(extracted_encoder, strict=True)
    autoencoder.decoder.load_state_dict(decoder, strict=True)
    return autoencoder
