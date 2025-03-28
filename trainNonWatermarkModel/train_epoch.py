import torch
from tqdm import tqdm

def train_model(model, dataloader, criterion, optimizer, device, desc="Training"):
    """
    Runs training epoch.

    Returns:
        avg_loss (float): average loss over the epoch.
        accuracy (float): training accuracy (in percent) over the epoch.
    """
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    loop = tqdm(dataloader, desc=desc, leave=True)
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += batch_size

        loop.set_postfix(loss=loss.item(), acc=100.0 * total_correct / total_samples)

    avg_loss = running_loss / total_samples if total_samples > 0 else 0
    accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy


def validate_model(model, dataloader, criterion, device, desc="Validation"):
    """
    Runs validation epoch.

    Returns:
        avg_loss (float): average loss over the validation set.
        accuracy (float): validation accuracy (in percent).
    """
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    loop = tqdm(dataloader, desc=desc, leave=True)
    with torch.no_grad():
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += inputs.size(0)
            loop.set_postfix(loss=loss.item(), acc=100.0 * total_correct / total_samples)

    avg_loss = running_loss / total_samples if total_samples > 0 else 0
    accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy
