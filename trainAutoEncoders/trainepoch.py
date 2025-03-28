from tqdm import tqdm


def train_autos(autoencoder, train_loader, optimizer, criterion, devices, epoch, total_rounds):
    autoencoder.train()
    epoch_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{total_rounds}]", leave=True)
    for images, _ in loop:
        images = images.to(devices)
        optimizer.zero_grad()
        _, decoded = autoencoder(images)
        loss = criterion(decoded, images)
        loss.backward()

        # Debug: print gradient norm from the first encoder layer.
        grad_norm = autoencoder.encoder[0].weight.grad.norm().item()
        loop.set_postfix(loss=loss.item(), grad_norm=grad_norm)

        optimizer.step()
        epoch_loss += loss.item()

    train_loss_epoch = epoch_loss / len(train_loader)
    return train_loss_epoch
