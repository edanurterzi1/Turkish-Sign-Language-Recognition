import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from . import config


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    device,
    epochs=None,
    patience=None,
    checkpoint_path=None,
    criterion=None,
    optimizer=None,
):
    if epochs is None:
        epochs = config.EPOCHS
    if patience is None:
        patience = config.PATIENCE
    if checkpoint_path is None:
        checkpoint_path = config.CHECKPOINT_PATH
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("=== Training Started ===\n")

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_loss = float("inf")
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total = 0
        correct = 0

        for images, labels in tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{epochs} [Train]",
            leave=False,
        ):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = total_loss / len(train_dataloader)
        train_accuracy = correct / total

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()

        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(
                val_dataloader,
                desc=f"Epoch {epoch + 1} [Val]",
                leave=False,
            ):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = correct / total

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0

            os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
            torch.save(model, checkpoint_path)
            print(f"--> New best model saved! Validation Loss: {best_loss:.5f}")
        else:
            counter += 1
            print(f"No improvement {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered!")
            break

        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} || "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
        )

    print(f"{'=' * 20} TRAINING COMPLETED {'=' * 20}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "best_val_loss": best_loss,
    }
