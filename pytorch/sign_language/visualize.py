import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from . import config


def plot_training_curves(train_accuracies, val_accuracies, train_losses, val_losses, save_path=None):
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(all_labels, all_preds, class_names, save_path=None):
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(15, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_test_samples(model, test_data, class_names, device, n_samples=12, save_path=None):
    plt.figure(figsize=(15, 10))

    mean = np.array(config.IMAGENET_MEAN)
    std = np.array(config.IMAGENET_STD)

    sample_indices = random.sample(range(len(test_data)), n_samples)

    model.eval()
    for i, idx in enumerate(sample_indices):
        img, label = test_data[idx]

        img_input = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_input)
        _, pred = torch.max(output, 1)

        img_np = img.numpy().transpose((1, 2, 0))
        img_np = (img_np * std) + mean

        plt.subplot(3, 4, i + 1)
        plt.imshow(np.clip(img_np, 0, 1))
        plt.title(f"True: {class_names[label]}\nPred: {class_names[pred.item()]}")
        plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
