import torch


def run_test_predictions(model, test_dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def print_classification_report(all_labels, all_preds, class_names):
    from sklearn.metrics import classification_report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("=" * 50)
    print("Classification Report")
    print("=" * 50)
    print(report)
    return report
