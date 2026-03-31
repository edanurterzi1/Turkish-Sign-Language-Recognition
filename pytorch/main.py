import os
import torch
from sign_language import config, data, model, training, evaluate, visualize
from sign_language import inference


def main():
    print("=" * 50)
    print("SIGN LANGUAGE MODEL - FULL PIPELINE (TRAIN AND TEST)")
    print("=" * 50)

    plots_dir = "pytorch/plots"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"[1] Checked/created directory: '{plots_dir}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2] Hardware device in use: {device}")

    print("[3] Loading datasets...")
    train_data, val_data, test_data, full_train_data = data.load_datasets()
    train_dl, val_dl, test_dl = data.make_dataloaders(train_data, val_data, test_data)

    class_names = full_train_data.classes
    print(f"    Classes: {class_names}")
    print(f"    Train: {len(train_data)} | Validation: {len(val_data)} | Test: {len(test_data)}")

    print("[4] Initializing model...")
    n_classes = len(class_names)
    cnn_model = model.SignLanguageCNN(num_classes=n_classes).to(device)

    print("\n" + "=" * 50)
    print("Starting Training Process...")

    history = training.train_model(cnn_model, train_dl, val_dl, device)

    training_plot_path = os.path.join(plots_dir, "training_curves.png")

    visualize.plot_training_curves(
        train_accuracies=history["train_accuracies"],
        val_accuracies=history["val_accuracies"],
        train_losses=history["train_losses"],
        val_losses=history["val_losses"],
        save_path=training_plot_path
    )

    print(f"\n[+] Training curves saved to: {training_plot_path}")

    print("\n" + "=" * 50)
    print("Starting Testing Process...")

    best_model = inference.load_trained_model(device=device)

    all_preds, all_labels = evaluate.run_test_predictions(best_model, test_dl, device)
    
    report = evaluate.print_classification_report(all_labels, all_preds, class_names)
    report_path = os.path.join(plots_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"[+] Classification report saved to: {report_path}")


    cm_plot_path = os.path.join(plots_dir, "confusion_matrix.png")

    visualize.plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        save_path=cm_plot_path
    )

    print(f"[+] Confusion Matrix saved to: {cm_plot_path}")

    sample_plot_path = os.path.join(plots_dir, "sample_predictions.png")

    visualize.plot_test_samples(
        best_model,
        test_data,
        class_names,
        device,
        n_samples=12,
        save_path=sample_plot_path
    )

    print(f"[+] Sample predictions saved to: {sample_plot_path}")

    print("\nPROCESS COMPLETED SUCCESSFULLY!")
    print("All plots can be found in the 'plots' directory.")


if __name__ == "__main__":
    main()