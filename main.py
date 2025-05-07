from dataset import get_data_loaders
from model import CNN
from train import train
from evaluate import evaluate
from utils import set_seed
import torch.nn as nn
import torch
from config import device
import numpy as np

if __name__ == "__main__":
    print(f"Using device: {device}")

    accuracies = []
    seeds = [42, 1337, 2025, 1234, 0]
    for seed in seeds:
        print(f"\n=== Training with seed {seed} ===")
        set_seed(seed)

        # Load data and model
        train_loader, test_loader = get_data_loaders()
        model = CNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Define paths
        log_dir = f"cnn_deep_V3.0_seed_{seed}"
        best_model_path = f"best_model_seed_{seed}.pth"

        # Train
        train(model, train_loader, test_loader, optimizer, loss_fn, save_path=best_model_path, log_dir=log_dir)

        # Final evaluation
        model.load_state_dict(torch.load(best_model_path))
        print(f"\n📦 Final evaluation for seed {seed}:")
        acc = evaluate(model, test_loader)
        accuracies.append(acc)
        
        
    # === Résumé global ===
    accuracies = np.array(accuracies)
    mean_acc = accuracies.mean()
    std_acc = accuracies.std()

    summary = (
        "\n🎯 Résumé multi-seed\n"
        f"→ Moyenne des accuracies: {mean_acc:.2f}%\n"
        f"→ Écart-type: {std_acc:.2f}%\n"
    )

    print(summary)

    # Écriture dans un fichier texte
    with open("results_summary.txt", "w") as f:
        f.write("Résultats Multi-Seed - MNIST CNN\n")
        f.write("="*40 + "\n")
        for i, acc in enumerate(accuracies, 1):
            f.write(f"Seed {i}: {acc:.2f}%\n")
        f.write("\n" + summary)