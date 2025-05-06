from dataset import get_data_loaders
from model import CNN
from train import train
from evaluate import evaluate
from utils import set_seed
import torch.nn as nn
import torch
from config import device

if __name__ == "__main__":
    print(f"Using device: {device}")
    set_seed(42)
    train_loader, test_loader = get_data_loaders()
    model = CNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_model_path = "best_model.pth"
    train(model, train_loader, test_loader, optimizer, loss_fn, save_path=best_model_path)

    # Load best model and evaluate
    model.load_state_dict(torch.load(best_model_path))
    print("\n📦 Loaded best saved model for final evaluation:")
    evaluate(model, test_loader)