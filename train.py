import torch
import torch.nn as nn
from evaluate import evaluate
from config import device

def train(model, train_loader, test_loader, optimizer, loss_fn, epochs=30, save_path="best_model.pth"):
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

        # Evaluate after each epoch
        acc = evaluate(model, test_loader, verbose=False)
        print(f"Test Accuracy: {acc:.2f}%")

        # Save model if accuracy improves
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved new best model with accuracy: {acc:.2f}%")