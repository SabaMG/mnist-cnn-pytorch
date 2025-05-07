from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from evaluate import evaluate
from config import device
from torchvision.utils import make_grid


def train(model, train_loader, test_loader, optimizer, loss_fn, epochs=30, save_path="best_model.pth"):
    best_accuracy = 0.0
    count = 0
    limit = 12
    dirName = "CNN_Training"

    writer = SummaryWriter(log_dir="runs/" + dirName)
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
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("LearningRate", current_lr, epoch)
            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

        # Evaluate after each epoch
        acc = evaluate(model, test_loader, verbose=False)
        print(f"Test Accuracy: {acc:.2f}%")
        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("Accuracy/test", acc, epoch)

        # Save model if accuracy improves
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved new best model with accuracy: {acc:.2f}%")
            count = 0
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
        if epoch == 0:  # pour ne le faire qu’une fois
            image_grid = make_grid(inputs[:16].cpu(), nrow=4, normalize=True)
            writer.add_image("Sample inputs", image_grid, epoch)
        count += 1
        if count > limit:
            print("Early stopping")
            break
    print("tansorboard logs saved in: ", dirName)
    writer.close()