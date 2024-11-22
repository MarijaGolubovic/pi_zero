from torch.utils.data import DataLoader
from pi_zero_pytorch import π0
import torch

from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor

import torch
import matplotlib.pyplot as plt
from load_dataset import RobotDataset


def plot_loss(epoch_losses):
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()
    plt.savefig('loss.jpg')


def train_model(model, train_data, num_epochs, optimizer, save_path):
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        step = 0
        for batch in train_data:
            images, joint_state, actions = batch

            optimizer.zero_grad()

            commands = torch.full((2, 1024), fill_value=1000, dtype=torch.long)  #torch.randint(0, 20_000, (2, 1024))

            loss, n = model(images, commands, joint_state, actions)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            step += 1

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_data)}")
        avg_loss = total_loss / len(train_data)
        epoch_losses.append(avg_loss)

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        plot_loss(epoch_losses)


if __name__ == "__main__":

    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    v = Extractor(v, return_embeddings_only=True)

    model = π0(
        dim=512,
        vit=v,
        vit_dim=1024,
        dim_action_input=6,
        dim_joint_state=6,
        num_tokens=20_000
    )

    dataset = RobotDataset(data_path='DATA', num_samples=100, batch_size=2)
    train_data = DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    save_path = 'trained_model.pth'

    train_model(model, train_data, num_epochs=1000, optimizer=optimizer, save_path=save_path)
