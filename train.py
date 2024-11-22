from torch.utils.data import Dataset, DataLoader
from pi_zero_pytorch import π0
import torch

from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor

import torch


def generate_dummy_data():
    images = torch.randn(3, 2, 256, 256)
    joint_state = torch.randn(12)
    actions = torch.randn(32, 6)

    return images, joint_state, actions


class RobotDataset(Dataset):
    def __init__(self, num_samples=10, batch_size=2):
        self.num_samples = num_samples
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        images, joint_state, actions = generate_dummy_data()
        return images, joint_state, actions


def train_model(model, train_data, num_epochs, optimizer, save_path):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        step = 0
        for batch in train_data:
            images, joint_state, actions = batch

            optimizer.zero_grad()

            commands = torch.randint(0, 20_000, (1, 1024))

            loss, n = model(images, commands, joint_state, actions)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            print(f'Step: {step}')
            step += 1

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_data)}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


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
        dim_joint_state=12,
        num_tokens=20_000
    )

    dataset = RobotDataset()
    train_data = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    save_path = 'trained_model.pth'

    train_model(model, train_data, optimizer=optimizer,
                save_path=save_path, num_epochs=2)
