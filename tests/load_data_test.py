from torch.utils.data import Dataset, DataLoader
from pi_zero_pytorch import π0

from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor

import cv2
import os
import torch
import numpy as np

from load_dataset import load_data

def image2tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor_image = torch.tensor(image).permute(2, 0, 1).float()
    return tensor_image


def text2tokens(text, vocab):
    tokens = [vocab[token] for token in text.split()]
    tokens_tensor = torch.tensor(tokens)
    max_length = 1024
    padded_tensor = torch.zeros(max_length, dtype=torch.long)
    padded_tensor[:len(tokens_tensor)] = tokens_tensor

    return padded_tensor


def arr2tensor(array):
    return torch.tensor(array, dtype=torch.float32)


def str2array(string):
    array = []
    for element in string.replace('[', '').replace(']', '').split(','):
        array.append(float(element))

    return np.array(array)


def train_model(model, train_data, num_epochs, optimizer, save_path):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        step = 0
        for batch in train_data:
            images, joint_state, actions = batch

            optimizer.zero_grad()

            commands = torch.randint(0, 20_000, (2, 1024))

            print(images.shape, joint_state.shape, actions.shape)
            loss, n = model(images, commands, joint_state, actions)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            step += 1

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_data)}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


path = 'DATA'
data_dir = os.listdir(path)

actions_path = path + '/actions'
observations_path = path + '/observations'

data_dir.remove('observations')
data_dir.remove('actions')


class RobotDataset(Dataset):
    def __init__(self, data_path='DATA', num_samples=None, batch_size=2):
        self.data = load_data(data_path) 
        self.num_samples = len(self.data) if num_samples is None else num_samples
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples 

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        image_tensor, current_pose_tensor, action_tensor = self.data[idx]

        return image_tensor, current_pose_tensor, action_tensor


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

    train_model(model, train_data, num_epochs=10, optimizer=optimizer, save_path=save_path)
