from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import os


def image2tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor_image = torch.tensor(image).permute(2, 0, 1).float()
    return tensor_image


def arr2tensor(array):
    return torch.tensor(array, dtype=torch.float32)


def text2tokens(text, vocab):
    tokens = [vocab[token] for token in text.split()]
    tokens_tensor = torch.tensor(tokens)
    max_length = 1024
    padded_tensor = torch.zeros(max_length, dtype=torch.long)
    padded_tensor[:len(tokens_tensor)] = tokens_tensor

    return padded_tensor.unsqueeze(0)


def str2array(string):
    array = [float(element) for element in string.replace(
        '[', '').replace(']', '').split(',')]

    return np.array(array)


def load_data(data_path='DATA'):
    data_dir = os.listdir(data_path)

    actions_path = os.path.join(data_path, 'actions')
    observations_path = os.path.join(data_path, 'observations')

    data_dir = [d for d in data_dir if d not in ['observations', 'actions']]

    all_data = []
    for data in data_dir:
        images_path = os.path.join(data_path, data)
        num_instances = len(os.listdir(images_path))

        action_file = os.path.join(actions_path, f'{data}.txt')
        pose_file = os.path.join(observations_path, f'{data}.txt')

        with open(action_file, 'r') as actions, open(pose_file, 'r') as poses:
            for idx in range(num_instances):
                img_path = os.path.join(images_path, f'{idx}.jpg')

                image = cv2.imread(img_path)
                if image is None:
                    print('[ERROR] Image is None!')
                    continue

                action = actions.readline().strip()
                current_pose = poses.readline().strip()

                image_tensor = image2tensor(image)
                action_tensor = arr2tensor(str2array(action))
                current_pose_tensor = arr2tensor(str2array(current_pose))

                all_data.append(
                    (image_tensor, current_pose_tensor, action_tensor))


    return all_data


class RobotDataset(Dataset):
    def __init__(self, data_path='DATA', num_samples=None, batch_size=2):
        self.data = load_data(data_path)
        self.num_samples = len(
            self.data) if num_samples is None else num_samples
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        image_tensor, current_pose_tensor, action_tensor = self.data[idx]

        return image_tensor, current_pose_tensor, action_tensor


if __name__ == '__main__':
    load_data()
