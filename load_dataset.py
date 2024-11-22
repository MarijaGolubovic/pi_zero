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

    return padded_tensor


def str2array(string):
    array = []
    for element in string.replace('[', '').replace(']', '').split(','):
        array.append(float(element))

    return np.array(array)


path = 'DATA'
data_dir = os.listdir(path)

actions_path = path + '/actions'
observations_path = path + '/observations'

data_dir.remove('observations')
data_dir.remove('actions')


def load_data():
    for data in data_dir:
        images_path = os.path.join(path, data)
        num_instances = len(os.listdir(images_path))

        act = os.path.join(actions_path, data) + '.txt'
        curr_pose = act = os.path.join(observations_path, data) + '.txt'
        actions = open(act)
        curresnt_poses = open(curr_pose)

        for idx in range(num_instances):
            img_path = os.path.join(images_path, str(idx))
            img_path += '.jpg'

            image = cv2.imread(img_path)
            action = actions.readline()
            current_pose = curresnt_poses.readline()

            image = image2tensor(image)
            action = arr2tensor(str2array(action))
            current_pose = arr2tensor(str2array(current_pose))
            print(image.shape, action.shape, current_pose.shape)

            return image, current_pose, action