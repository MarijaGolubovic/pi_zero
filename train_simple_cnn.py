import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import ast
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d

def plot_loss(epoch_losses):
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    #plt.show()
    plt.savefig('loss_simple_cnn.jpg')

def load_data(data_path='DATA'):
    data_dir = os.listdir(data_path)

    actions_path = os.path.join(data_path, 'actions')
    observations_path = os.path.join(data_path, 'observations')

    data_dir = [d for d in data_dir if d not in ['observations', 'actions']]

    image_data = []

    action_data = []
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

                image_data.append(image)
                action_data.append(action)

    return image_data, action_data



class ImageActionDataset(Dataset):
    def __init__(self, image, labels, target_size=(96, 96)):
        self.image = image
        self.labels = labels
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = ast.literal_eval(self.labels[idx])

        label = torch.tensor(label, dtype=torch.float32)

        image = self.transform(self.image[idx])

        return image, label



class DoubleEfficientNet(nn.Module):
    
    def __init__(self, output_dim=6):
        super().__init__()
        self.depth_conv = Conv2d(1, 3, 1) 
        effnet_weights = EfficientNet_B0_Weights.DEFAULT
        self.rgb_backbone = efficientnet_b0(weights=effnet_weights)
        self.depth_backbone = efficientnet_b0() 
        self.rgb_backbone.classifier = nn.Identity()
        self.depth_backbone.classifier = nn.Identity()
        self.fc1 = nn.Linear(1280, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, rgb):
        rgb_features = self.rgb_backbone(rgb)
        
        x = torch.flatten(rgb_features, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, dataloader, epochs=10, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    loss_arr = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")
        loss_arr.append(total_loss)
        
    plot_loss(loss_arr)

if __name__ == "__main__":
    image_data, action_data = load_data()
    print(len(action_data))

    dataset = ImageActionDataset(image=image_data, labels=action_data, target_size=(96, 96))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = DoubleEfficientNet()

    train(model, dataloader, epochs=100, lr=0.001, device='cuda')

    torch.save(model.state_dict(), "simple_cnn.pth")
    print("Model is saved as 'simple_cnn.pth'")
