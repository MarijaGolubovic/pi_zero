from pi_zero_pytorch import π0
import torch

from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor

import torch

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
model.load_state_dict(torch.load('trained_model.pth', weights_only=True))
model.eval()

images = torch.randn(1, 3, 2, 256, 256)

commands = torch.randint(0, 20_000, (1, 1024))
joint_state = torch.randn(1, 12)

loss= model(images, commands, joint_state, trajectory_length = 2)
print(loss)


# print(model)
