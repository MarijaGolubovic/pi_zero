import torch
from pi_zero_pytorch import π0
import cv2
import numpy as np


def image2tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor_image = torch.tensor(image).permute(2, 0, 1).float()
    image = tensor_image.unsqueeze(0)
    return image


def text2tokens(text, vocab):
    tokens = [vocab[token] for token in text.split()]
    tokens_tensor = torch.tensor(tokens)
    max_length = 1024
    padded_tensor = torch.zeros(max_length, dtype=torch.long)
    padded_tensor[:len(tokens_tensor)] = tokens_tensor
    commands = padded_tensor.unsqueeze(0)

    return commands


def arr2tensor(array):
    return torch.tensor(array, dtype=torch.float32).unsqueeze(0)


vocab = {"pick": 10, "object": 25, "<pad>": 0}
text = "pick object"
image = cv2.imread("tests/0.jpg")
state = np.array([0.011070077462348504, -0.042870880986626066, 0.1359740293765386, -
                 0.0021951750158317007, 0.018851750803338372, -0.13286090543858572])
action = np.array([0.011232444446087786, -0.035728549544663604, 0.11624150512108208, -
                  0.00034638403547192627, 0.00503887241426521, -0.040307197549534206])


def test_pi_zero_with_vit():
    from vit_pytorch import ViT
    from vit_pytorch.extractor import Extractor

    v = ViT(
        image_size=96,
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
        dim=96,
        vit=v,
        vit_dim=1024,
        dim_action_input=6,
        dim_joint_state=6,
        num_tokens=20_000
    )


    images = image2tensor(image)
    commands = text2tokens(text, vocab)
    joint_state = arr2tensor(state)
    actions = arr2tensor(action).unsqueeze(0)

    loss, _ = model(images, commands, joint_state, actions)
    loss.backward()


    sampled_actions = model(images, commands, joint_state,
                            trajectory_length=1)
    print(sampled_actions)

    assert sampled_actions.shape == (1, 1, 6)


if __name__ == "__main__":
    test_pi_zero_with_vit()
