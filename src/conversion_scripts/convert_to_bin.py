import os
import numpy as np
import torch

def save_to_bin(path: str):
    state_dict = torch.load(path)

    for key, value in state_dict.items():
        print(f"Layer: {key}")
        print(f"Shape: {tuple(value.shape)}")

        param = value.cpu().numpy()
        if param.ndim == 1:
            param = param.reshape((-1, 1))

        param.tofile(os.path.join("src/weights/processed", f"{key}-{(tuple(param.shape))}.bin"))

if __name__ == "__main__":
    os.makedirs("src/weights/raw", exist_ok=True)
    save_to_bin(os.path.join("src/weights/raw", "mlp_model.pth"))
