import torch
from pathlib import Path

ckpt_path = Path("pretrained_weights/supervised_suprem_segresnet_2100.pth")

ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt.get("net", ckpt.get("state_dict", ckpt))

print("Top-level keys:", ckpt.keys() if isinstance(ckpt, dict) else type(ckpt))
print("Number of tensors:", len(state))

for name, tensor in state.items():
    if any(word in name.lower() for word in ["final", "out", "head", "controller", "organ_embedding", "precls"]):
        shape = tuple(tensor.shape) if hasattr(tensor, "shape") else type(tensor)
        print(name, shape)