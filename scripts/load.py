import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NanoGPT import GPT, GPTConfig

checkpoint_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "checkpoints",
    "parameter_checkpoint_19072.pt",
)

with torch.serialization.safe_globals([(GPTConfig, "__main__.GPTConfig")]):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model_state_dict = checkpoint["model"]
config = checkpoint["config"]

model = GPT(config)


model = torch.compile(model)

model.load_state_dict(model_state_dict)

print(f"Model loaded at step {checkpoint['step']}")
print(f"Validation loss: {checkpoint['val_loss']}")

model.eval()
