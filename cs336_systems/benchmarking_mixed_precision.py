from matplotlib.pyplot import step
import torch
import torch.nn as nn

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        print("fc1 output dtype:", x.dtype)
        x = self.relu(x)
        x = self.ln(x)
        print("ln output dtype:", x.dtype)
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    model = ToyModel(20, 5)
    dtype = torch.float16
    x = torch.randn(16, 20, device="cuda", dtype=dtype)
    target = torch.randint(0, 5, (16,), device="cuda")  # 16 samples, 5 classes

    optimizer = AdamW(model.parameters())

    def step_forward_backward():
        optimizer.zero_grad()
        out = model(x)
        print("Model parameter dtype:", next(model.parameters()).dtype)
        print("Final output dtype:", out.dtype)
        loss = cross_entropy(out, target)
        print("Loss dtype:", loss.dtype)
        loss.backward()
        optimizer.step()

    with torch.autocast(device_type="cuda",dtype=dtype):
        step_forward_backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name:10s} param dtype={param.dtype}, grad dtype={param.grad.dtype}")