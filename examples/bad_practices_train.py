"""
Toy training script #2 for Pulse -- deliberately written BADLY on purpose.

This is a test case for the static checker: it has known, hardcoded
inefficiencies baked in, so we can check that Pulse's checker correctly
flags them. Don't "fix" this file -- it's supposed to be bad.

Known issues planted here (the "answer key"):
  - DataLoader uses num_workers=0 (hardcoded, not parameterized)
  - DataLoader doesn't set pin_memory at all
  - No mixed precision (AMP) anywhere in the script
  - print() called every single step inside the training loop
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class FakeImageDataset(Dataset):
    def __init__(self, num_samples=500, image_size=64):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, self.image_size, self.image_size)
        label = torch.randint(0, 10, (1,)).item()
        return image, label


class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FakeImageDataset()
    # BAD: num_workers hardcoded to 0, no pin_memory at all
    loader = DataLoader(dataset, batch_size=32, num_workers=0)

    model = TinyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    step = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # BAD: printing every single step
        print(f"step {step}, loss {loss.item()}")
        step += 1
        if step >= 20:
            break


if __name__ == "__main__":
    main()
