"""
Toy training script #1 for Pulse.

This is intentionally simple: a small CNN on randomly generated
"fake" image data. Its only job is to follow the Pulse contract so
we can test the profiler/diagnosis engine against something we fully
understand before pointing Pulse at a real dataset.

Contract requirements this script follows:
  - Tunable settings (batch_size, num_workers, amp) come in as CLI flags.
  - Per-step timing gets printed as one JSON line per step, so Pulse
    can read stdout and parse it without guessing.
"""

import argparse
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class FakeImageDataset(Dataset):
    """Randomly generated image-like data, standing in for a real dataset."""

    def __init__(self, num_samples=2000, image_size=64, artificial_delay=0.0):
        self.num_samples = num_samples
        self.image_size = image_size
        # artificial_delay lets us simulate a "slow data loading" bottleneck
        # on purpose later, for the Milestone 1 test suite.
        self.artificial_delay = artificial_delay

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.artificial_delay > 0:
            time.sleep(self.artificial_delay)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", type=lambda v: str(v).lower() == "true", default=False)
    parser.add_argument("--steps", type=int, default=100,
                         help="Number of training steps to run (used for short pilot runs)")
    parser.add_argument("--artificial-delay", type=float, default=0.0,
                         help="Seconds to sleep per data item, to simulate a slow data pipeline")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FakeImageDataset(artificial_delay=args.artificial_delay)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = TinyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    step = 0
    data_iter = iter(loader)

    while step < args.steps:
        t0 = time.perf_counter()
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, labels = next(data_iter)
        t1 = time.perf_counter()

        images, labels = images.to(device), labels.to(device)
        t2 = time.perf_counter()

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=args.amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        t3 = time.perf_counter()

        scaler.scale(loss).backward()
        t4 = time.perf_counter()

        scaler.step(optimizer)
        scaler.update()
        t5 = time.perf_counter()

        # One JSON line per step -- this is what Pulse's profiler reads.
        print(json.dumps({
            "step": step,
            "data_loading_ms": (t1 - t0) * 1000,
            "transfer_ms": (t2 - t1) * 1000,
            "forward_ms": (t3 - t2) * 1000,
            "backward_ms": (t4 - t3) * 1000,
            "optimizer_ms": (t5 - t4) * 1000,
            "loss": loss.item(),
        }), flush=True)

        step += 1


if __name__ == "__main__":
    main()
