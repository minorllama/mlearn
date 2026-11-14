# basic example of training snn for mnist
#
#$ python ./snn_mnist.py 
## epoch 01  loss 1.608
## epoch 02  loss 0.584
## epoch 03  loss 0.265
## epoch 04  loss 0.130
## epoch 05  loss 0.099
## epoch 06  loss 0.074
## epoch 07  loss 0.054
## epoch 08  loss 0.039
## epoch 09  loss 0.032
## epoch 10  loss 0.034
## epoch 11  loss 0.021
## epoch 12  loss 0.020
## epoch 13  loss 0.015
## epoch 14  loss 0.012
## epoch 15  loss 0.017
# test_accuracy: 96.9%
## test images with zero output spikes: 0.0%

import torch
import snntorch as snn

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from snntorch import surrogate
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# basic SNN classifier
class SpikingClassifier(nn.Module):
    def __init__(self, steps=20):
        super().__init__()
        self.steps = steps
        spike_grad = surrogate.fast_sigmoid(slope=5)

        self.input_layer = nn.Linear(64, 128)
        self.hidden_lif = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.output_layer = nn.Linear(128, 10)
        self.output_lif = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    def forward(self, images):
        # the image supplies the same current at each simulation step.
        hidden_current = self.input_layer(images)

        # start each batch with fresh membrane states.
        batch_size = images.size(0)
        hidden_mem = torch.zeros_like(hidden_current)
        output_mem = images.new_zeros(batch_size, 10)
        spike_counts = torch.zeros_like(output_mem)

        for _ in range(self.steps):
            hidden_spikes, hidden_mem = self.hidden_lif(hidden_current, hidden_mem)

            output_current = self.output_layer(hidden_spikes)
            output_spikes, output_mem = self.output_lif(output_current, output_mem)

            # keep the sum connected to autograd.
            spike_counts = spike_counts + output_spikes

        return spike_counts  # [batch_size, 10]


def main():
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pixels, labels = load_digits(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        pixels, labels, test_size=0.2, random_state=7, stratify=labels
    )

    train_data = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32) / 16.0,
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_images = torch.tensor(x_test, dtype=torch.float32, device=device) / 16.0
    test_labels = torch.tensor(y_test, dtype=torch.long, device=device)

    model = SpikingClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, 16):
        total_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            counts = model(images)
            loss = criterion(counts, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        print(f"## epoch {epoch:02d}  loss {total_loss / len(train_data):.3f}")

    model.eval()
    with torch.inference_mode():
        counts = model(test_images)
        predictions = counts.argmax(dim=1)
        accuracy = (predictions == test_labels).float().mean().item()
        silent = (counts.sum(dim=1) == 0).float().mean().item()

    print(f"# test_accuracy: {accuracy:.1%}")
    print(f"## test images with zero output spikes: {silent:.1%}")


if __name__ == "__main__":
    main()



