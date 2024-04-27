import torch
import torch.nn.functional as F

from appendix_A5 import NeuralNetwork
from appendix_A6 import train_loader

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):

    model.train()

    for batch_idx, (features, labels) in enumerate(train_loader):

        logits = model(features)

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### LOGGING
        print(
            f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f"  |  Batch {batch_idx:03d}/{len(train_loader):03d}"
            f"  |  Train Loss: {loss:.2f}"
        )

    model.eval()
    # Optional model evaluation
