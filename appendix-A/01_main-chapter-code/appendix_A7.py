import torch
import torch.nn.functional as F

from appendix_utils import NeuralNetwork, train_loader, test_loader
from appendix_utils import compute_accuracy
from appendix_utils import X_train, y_train

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

        # LOGGING
        print(
            f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f"  |  Batch {batch_idx:03d}/{len(train_loader):03d}"
            f"  |  Train Loss: {loss:.2f}"
        )

    model.eval()
    # Optional model evaluation

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)

model.eval()
with torch.no_grad():
    outputs = model(X_train)
print(outputs)

torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(probas)

predictions = torch.argmax(outputs, dim=1)
print(predictions)

print(torch.sum(predictions == y_train))

print(compute_accuracy(model, test_loader))
