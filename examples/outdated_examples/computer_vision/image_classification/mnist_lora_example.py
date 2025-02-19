import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

from fedot_ind.core.models.nn.network_modules.layers.lora import linear_layer_parameterization

# Make torch deterministic
_ = torch.manual_seed(228)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST train and test dataset
mnist_trainset = datasets.MNIST(
    root="./examples/data",
    train=True,
    download=True,
    transform=transform,
)

mnist_testset = datasets.MNIST(
    root="./examples/data",
    train=False,
    download=True,
    transform=transform,
)

# Create a dataloaders for the training and testing
train_loader = torch.utils.data.DataLoader(
    mnist_trainset,
    batch_size=10,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    mnist_testset,
    batch_size=10,
    shuffle=True,
)

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DummyOverComplicatedNeuralNetwork(nn.Module):
    """
    DummyOverComplicatedNeuralNetwork an overly expensive NN to classify MNIST digits
    I hate Python, so I don't care about efficiency
    """

    def __init__(self, hidden_size_1=1000, hidden_size_2=2000):
        super(DummyOverComplicatedNeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(28 * 28, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28 * 28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


docnn_model = DummyOverComplicatedNeuralNetwork().to(device)


def train(train_loader, net, epochs=5, total_iterations_limit=None):
    cross_el = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    total_iterations = 0

    for epoch in range(epochs):
        net.train()

        loss_sum = 0
        num_iterations = 0

        data_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        if total_iterations_limit is not None:
            data_iterator.total = total_iterations_limit
        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = net(x.view(-1, 28 * 28))
            loss = cross_el(output, y)
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            data_iterator.set_postfix(loss=avg_loss)
            loss.backward()
            optimizer.step()

            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return


train(train_loader, docnn_model, epochs=1)

# Let's keep clone original_weights so later on we have an opportunity to prove
# that fine-tuning with LoRA doesn't  impact the original weights
original_weights = {}

for name, param in docnn_model.named_parameters():
    original_weights[name] = param.clone().detach()


def test():
    correct = 0
    total = 0

    wrong_counts = [0 for i in range(10)]

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = docnn_model(x.view(-1, 784))

            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                else:
                    wrong_counts[y[idx]] += 1
                total += 1

    print(f"Accuracy: {round(correct / total, 3)}")

    for i in range(len(wrong_counts)):
        print(f"Incorrect counts for {i}: {wrong_counts[i]}")


test()

# Let's visualize how many parameters are in the original network, before introducing the LoRA matrices.
# Size of the weights matrices of the network and save total number of
# parameters
total_parameters_original = 0

for index, layer in enumerate(
        [docnn_model.linear1, docnn_model.linear2, docnn_model.linear3]):
    total_parameters_original += layer.weight.nelement() + layer.bias.nelement()

    print(f"Layer {index + 1}: W: {layer.weight.shape} + B: {layer.bias.shape}")

print(f"Total number of parameters: {total_parameters_original:,}")

# Define the LoRA parameterization.
parametrize.register_parametrization(
    docnn_model.linear1,
    "weight",
    linear_layer_parameterization(docnn_model.linear1, device),
)

parametrize.register_parametrization(
    docnn_model.linear2,
    "weight",
    linear_layer_parameterization(docnn_model.linear2, device),
)

parametrize.register_parametrization(
    docnn_model.linear3,
    "weight",
    linear_layer_parameterization(docnn_model.linear3, device),
)


def enable_disable_lora(enabled=True):
    for layer in [
            docnn_model.linear1,
            docnn_model.linear2,
            docnn_model.linear3]:
        layer.parametrizations["weight"][0].enabled = enabled


total_parameters_lora = 0
total_parameters_non_lora = 0
for index, layer in enumerate(
        [docnn_model.linear1, docnn_model.linear2, docnn_model.linear3]):
    total_parameters_lora += layer.parametrizations["weight"][0].lora_A.nelement(
    ) + layer.parametrizations["weight"][0].lora_B.nelement()
    total_parameters_non_lora += layer.weight.nelement() + layer.bias.nelement()

    print(
        f"Layer {index + 1}: W: {layer.weight.shape} + B: {layer.bias.shape} + Lora_A: {layer.parametrizations['weight'][0].lora_A.shape} + Lora_B: {layer.parametrizations['weight'][0].lora_B.shape}")

# The non-LoRA parameters count must match the original network
assert total_parameters_non_lora == total_parameters_original
print(f"Params (original): {total_parameters_non_lora:,}")
print(
    f"Params (original + LoRA): {total_parameters_lora + total_parameters_non_lora:,}")
print(f"Params introduced by LoRA: {total_parameters_lora:,}")

parameters_growth = (total_parameters_lora / total_parameters_non_lora) * 100
print(f"Parameters growth: {parameters_growth:.3f}%")

# Freeze all the parameters of the original network and only fine-tuning the ones introduced by LoRA.
# Then fine-tune the model on the digit 9 and only for 100 batches.
# Freeze the non-Lora parameters
for name, param in docnn_model.named_parameters():
    if "lora" not in name:
        print(f"Freezing non-LoRA parameter {name}")
        param.requires_grad = False

# Load the MNIST dataset again, by keeping only the digit 9
mnist_trainset = datasets.MNIST(
    root="./examples/data",
    train=True,
    download=True,
    transform=transform,
)

exclude_indices = mnist_trainset.targets == 9
mnist_trainset.data = mnist_trainset.data[exclude_indices]
mnist_trainset.targets = mnist_trainset.targets[exclude_indices]
# Create a dataloader for the training
train_loader = torch.utils.data.DataLoader(
    mnist_trainset,
    batch_size=10,
    shuffle=True,
)

# Train the network with LoRA only on the digit 9 and only for 100 batches
# (hoping that it would improve the performance on the digit 9)
train(train_loader, docnn_model, epochs=1, total_iterations_limit=100)

# Verify that the fine-tuning didn't alter the original weights, but only the ones introduced by LoRA.
# Check that the frozen parameters are still unchanged by the fine-tuning
assert torch.all(docnn_model.linear1.parametrizations.weight.original ==
                 original_weights["linear1.weight"])
assert torch.all(docnn_model.linear2.parametrizations.weight.original ==
                 original_weights["linear2.weight"])
assert torch.all(docnn_model.linear3.parametrizations.weight.original ==
                 original_weights["linear3.weight"])

enable_disable_lora(enabled=True)
# The new linear1.weight is obtained by the "forward" function of our LoRA parametrization
# The original weights have been moved to net.linear1.parametrizations.weight.original
# More info here:
# https://pytorch.org/tutorials/intermediate/parametrizations.html#inspecting-a-parametrized-module
assert torch.equal(
    docnn_model.linear1.weight, docnn_model.linear1.parametrizations.weight.original +
    (docnn_model.linear1.parametrizations.weight[0].lora_B @ docnn_model.linear1.parametrizations.weight[0].lora_A) *
    docnn_model.linear1.parametrizations.weight[0].scale)

enable_disable_lora(enabled=False)
# If we disable LoRA, the linear1.weight is the original one
assert torch.equal(
    docnn_model.linear1.weight,
    original_weights["linear1.weight"])


# Test the network with LoRA enabled (the digit 9 should be classified better)
enable_disable_lora(enabled=True)
test()

# Test the network with LoRA disabled (the accuracy and errors counts must
# be the same as the original network)
enable_disable_lora(enabled=False)
test()
