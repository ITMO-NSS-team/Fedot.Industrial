
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as v2


def load_mnist_data():
    transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize((0.1307,), (0.3081,))
    ])

    # Load the MNIST dataset
    mnist_trainset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform)

    # Create a dataloader for the training
    train_loader = torch.utils.data.DataLoader(
        mnist_trainset, batch_size=10, shuffle=True)

    # Load the MNIST test set
    mnist_testset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform)
    test_loader = torch.utils.data.DataLoader(
        mnist_testset, batch_size=10, shuffle=True)


for index, layer in enumerate(
        [docnn_model.linear1, docnn_model.linear2, docnn_model.linear3]):
    total_parameters_original += layer.weight.nelement() + layer.bias.nelement()

    print(f"Layer {index + 1}: W: {layer.weight.shape} + B: {layer.bias.shape}")

print(f"Total number of parameters: {total_parameters_original:,}")


docnn_model_copy.linear1.weight = nn.Parameter(updated_weight[0]).to(device)
docnn_model_copy.linear2.weight = nn.Parameter(updated_weight[1]).to(device)
docnn_model_copy.linear3.weight = nn.Parameter(updated_weight[2]).to(device)


# %%
total_parameters_lora = 0
total_parameters_non_lora = 0
for index, layer in enumerate(
        [docnn_model_copy.linear1, docnn_model_copy.linear2, docnn_model_copy.linear3]):
    total_parameters_lora += layer.parametrizations["weight"][0].lora_A.nelement(
    ) + layer.parametrizations["weight"][0].lora_B.nelement()
    total_parameters_non_lora += layer.weight.nelement() + layer.bias.nelement()

    print(
        f"Layer {index + 1}: W: {layer.weight.shape} + B: {layer.bias.shape} + Lora_A: {layer.parametrizations['weight'][0].lora_A.shape} + Lora_B: {layer.parametrizations['weight'][0].lora_B.shape}")

assert total_parameters_non_lora == total_parameters_original
print(f"Params (original): {total_parameters_non_lora:,}")
print(
    f"Params (original + LoRA): {total_parameters_lora + total_parameters_non_lora:,}")
print(f"Params introduced by LoRA: {total_parameters_lora:,}")

parameters_growth = (total_parameters_lora / total_parameters_non_lora) * 100
print(f"Parameters growth: {parameters_growth:.3f}%")


mnist_trainset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform)
# Create a dataloader for the training
exclude_indices = mnist_trainset.targets == 9
mnist_trainset.data = mnist_trainset.data[exclude_indices]
mnist_trainset.targets = mnist_trainset.targets[exclude_indices]
# Create a dataloader for the training
train_loader = torch.utils.data.DataLoader(
    mnist_trainset, batch_size=10, shuffle=True)


enable_disable_lora(enabled=True)
# The new linear1.weight is obtained by the "forward" function of our LoRA parametrization
# The original weights have been moved to net.linear1.parametrizations.weight.original
# More info here:
# https://pytorch.org/tutorials/intermediate/parametrizations.html#inspecting-a-parametrized-module
diff = docnn_model_copy.linear1.weight - (docnn_model_copy.linear1.parametrizations.weight.original + (
    docnn_model_copy.linear1.parametrizations.weight[0].lora_A @
    docnn_model_copy.linear1.parametrizations.weight[0].lora_B) *
    docnn_model_copy.linear1.parametrizations.weight[0].scale)
aprox_error = torch.linalg.norm(diff, 'fro')

enable_disable_lora(enabled=False)
# If we disable LoRA, the linear1.weight is the original one
assert torch.equal(
    docnn_model_copy.linear1.weight,
    original_weights["linear1.weight"])
# Test with LoRA enabled
enable_disable_lora(enabled=True)
run()
# %%
# Test with LoRA enabled
enable_disable_lora(enabled=False)
run()
