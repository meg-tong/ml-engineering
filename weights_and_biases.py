# %% 
import time

import torch as t
from einops import rearrange
from fancy_einsum import einsum
from plotly.subplots import make_subplots
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.notebook import tqdm_notebook

import arena_utils
import resnet_replication
import wandb

device = "cuda" if t.cuda.is_available() else "cpu"
# %%
cifar_mean = [0.485, 0.456, 0.406]
cifar_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std)
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

arena_utils.show_cifar_images(trainset, rows=3, cols=5)
# %%
def train() -> None:

    wandb.init()

    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr

    model = resnet_replication.ResNet34().to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    examples_seen = 0
    start_time = time.time()

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(epochs):

        progress_bar = tqdm_notebook(trainloader)

        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

            examples_seen += len(x)
            wandb.log({"train_loss": loss, "elapsed": time.time() - start_time}, step=examples_seen)


        with t.inference_mode():

            accuracy = 0
            total = 0

            for (x, y) in testloader:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {accuracy}/{total}")

    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)

sweep_config = {
    'method': 'bayes',
    'name': 'w0d4_resnet_sweep_10',
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': 
    {
        'batch_size': {'values': [32, 64, 128, 256]},
        'epochs': {'values': [2]},
        'lr': {'max': 0.003, 'min': 0.00005, 'distribution': 'log_uniform_values'}
     }
}

sweep_id = wandb.sweep(sweep=sweep_config, project='w0d4_resnet')

wandb.agent(sweep_id=sweep_id, function=train, count=40)
# %%
