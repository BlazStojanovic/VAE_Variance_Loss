"""
Project: Variance Loss VAE
Description: Reinforce continous (Gaussian latent space) VAE, continous output fmnist

Not my work, adapted from pytorch examples https://github.com/pytorch/examples

"""

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import OneHotCategorical
from torch.distributions import Normal

# Parser
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

# PATHS
dset = 'fmnist'
datapth = '../../../shared/Data'
n_save_pth = "../../trained_models/"
n_name = "RC_VAE_{}e_{}.pth".format(args.epochs, dset)
plots_pth = '../../plots/RC_VAE/' + dset + '/'

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(datapth, train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(datapth, train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.latent_dim = 2

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, self.latent_dim)
        self.fc22 = nn.Linear(400, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        std = torch.exp(0.5*logvar)
        posterior = Normal(mu, std)
        z = posterior.sample()
        log_prob = posterior.log_prob(z)
        entropy = posterior.entropy()
        return self.decode(z), log_prob, entropy


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def loss_function(recon_x, x, log_prob, entropy):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduce=False).sum(-1)
    log_prob = log_prob.sum(-1)
    reinforce_loss = torch.sum(log_prob*BCE.detach())
    loss = BCE.sum() + reinforce_loss - reinforce_loss.detach() + entropy.sum()
    return loss


def train(epoch):
    model.train()
    train_loss = 0

    losses = torch.zeros((len(train_loader), 1))

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, log_prob, entropy = model(data)
        loss = loss_function(recon_batch, data, log_prob, entropy)
        loss.backward()
        losses[batch_idx] = loss
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    return losses


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), plots_pth + 'reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


if __name__ == "__main__":
    latent_dim=model.latent_dim
    
    train_loss = []
    test_loss = []

    for epoch in range(1, args.epochs + 1):
        trainls = train(epoch)
        train_loss.append(trainls.detach().cpu().numpy())
        tl = test(epoch)
        test_loss.append(tl)
        with torch.no_grad():
            sample = torch.randn(64, latent_dim).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       plots_pth + 'sample_' + str(epoch) + '.png')

    import numpy as np

    np.save(plots_pth+"train_loss.npy", train_loss)
    np.save(plots_pth+"test_loss.npy", test_loss)
    torch.save(model.state_dict(), n_save_pth + n_name)

