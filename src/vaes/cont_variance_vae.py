"""
Project: Variance Loss VAE
Description: Variance continuous (Gaussian latent space) VAE, continous output fmnist

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
from torch.distributions import Normal
from torch import random

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
n_name = "VC_VAE_{}e_{}.pth".format(args.epochs, dset)
plots_pth = '../../plots/VC_VAE/' + dset + '/'


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
        self.samples = 5
        self.img_size = 28

        self.fc1 = nn.Linear(self.img_size*self.img_size, 400)
        self.mu1 = nn.Linear(400, self.latent_dim)
        self.logvar1 = nn.Linear(400, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, 400)
        # self.fc4 = nn.Linear(400, self.img_size*self.img_size)
        self.mu2 = nn.Linear(400, self.img_size*self.img_size)
        self.logvar2 = nn.Linear(400, self.img_size*self.img_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.mu1(h1), self.logvar1(h1) # mu and variance

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.mu2(h3))#, self.logvar2(h3)

    def normal_sample(self, mu, std, num_samples=1):
        return torch.normal(mu, std)

    def forward(self, x):
        # run through encoder to get distribution params
        mu1, logvar1 = self.encode(x.view(-1, self.img_size*self.img_size)) # shape B x D
        # mu1 = mu1.view(-1, 1, self.img_size, self.img_size)
        # logvar1 = logvar1.view(-1, 1, self.img_size, self.img_size)

        std1 = torch.exp(0.5*logvar1)
        posterior = Normal(mu1, std1) # q(z|x)
        prior = Normal(torch.zeros_like(mu1), torch.ones_like(std1)) # N(0, 1)

        # Samples
        z = posterior.sample((self.samples,)) # ~ q(z|x)
        # z = self.normal_sample(mu1, std1, self.samples) # B x lD
        # print("Z", z.shape)
        z = z.detach()

        log_posterior = posterior.log_prob(z) # log q(z|x)  S x B x lD
        # print("qzx", log_posterior.shape)
        log_prior = prior.log_prob(z) # log p(z) ~ N(0, 1),  S x B x lD
        # print("pz", log_prior.shape)

        mu2 = self.decode(z) 

        # print(mu2.shape)

        # print("mu2", mu2.shape)
        # print("logvar2", logvar2.shape)
        # std2 = torch.exp(0.5*logvar2)
        likelihood = Normal(mu2, torch.ones_like(mu2)) # log p(x, z) ~ log N(mu1, std2), S x B x eD

        log_likelihood = likelihood.log_prob(x.view(-1, self.img_size*self.img_size)) # shape S x B x eD
        # log_likelihood = F.mse_loss() # shape S x B x eD
        # print("likelihood", log_likelihood.shape)

        elbos = torch.sum(log_likelihood, axis=(1, 2)) - torch.sum(log_posterior, axis=(1, 2)) + torch.sum(log_prior, axis=(1, 2))
        
        return mu2[0], -torch.mean(elbos), torch.var(elbos, axis=0)


model = VAE().to(device)
# model.load_state_dict(torch.load(n_save_pth + n_name))
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, log_prob, entropy):
def loss_function(loss):
    return loss


def train(epoch):
    model.train()
    train_loss = 0

    losses = torch.zeros((len(train_loader), 1))

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, elbo, loss = model(data)
        loss = loss_function(loss)
        loss.backward()
        losses[batch_idx] = elbo
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tELBO: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data), elbo/len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    return losses


def test(epoch):
    model.eval()
    test_loss = 0
    test_elbo = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, elbo, loss  = model(data)
            test_loss += loss_function(loss).item()
            test_elbo += elbo
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), plots_pth + 'reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return elbo / len(test_loader.dataset)


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


# def train(epoch):
#     model.train()
#     train_loss = 0
#     for batch_idx, (data, _) in enumerate(train_loader):
#         data = data.to(device)
#         optimizer.zero_grad()
#         recon_batch, log_prob, entropy = model(data)
#         loss = loss_function(recon_batch, data, log_prob, entropy)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss.item() / len(data)))

#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#           epoch, train_loss / len(train_loader.dataset)))


# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, _) in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(), plots_pth + 'reconstruction_' + str(epoch) + '.png', nrow=n)

#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))


# if __name__ == "__main__":
#     latent_dim=model.latent_dim
#     for epoch in range(1, args.epochs + 1):
#         train(epoch)
#         test(epoch)
#         with torch.no_grad():
#             sample = torch.randn(64, latent_dim).to(device)
#             sample = model.decode(sample).cpu()
#             save_image(sample.view(64, 1, 28, 28),
#                        plots_pth + 'sample_' + str(epoch) + '.png')


#     torch.save(model.state_dict(), n_save_pth + n_name)


