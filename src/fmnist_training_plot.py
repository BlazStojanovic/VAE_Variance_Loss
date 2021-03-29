import numpy as np
import matplotlib.pyplot as plt


import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'


dsetsize = 60000
batch = 128

R_train_loss = np.load("../plots/RC_VAE/fmnist/train_loss.npy", allow_pickle=True)
R_train_loss = np.reshape(R_train_loss, (-1, 1))
R_test_loss = np.load("../plots/RC_VAE/fmnist/test_loss.npy", allow_pickle=True)

V_train_loss = np.load("../plots/V_VAE/fmnist/train_loss.npy", allow_pickle=True)
V_train_loss = np.reshape(V_train_loss, (-1, 1))
V_test_loss = np.load("../plots/V_VAE/fmnist/test_loss.npy", allow_pickle=True)

VC_train_loss = np.load("../plots/VC_VAE/fmnist/train_loss.npy", allow_pickle=True)
VC_train_loss = np.reshape(VC_train_loss, (-1, 1))
VC_test_loss = np.load("../plots/VC_VAE/fmnist/test_loss.npy", allow_pickle=True)

fig = plt.figure(figsize=(12, 7))
plt.plot(np.arange(len(R_train_loss))*batch, R_train_loss/128, 'b-', label="REINFORCE - train set", alpha=0.3)
plt.plot((np.arange(len(R_test_loss)) + 1)*dsetsize, R_test_loss, 'bo-', linewidth=4, label="REINFORCE - test set")

plt.plot(np.arange(len(V_train_loss))*batch, V_train_loss/128, 'r-', label="REPARAM - train set", alpha=0.3)
plt.plot((np.arange(len(V_test_loss)) + 1)*dsetsize, V_test_loss, 'ro-', linewidth=4, label="REPARAM - test set")

plt.plot(np.arange(len(VC_train_loss))*batch, VC_train_loss/128, 'g-', label="REPARAM - train set", alpha=0.3)
plt.plot((np.arange(len(VC_test_loss)) + 1)*dsetsize, VC_test_loss, 'go-', linewidth=4, label="REPARAM - test set")


plt.ylabel("-Elbo")
plt.xlabel("Training samples evaluated")

plt.ylim(200, 800)
plt.xlim()

plt.legend()
plt.savefig("../plots/fmnist_learning.png", bbox_inches='tight')
plt.show()
