from __future__ import print_function

import vaes.vanila_vae as vae
import vaes.cont_reinforce_vae as rcvae
import vaes.cont_variance_vae2 as vcvae
import torch
import torch.utils.data
from torchvision.utils import save_image, make_grid

if __name__ == '__main__':
	# PATH = "../trained_models/Vanilla_VAE_30e_f.pth"
	# PATH = "../trained_models/RC_VAE_20e_fmnist.pth"
	PATH = "../trained_models/VC_VAE_20e_fmnist.pth"


	cuda = False
	device = torch.device("cuda" if cuda else "cpu")

	# model = vae.VAE()
	# model = rcvae.VAE()
	model = vcvae.VAE()
	

	model.load_state_dict(torch.load(PATH))
	z1min = -2.0
	z1max = 2.0
	z2min = -2.0
	z2max = 2.0
	n1z = 20
	n2z = 20

	ps = 28

	z1 = torch.linspace(z1min, z1max, n1z).to(device)
	z2 = torch.linspace(z1min, z1max, n1z).to(device)
	nz = n1z*n2z

	sample = torch.zeros(nz, 2).to(device)

	for i in range(n1z):
		for j in range(n2z):
			idx = i * n2z + j
			sample[idx][0] = z1[i]
			sample[idx][1] = z2[j]

	decodes = model.sample_img(sample).cpu().view(nz, 1, ps, ps)
	grid = make_grid(decodes, nrow=n1z)
	
	# save_image(grid, '../plots/V_VAE/fmnist/manifold.png')
	# save_image(grid, '../plots/RC_VAE/fmnist/manifold.png')
	save_image(grid, '../plots/VC_VAE/fmnist/manifold.png')