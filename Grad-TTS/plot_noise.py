import torch
import matplotlib.pyplot as plt
from model.diffusion import get_noise
beta_min = 0.05
beta_max = 20
n_timesteps = 50
h = 1/n_timesteps

t = torch.tensor([1.0 - (i + 0.5) * h for i in range(n_timesteps)])

for i in range(3):

    y = torch.load("./out/noise_est_{}_50.pt".format(i))
    cum_noise = get_noise(t, beta_min, beta_max, cumulative=True)
    y *= torch.sqrt(1.0 - torch.exp(-cum_noise))
    # x = torch.arange(1, len(y)+1)
    plt.plot(t, y, label="sample {}".format(i))

plt.legend()
plt.savefig("./out/noise_est_{}_training.png".format(n_timesteps))

# for i in range(3):
#     sampled = torch.load("./out/noise_{}_50.pt".format(i))
#     cum_noise = get_noise(t, beta_min, beta_max, cumulative=True)
#     lambda_t = -sampled/torch.sqrt(1.0 - torch.exp(-cum_noise))
#     plt.plot(t, lambda_t, label="sample {}".format(i))

# plt.legend()
# plt.savefig("./out/noise_{}.png".format(n_timesteps))