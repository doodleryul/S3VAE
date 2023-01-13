from torch import nn
import torch.distributions as dist
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, image_shape=1):
        super(Encoder, self).__init__()
        self.image_shape = image_shape
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=self.image_shape, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        shape, feature_shape = x.shape[:-3], x.shape[-3:]
        x = x.reshape(-1, *feature_shape)
        encoded = self.net(x)
        encoded = encoded.reshape(*shape, -1)
        return F.elu(encoded)


class LSTMEncoder(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, z_size=64):
        super(LSTMEncoder, self).__init__()
        self.z_size = z_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear_t = nn.Linear(hidden_size, self.z_size*2)
        self.linear_f = nn.Linear(hidden_size, self.z_size*2)

    def forward(self, x):
        outputs, hidden = self.lstm(x)
        mu_sigma_t = self.linear_t(outputs)  # B,T,mu+sigma
        mu_sigma_f = self.linear_f(outputs[:,-1,:])  # B,mu+sigma

        mu_t, sigma_t = mu_sigma_t[:,:,:self.z_size], mu_sigma_t[:,:,self.z_size:]  # B,T,mu+sigma
        z_t_distributions = dist.Normal(mu_t, F.softplus(sigma_t))

        mu_f, sigma_f = mu_sigma_f[:,:self.z_size], mu_sigma_f[:,self.z_size:]
        z_f_distributions = dist.Normal(mu_f, F.softplus(sigma_f))
        return z_t_distributions, z_f_distributions


class Decoder(nn.Module):
    def __init__(self, input_size=384,  image_shape=3):
        super().__init__()
        self.image_shape = image_shape

        self.main_net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_size, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=self.image_shape, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, batch):
        shape, feature_shape = batch.shape[:-1], batch.shape[-1]
        batch = batch.reshape(-1, feature_shape, 1, 1)
        decoded = self.main_net(batch)
        feature_shape = decoded.shape[1:]
        decoded = decoded.reshape(*shape, *feature_shape)
        return dist.Normal(loc=decoded, scale=1)


class DynamicFacorPrediction(nn.Module):
    def __init__(self, z_t_size, patch_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=z_t_size, out_features=z_t_size),
            nn.Linear(in_features=z_t_size, out_features=z_t_size),
            nn.Linear(in_features=z_t_size, out_features=patch_size*patch_size)
        )

    def forward(self, x):
        x = self.net(x)
        return x