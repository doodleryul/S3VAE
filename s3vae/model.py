from torch import nn
import torch.distributions as dist
import torch.nn.functional as F
from s3vae.loss import calculate_vae_loss, calculate_dfp_loss, calculate_mi_loss
from itertools import chain
import torch
import os

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
    def __init__(self, z_t_size, label_num=3):
        # patchsize가 3이면 9가 label_num, 얼굴 이미지의 경우 label은 3, 거리가 3개니까
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=z_t_size, out_features=z_t_size),
            nn.Linear(in_features=z_t_size, out_features=z_t_size),
            nn.Linear(in_features=z_t_size, out_features=label_num)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class S3VAE:
    def __init__(self, config) -> None:
        self.device = config['device']
        self._config = config
        self.build_model()

    def build_model(self):
        self._encoder = Encoder(self._config['image_shape']).to(self.device)
        self._lstm_encoder = LSTMEncoder(input_size=128).to(self.device) # 앞의 feature size
        self._prior_lstm_encoder = LSTMEncoder(self._config['z_size']).to(self.device)
        self._decoder = Decoder(input_size=128, image_shape=self._config['image_shape']).to(self.device) # z_t+z_f size
        self._dfp = DynamicFacorPrediction(z_t_size=self._lstm_encoder.z_size, label_num=self._config['label_num']).to(self.device)
        self._dfp_criterion = nn.CrossEntropyLoss()
        self._triplet_loss = torch.nn.TripletMarginLoss(margin=self._config['margin'])
        self._model = chain(self._encoder.parameters(), 
                            self._lstm_encoder.parameters(),
                            self._prior_lstm_encoder.parameters(),
                            self._decoder.parameters(),
                            self._dfp.parameters()
                                    )
        self._optimizer = torch.optim.Adam(self._model, lr=self._config['learning_rate'])

    def save(self, path):
        torch.save(self._encoder.state_dict(), os.path.join(path, 'encoder.pt'))
        torch.save(self._lstm_encoder.state_dict(), os.path.join(path, 'lstm_encoder.pt'))
        torch.save(self._prior_lstm_encoder.state_dict(), os.path.join(path, 'prior_lstm_encoder.pt'))
        torch.save(self._decoder.state_dict(), os.path.join(path, 'decoder.pt'))
        torch.save(self._dfp.state_dict(), os.path.join(path, 'dfp.pt'))

    def load(self, path):
        self._encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pt'), map_location=self.device))
        self._lstm_encoder.load_state_dict(torch.load(os.path.join(path, 'lstm_encoder.pt'), map_location=self.device))
        self._prior_lstm_encoder.load_state_dict(torch.load(os.path.join(path, 'prior_lstm_encoder.pt'), map_location=self.device))
        self._decoder.load_state_dict(torch.load(os.path.join(path, 'decoder.pt'), map_location=self.device))
        self._dfp.load_state_dict(torch.load(os.path.join(path, 'dfp.pt'), map_location=self.device))
        self._encoder.eval()
        self._lstm_encoder.eval()
        self._prior_lstm_encoder.eval()
        self._decoder.eval()
        self._dfp.eval()

    def compute_loss(self, x, negative_x, labels=None):
        x = x.to(self.device)
        negative_x = negative_x.to(self.device)
        if labels:
            labels = labels.to(self.device)
        
        encoded = self._encoder(x.float())

        # static consistency constraint
        shuffle_idx = torch.randperm(encoded.shape[1]).contiguous()
        shuffled_encoded = encoded[:, shuffle_idx]
        _, z_f_pos_dist = self._lstm_encoder(encoded)

        permuted_encoded = self._encoder(negative_x.float())
        _, z_f_neg_dist = self._lstm_encoder(permuted_encoded)

        # forward
        z_t_dist, z_f_dist = self._lstm_encoder(encoded)
        z_t_sample, z_f_sample = z_t_dist.rsample(), z_f_dist.rsample()
        
        z_t_prior, _ = self._prior_lstm_encoder(z_t_sample.detach())

        z_f_repeated = z_f_sample.unsqueeze(dim=1).repeat(1, z_t_sample.shape[1], 1)  # repeat by time
        z = torch.cat((z_t_sample, z_f_repeated), dim=2)

        x_hat = self._decoder.forward(z)
        
        dfp_pred = self._dfp(z_t_sample)
        
        vae = calculate_vae_loss(x, x_hat, z_f_dist, z_t_dist, z, z_t_prior)
        dfp = calculate_dfp_loss(x, dfp_pred, labels, self._dfp_criterion, self.device)
        scc = self._triplet_loss(z_f_sample, z_f_pos_dist.rsample(), z_f_neg_dist.rsample())
        mi = calculate_mi_loss(z_t_dist, z_f_dist)
        loss = self._config['lambda_vae']*vae + self._config['lambda_dfp']*dfp + self._config['lambda_scc']*scc + self._config['lambda_mi']*mi

        return loss, vae, dfp, scc, mi, x_hat

    def train_step(self, x, negative_x, labels):
        loss, vae, dfp, scc, mi, _ = self.compute_loss(x, negative_x, labels)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss, vae, scc, dfp, mi

    def validate_step(self, x, negative_x, labels):
        loss, vae, dfp, scc, mi, x_hat = self.compute_loss(x, negative_x, labels)
        return loss, vae, scc, dfp, mi, x_hat