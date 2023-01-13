from s3vae.models import Encoder, LSTMEncoder, Decoder, DynamicFacorPrediction
from s3vae.losses import calculate_vae_loss, calculate_dfp_loss, calculate_mi_loss
from itertools import chain
import torch
import os
import torch.nn as nn

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
        self._dfp = DynamicFacorPrediction(z_t_size=self._lstm_encoder.z_size).to(self.device)
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

    def compute_loss(self, x, negative_x):
        x = x.to(self.device)
        negative_x = negative_x.to(self.device)
        
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
        dfp = calculate_dfp_loss(x, dfp_pred, self._dfp_criterion, self.device)
        scc = self._triplet_loss(z_f_sample, z_f_pos_dist.rsample(), z_f_neg_dist.rsample())
        mi = calculate_mi_loss(z_t_dist, z_f_dist)
        loss = self._config['lambda_vae']*vae + self._config['lambda_dfp']*dfp + self._config['lambda_scc']*scc + self._config['lambda_mi']*mi

        return loss, vae, dfp, scc, mi, x_hat

    def train_step(self, x, negative_x):
        loss, vae, dfp, scc, mi, _ = self.compute_loss(x, negative_x)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss, vae, scc, dfp, mi

    def validate_step(self, x, negative_x):
        loss, vae, dfp, scc, mi, x_hat = self.compute_loss(x, negative_x)
        return loss, vae, scc, dfp, mi, x_hat