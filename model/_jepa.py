import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class JEPA(nn.Module):
    def __init__(self, encoder_x, encoder_y, predictor, hparams, training=True):
        super(JEPA, self).__init__()

        self.encoder_x = encoder_x
        self.encoder_y = encoder_y
        self.predictor = predictor
        self.norm = nn.LayerNorm(hparams.embedding_dim)

        self.enc_mu = nn.Linear(hparams.embedding_dim, hparams.latent_dim)
        self.enc_logvar = nn.Linear(hparams.embedding_dim, hparams.latent_dim)
        self.projection_v = MLP(hparams.embedding_dim, hparams.latent_dim, hparams.hidden_dim, hparams.mlp_num_layers)

        self.hparams = hparams
        self.training = training

    def _get_vicreg_loss(self, z_a, z_b):
        assert z_a.shape == z_b.shape and len(z_a.shape) == 2

        # invariance loss
        loss_inv = F.mse_loss(z_a, z_b)

        # variance loss
        std_z_a = torch.sqrt(
            z_a.var(dim=0) + self.hparams.variance_loss_epsilon)
        std_z_b = torch.sqrt(
            z_b.var(dim=0) + self.hparams.variance_loss_epsilon)
        loss_v_a = torch.mean(F.relu(1 - std_z_a))
        loss_v_b = torch.mean(F.relu(1 - std_z_b))
        loss_var = loss_v_a + loss_v_b

        # covariance loss
        N, D = z_a.shape
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = ((z_a.T @ z_a) / (N - 1)).square()  # DxD
        cov_z_b = ((z_b.T @ z_b) / (N - 1)).square()  # DxD
        loss_c_a = (cov_z_a.sum() - cov_z_a.diagonal().sum()) / D
        loss_c_b = (cov_z_b.sum() - cov_z_b.diagonal().sum()) / D
        loss_cov = loss_c_a + loss_c_b

        weighted_inv = loss_inv * self.hparams.invariance_loss_weight
        weighted_var = loss_var * self.hparams.variance_loss_weight
        weighted_cov = loss_cov * self.hparams.covariance_loss_weight

        loss = weighted_inv + weighted_var + weighted_cov

        return {
            "loss": loss,
            "loss_invariance": weighted_inv,
            "loss_variance": weighted_var,
            "loss_covariance": weighted_cov,
        }
    
    def _reparam(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new_empty(std.size()).normal_()
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def _get_embedding(self, x, y):
        x_embed = self.encoder_x(x)
        x_embed = self.norm(x_embed)

        y_embed = self.encoder_y(y)
        y_embed = self.norm(y_embed)

        return x_embed, y_embed
    
    def _get_mu_logvar(self, mu_logvar):
        mu = self.enc_mu(mu_logvar)
        logvar = self.enc_logvar(mu_logvar)
        return mu, logvar
    
    def _reparam(self, mu, logvar):
        """Reparameterisation trick to sample z values. 
        This is stochastic during training,  and returns the mode during evaluation."""
        
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def _get_z(self, x):
        x_embed = self.encoder_x(x)
        mu, logvar = self._get_mu_logvar(x_embed)
        z = self._reparam(mu, logvar)
        return z

    # frames: (b, num_frames=22, c, h, w)
    def forward(self, x, y):

        # x_embed: (b, embed_dim)
        # y_embed: (b, embed_dim)
        x_embed, y_embed = self._get_embedding(x, y)

        mu, logvar = self._get_mu_logvar(x_embed)
        z = self._reparam(mu, logvar)
        # L1 norm of z (b, embed_dim), want to minimize this
        loss_latent = torch.linalg.vector_norm(z, ord=1, dim=1).mean()

        losses = self._get_vicreg_loss(x_embed, y_embed)

        losses["loss_latent"] = loss_latent
        losses["loss"] += (loss_latent * self.hparams.latent_loss_lambda)

        x_embed = x_embed + z

        # Predicted y (b, embed_dim)
        pred_y = self.predictor(x_embed)

        return pred_y, y_embed, losses


class MLP(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim, num_layers
    ):
        super().__init__()
        assert num_layers >= 0, "negative layers?!?"

        if num_layers == 0:
            self.net = nn.Identity()
            return

        if num_layers == 1:
            self.net = nn.Linear(input_dim, output_dim)
            return

        linear_net = nn.Linear

        layers = []
        prev_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(linear_net(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
