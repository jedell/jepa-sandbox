import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from masks.utils import apply_masks


class JEPA(nn.Module):
    def __init__(self, encoder, target_encoder, predictor, hparams, training=True):
        super(JEPA, self).__init__()

        self.encoder = encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.norm = nn.LayerNorm(hparams.embedding_dim)

        self.enc_mu = nn.Linear(hparams.embedding_dim, hparams.latent_dim)
        self.enc_logvar = nn.Linear(hparams.embedding_dim, hparams.latent_dim)
        self.projection_v = MLP(hparams.embedding_dim, hparams.latent_dim, hparams.hidden_dim, hparams.mlp_num_layers)

        self.hparams = hparams
        self.training = training

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def vicreg_loss(self, z_a, z_b):
        assert z_a.shape == z_b.shape and len(z_a.shape) == 2

        # invariance loss
        loss_inv = F.mse_loss(z_a, z_b)

        # variance loss
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)

        std_z_a = torch.sqrt(z_a.var(dim=0) + self.hparams.variance_loss_epsilon)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.hparams.variance_loss_epsilon)
        loss_v_a = torch.mean(F.relu(1 - std_z_a)) / 2
        loss_v_b = torch.mean(F.relu(1 - std_z_b)) / 2
        loss_var = loss_v_a + loss_v_b

        # covariance loss
        N, D = z_a.shape # batch size, ...
        cov_z_a = ((z_a.T @ z_a) / (N - 1))  # DxD
        cov_z_b = ((z_b.T @ z_b) / (N - 1))  # DxD
        loss_cov = self.off_diagonal(cov_z_a).pow_(2).sum().div(
            D
        ) + self.off_diagonal(cov_z_b).pow_(2).sum().div(D)

        weighted_inv = self.hparams.invariance_loss_weight * loss_inv
        weighted_var = self.hparams.variance_loss_weight * loss_var
        weighted_cov = self.hparams.covariance_loss_weight * loss_cov

        loss = weighted_inv + weighted_var + weighted_cov

        return {
            "loss": loss,
            "loss_invariance": weighted_inv,
            "loss_variance": weighted_var,
            "loss_covariance": weighted_cov,
        }
    
    # def _reparam(self, mu, logvar):
    #     if self.training:
    #         std = logvar.mul(0.5).exp_()
    #         eps = std.new_empty(std.size()).normal_()
    #         return eps.mul_(std).add_(mu)
    #     else:
    #         return mu

    def forward_context(self, c, target_enc, masks_enc, masks_pred):
        z = self.encoder(c, masks_enc)
        z = self.predictor(z, target_enc, masks_enc, masks_pred)
        # z = self.norm(z)

        return z
    
    def forward_target(self, y, masks_pred):
        with torch.no_grad():
            # print(f"Shape of y: {y.shape}")
            h = self.target_encoder(y)
            # print(f"Shape of h after target_encoder: {h[0].shape}, {h[1].shape}, {len(h)}")
            h = F.layer_norm(h, (h.size(-1),))
            # print(f"Shape of h after layer_norm: {h[0].shape}, {h[1].shape}, {len(h)}")
            h = apply_masks(h, masks_pred, concat=False)
            # print(f"Shape of h after apply_masks: {h[0].shape}, {h[1].shape}, {len(h)}")
        return h
    
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
        
    def loss_fn(self, z, h, mask_pred):
        loss = 0.
        for zi, hi in zip(z, h):
            loss += torch.mean(torch.abs(zi - hi)**self.hparams.loss_exp) / self.hparams.loss_exp
        loss /= len(mask_pred)
        return loss

    def reg_fn(self, z):
        return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)


    # frames: (b, num_frames=11, c, h, w)
    # concatenated over batch dim (b * num_frames, c, h, w)
    def forward(self, x, masks_pred, masks_enc):

        # print(f"Shape of x: {x.shape}, len of x: {len(x)}")
        # print(f"len mask_pred and mask_enc: {len(masks_pred)}, {len(masks_enc)}")
        target_enc = self.forward_target(x, masks_pred)
        # print(f"Shape of target_enc: {target_enc[0].shape}, {target_enc[1].shape}, len of target_enc: {len(target_enc)}")

        context_enc = self.forward_context(x, target_enc, masks_enc, masks_pred)
        # print(f"Shape of context_enc: {context_enc[0].shape}, {context_enc[1].shape}, len of context_enc: {len(context_enc)}")

        loss_jepa = self.loss_fn(context_enc, target_enc, masks_pred)

        pstd_z = self.reg_fn(context_enc)
        loss_reg = torch.mean(F.relu(1.-pstd_z))

        losses = {"loss": loss_jepa + self.hparams.reg_loss_coeff * loss_reg}

        # TODO, for ssl predicting 11 from 11 frames use vicreg
        # mu, logvar = self._get_mu_logvar(context_enc)
        # print(f"Shape of mu: {mu.shape}, Shape of logvar: {logvar.shape}")

        # z = self._reparam(mu, logvar)
        # print(f"Shape of z: {z.shape}")

        # # L1 norm of z (b, embed_dim), want to minimize this
        # loss_latent = torch.linalg.vector_norm(z, ord=1, dim=1).mean()
        # print(f"Value of loss_latent: {loss_latent}")

        # losses = self.vicreg_loss(context_enc, target_enc)
        # print(f"Losses before adding loss_latent: {losses}")

        # losses["loss_latent"] = loss_latent
        # losses["loss"] += (loss_latent * self.hparams.latent_loss_lambda)
        # print(f"Losses after adding loss_latent: {losses}")

        # context_enc = context_enc + z
        # print(f"Shape of updated context_enc: {context_enc.shape}")

        # target_pred = self.predictor(context_enc)
        # print(f"Shape of target_pred: {target_pred.shape}")

        return context_enc, target_enc, losses


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
