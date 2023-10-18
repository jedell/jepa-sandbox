class ModelParams:

    invariance_loss_weight: float = 25.0
    variance_loss_weight: float = 25.0
    covariance_loss_weight: float = 1.0
    variance_loss_epsilon: float = 1e-04
    latent_loss_lambda: float = 1.0

    embedding_dim: int = 128
    latent_dim: int = 128
    hidden_dim: int = 256
    mlp_num_layers: int = 3
    dropout: float = 0.0

JEPAParams = ModelParams

