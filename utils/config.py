class ModelParams:

    invariance_loss_weight: float = 25.0
    variance_loss_weight: float = 25.0
    covariance_loss_weight: float = 1.0
    variance_loss_epsilon: float = 1e-04
    latent_loss_lambda: float = 1.0
    loss_exp: float = 1.0
    reg_loss_coeff: float = 0.0

    embedding_dim: int = 128
    latent_dim: int = 128
    hidden_dim: int = 256
    mlp_num_layers: int = 3
    dropout: float = 0.0

JEPAParams = ModelParams

import yaml

class ConfigLoader:
    def __init__(self, config_path='jepa/configs/small.yaml'):
        self.config_path = config_path
        self.configs = self.load_configs()

    def load_configs(self):
        with open(self.config_path, 'r') as file:
            configs = yaml.safe_load(file)
        return configs

    def get_data_configs(self):
        data_configs = self.configs.get('data', {})

        collate = data_configs.get('collate', {})
        # convert to crop size from list to tuple
        collate['crop_size'] = tuple(collate['crop_size'])

        return {
            'unlabel_dir': data_configs.get('unlabel_dir', 'data/Dataset_Student/unlabeled'),
            'batch_size': data_configs.get('batch_size', 24),
            'patch_size': data_configs.get('patch_size', 16),
            'pin_mem': data_configs.get('pin_mem', True),
            'num_workers': data_configs.get('num_workers', 4),
            'drop_last': data_configs.get('drop_last', True),
            'num_clips': data_configs.get('num_clips', 1),
            'collate': collate,
        }

    def get_mask_configs(self):
        mask_configs = self.configs.get('mask', [])
        return [self.parse_mask_config(config) for config in mask_configs]

    @staticmethod
    def parse_mask_config(config):
        return {
            'aspect_ratio': tuple(config.get('aspect_ratio', (0.75, 1.5))),
            'num_blocks': config.get('num_blocks', 8),
            'spatial_scale': tuple(config.get('spatial_scale', (0.15, 0.15))),
            'temporal_scale': tuple(config.get('temporal_scale', (1.0, 1.0))),
            'max_temporal_keep': config.get('max_temporal_keep', 1.0),
            'max_keep': config.get('max_keep', None),
        }

    def get_optimizer_configs(self):
        optimizer_configs = self.configs.get('optimizer', {})
        return {
            'ipe': optimizer_configs.get('ipe', 300),
            'ipe_scale': optimizer_configs.get('ipe_scale', 1.25),
            'clip_grad': optimizer_configs.get('clip_grad', 10.0),
            'weight_decay': optimizer_configs.get('weight_decay', 0.04),
            'final_weight_decay': optimizer_configs.get('final_weight_decay', 0.4),
            'num_epochs': optimizer_configs.get('num_epochs', 300),
            'warmup': optimizer_configs.get('warmup', 40),
            'start_lr': optimizer_configs.get('start_lr', 0.0002),
            'lr': optimizer_configs.get('lr', 0.000625),
            'final_lr': optimizer_configs.get('final_lr', 1.0e-06),
            'ema': tuple(optimizer_configs.get('ema', [0.998, 1.0])),
        }

# Example usage:
# config_loader = ConfigLoader()
# mask_configs = config_loader.get_mask_configs()
