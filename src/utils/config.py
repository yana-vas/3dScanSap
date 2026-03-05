import yaml
from pathlib import Path
from typing import Any, Optional


class ConfigDict(dict):

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        except KeyError:
            raise AttributeError(f"Config has no attriibute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class Config:

    DEFAULTS = {
        'model': {
            'latent_dim': 256,
            'hidden_dim': 256,
            'num_layers': 5,
            'encoder': 'resnet18'
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 0.0001,
            'num_epochs': 100,
            'num_points': 2048
        },
        'inference': {
            'grid_resolution': 64,
            'threshold': 0.5
        },
        'data': {
            'image_size': 224,
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225]
        }
    }

    def __init__(self, config_dict: Optional[dict] = None):
        self._config = ConfigDict(self.DEFAULTS.copy())
        if config_dict:
            self._deep_update(self._config, config_dict)

    def _deep_update(self, base: dict, update: dict) -> None:
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        config = cls()
        yaml_path = Path(path)
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    config._deep_update(config._config, yaml_config)
        return config

    @property
    def model(self) -> ConfigDict:
        return ConfigDict(self._config['model'])

    @property
    def training(self) -> ConfigDict:
        return ConfigDict(self._config['training'])

    @property
    def inference(self) -> ConfigDict:
        return ConfigDict(self._config['inference'])

    @property
    def data(self) -> ConfigDict:
        return ConfigDict(self._config['data'])


def load_config(path: str = 'configs/default.yaml') -> Config:
    return Config.from_yaml(path)
