from afr.models.cnn import CNN
import yaml
from omegaconf import OmegaConf



import torch


path = '/u/mrudolph/documents/afr/artifacts/afr/cheetah/2025-09-14-07-05-40/pla_models.pt'
config_path = '/u/mrudolph/documents/afr/artifacts/afr/cheetah/2025-09-14-07-05-40/config.yaml'

config = OmegaConf.load(config_path)
state_dict = torch.load(path)

encoder_dict = state_dict['encoder']


encoder = CNN(config.model.encoder)
encoder.load_state_dict(encoder_dict)

import pdb; pdb.set_trace()