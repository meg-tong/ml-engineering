import glob

import torch as t
import yaml
import os
import requests

import transformer_replication

def load_transformer(run_id, model_class, base_config: transformer_replication.TransformerConfig = None, vocab_size = None):
    root = '/Users/m/Documents/arena/wandb/'
    model_path = glob.glob(
        f'{root}/run-*-{run_id}/files/model_state_dict.pt'
    )[0]

    if base_config is None:
        yaml_path = glob.glob(
            f'{root}/run-*-{run_id}/files/config.yaml'
        )[0]
        with open(yaml_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f)

        base_config = transformer_replication.TransformerConfig(
            num_layers=yaml_cfg['num_layers']['value'],
            num_heads=yaml_cfg['num_heads']['value'],
            vocab_size=(yaml_cfg['vocab_size']['value'] if vocab_size is None else vocab_size),
            hidden_size=yaml_cfg['hidden_size']['value'],
            max_seq_len=yaml_cfg['max_seq_len']['value'],
            dropout=yaml_cfg['dropout']['value']
        )

    model = model_class(base_config)
    state_dict = t.load(
        model_path
    )
    model.load_state_dict(state_dict)
    return model

def maybe_download(url: str, path: str) -> None:
    '''
    Download the file from url and save it to path. 
    If path already exists, do nothing.
    '''
    if os.path.isfile(path):
        return
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.raw.read())