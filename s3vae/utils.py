import yaml

def load_config(config_dir):
    with open(config_dir, 'r') as yml:
        config = yaml.safe_load(yml)

    return config