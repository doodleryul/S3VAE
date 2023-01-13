import click
from s3vae.train import run
from s3vae.utils import load_config
from pkg_resources import resource_filename

@click.group()
def cli():
    pass

@cli.command(short_help='Generate moving mnist with s3vae')
@click.option('--config_filename', '-c', type=click.STRING, help='config filename')
def generate_moving_mnist(config_filename):
    config_dir = resource_filename(__name__, f'../{config_filename}')
    data_dir = resource_filename(__name__, '../data')
    print(data_dir)
    config = load_config(config_dir)
    run(config, data_dir)


if __name__ == '__main__':
    cli()