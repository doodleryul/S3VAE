import click
from s3vae.train import run_prototype
from s3vae.utils import load_config
from pkg_resources import resource_filename

@click.group()
def cli():
    pass

@cli.command(short_help='Generate moving mnist with s3vae')
# @click.option('--config_filename', '-c', type=click.STRING, help='config filename')
def generate_sequential_image():
    config_filename = resource_filename(__name__, '../s3vae_config.yaml')
    config = load_config(config_filename)
    run_prototype(config)


if __name__ == '__main__':
    cli()
