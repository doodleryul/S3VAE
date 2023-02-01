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
    config = load_config(config_filename)
    data_dir = config['filename']
    run(config, data_dir)


if __name__ == '__main__':
    cli()
