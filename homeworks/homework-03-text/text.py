from pathlib import Path

import click
from loguru import logger

from cli import preprocess as _preprocess
from cli import regressors
from cli import train as _train


@click.group()
def cli():
    pass


@cli.command('preprocess')
@click.argument('input_file',
                type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument('output_file',
                type=click.Path(dir_okay=False, writable=True))
def preprocess(input_file: str, output_file: str):
    _preprocess(input_file, output_file)


@cli.command()
@click.option('--data-path',
              type=click.Path(dir_okay=True, readable=True),
              default='data')
@click.option('--pred-path',
              type=click.Path(file_okay=False, writable=True),
              default='predictions')
@click.argument('model_name', type=click.Choice(list(regressors.keys()), case_sensitive=False))
def train(data_path: str, pred_path: str, model_name: str):
    _train(model_name, Path(data_path), Path(pred_path))


if __name__ == '__main__':
    cli()
