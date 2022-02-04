import click
from cli import preprocess as _preprocess
from cli import train as _train
from cli import regressors
from loguru import logger


@click.group()
def cli():
    pass


@cli.command('preprocess')
@click.option('-i', '--input', 'input_file',
              type=click.Path(exists=True, dir_okay=False, readable=True),
              default='data/train.csv')
@click.option('-o', '--output', 'output_file',
              type=click.Path(dir_okay=False, writable=True),
              default='data/processed.pickle')
def preprocess(input_file: str, output_file: str):
    _preprocess(input_file, output_file)


@cli.command()
@click.option('--train-path',
              type=click.Path(dir_okay=False, readable=True),
              default='data/processed_train.pickle')
@click.option('--test-path',
              type=click.Path(dir_okay=False, readable=True),
              default='data/processed_test.pickle')
@click.option('--pred-path',
              type=click.Path(file_okay=False, writable=True),
              default='predictions')
@click.argument('model_name', type=click.Choice(list(regressors.keys()), case_sensitive=False))
def train(train_path: str, test_path: str, pred_path: str, model_name: str):
    _train(model_name, train_path, test_path, pred_path)


if __name__ == '__main__':
    cli()
