import click
from src.train import train
from src.evaluate import evaluate


@click.group()
def cli():
    """CLI for classic ML pipeline."""


@cli.command(name="train-cmd")
@click.option("--raw-name", default="sample.csv")
@click.option("--model-name", default="logreg")
def train_cmd(raw_name: str, model_name: str):
    result = train(raw_name=raw_name, model_name=model_name)
    click.echo(result)


@cli.command(name="evaluate-cmd")
@click.option("--raw-name", default="sample.csv")
@click.option("--model-path", default="models/model_logreg.joblib")
def evaluate_cmd(raw_name: str, model_path: str):
    result = evaluate(raw_name=raw_name, model_path=model_path)
    click.echo(result)


if __name__ == "__main__":
    cli()
