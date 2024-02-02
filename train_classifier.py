import click
from lib import Classifier

@click.command()
@click.argument('data_dir')
@click.argument('model_dir')
def train(data_dir, model_dir):
    classifier = Classifier(model_dir, None, {})
    classifier.train(data_dir)
    print("Model trained successfully")


if __name__ == '__main__':
    train()
