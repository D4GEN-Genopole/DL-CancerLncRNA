import click
from utils.evaluate import SequencesEvaluator, ExpressionsEvaluator, SequencesExpressionsEvaluator
from models.sequences.baselines import *
from models.expressions.baselines import *
from models.sequences_expressions.baselines import *
from models.sequences.gru import GRUModel
from utils.visualize import Visualization
import os


@click.group()
def cli():
    pass


@click.command(name='eval',
               context_settings={'ignore_unknown_options': True,
                                 'allow_extra_args': True})
@click.argument('model_cls')
@click.pass_context
def eval(ctx, model_cls):
    try:
        model_cls = globals()[model_cls]
    except KeyError:
        raise Warning(f'Unknown model class {model_cls} ! Try importing it in cli.py')
    assert 'sequences' in model_cls.__module__ or 'expressions' in model_cls.__module__

    kwargs = {}
    for i in range(0, len(ctx.args), 2):
        key, value = ctx.args[i], ctx.args[i + 1]
        key = key.lstrip('-')
        try:
            value = float(value)
        except ValueError:
            pass
        kwargs[key] = value

    model = model_cls(**kwargs)
    evaluator_cls = SequencesExpressionsEvaluator if 'sequences_expressions' in model_cls.__module__ \
                    else SequencesEvaluator if 'sequences' in model_cls.__module__ \
                    else ExpressionsEvaluator
    evaluator_cls(model).evaluate()

@click.command(name='visualize_data',
               context_settings={'ignore_unknown_options': True,
                                 'allow_extra_args': True})
def visualize_data():
    viz = Visualization('data')
    viz.plot_cancer_sequence()
    viz.plot_cancer_expression()
    viz.plot_intersection()


cli.add_command(eval)
cli.add_command(visualize_data)
if __name__ == '__main__':
    cli()
