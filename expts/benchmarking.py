import os
import json
import glob
import click
import natsort
from itertools import product
from sklearn.model_selection import ParameterGrid
from multiomic_modeling.utilities import natural_keys


def get_params(algo, debug):
        n_epochs, n_samples = (2, 1000) if debug else (100, None)

        format_keys = ["model_params.model_name",
                       "model_params.d_model",
                       "model_params.n_layers",
                       "model_params.n_heads",
                       "model_params.dropout",
                       "seed"]

        d_model = 256
        batch_size = 192
        shared_model_params = dict(
            d_model=[64 if debug else d_model],
            d_ff=[64 if debug else d_model*4],
            n_heads=[2 if debug else 8],
            n_layers=[2 if debug else 4],
            dropout=[0.1],
            embedding_dropout=[0],
            share_embedding=[False],
            lr=[1],   # [1],
            optimizer=["Adam"],
            lr_scheduler=["cosine_with_restarts"],   # ["cos_w_restart"],
            n_epochs=[100],
            precision=[16],
            batch_size=[8 if debug else batch_size],
            early_stopping=[False],
            auto_scale_batch_size=[False],
            accumulate_grad_batches=[256//batch_size],
            min_batch_size=[32],
            max_batch_size=[258],
            amp_backend=['native'],
            amp_level=['O2'],
            auto_lr_find=[False],
            min_lr=[1e-6],
            max_lr=[1],
        )

        other_shared_params = dict(
            fit_params=[dict(nb_ckpts=10, verbose=1)],
            predict_params=[dict(nb_ckpts=10)],
            outfmt_keys=[format_keys],
            seed=[42] if debug else [42],
        )

        torchmt = dict(
            model_params=list(ParameterGrid(dict(
                model_name=['torchmt'],
                **shared_model_params,
            ))),
            **other_shared_params
        )

        return locals().get(algo, None)


@click.command()
@click.option('-a', '--algos', default='single', type=str, help='Names of the algos')
@click.option('--debug', is_flag=True, default=False)
@click.option('-o', '--outfile', type=str, default='configs.json')
def generate_config_file(algos, debug, outfile):
    print(algos, debug, outfile)
    algo_grids = []
    for algo in algos.split():
        p = get_params(algo, debug)
        if isinstance(p, list):
            algo_grids += p
        elif isinstance(p, dict):
            algo_grids.append(p)
        else:
            raise Exception("Algo parameter must be a list or a dict")

    expt_config = algo_grids
    grid = ParameterGrid(expt_config)
    print(f'>> There are {len(grid)} experiments')
    with open(outfile, 'w') as f:
        json.dump(expt_config, f, indent=2)


if __name__ == '__main__':
    generate_config_file()