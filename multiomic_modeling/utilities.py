import os
import re
import sys
import json
import types
import hashlib
import functools
import subprocess
from copy import deepcopy
from joblib import Parallel, delayed
from collections.abc import MutableMapping

def run_bash_cmd(cmd, shell=False, log=None):
    print(">>> Running the following command:")
    print(cmd)
    print(">>> Output")
    if isinstance(cmd, str):
        proc = subprocess.Popen(cmd, bufsize=1, stdout=sys.stdout, stderr=sys.stderr, shell=True)
        stdout, stderr = proc.communicate()
        if log is not None:
            with open(log, "a") as LOG:
                LOG.write(stdout)
    else:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              encoding='utf-8', shell=shell)
        if log is not None:
            with open(log, "a") as LOG:
                LOG.write(proc.stdout)
    return proc


def is_callable(func):
    FUNCTYPES = (types.FunctionType, types.MethodType, functools.partial)
    return func and (isinstance(func, FUNCTYPES) or callable(func))


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def params_to_hash(all_params):
    all_params = deepcopy(all_params)
    # all_params['dataset_params'].pop('nb_query_samples', 0)
    uid = hashlib.sha1(json.dumps(all_params, sort_keys=True).encode()).hexdigest()
    return uid


def parent_at_depth(filename, depth=1):
    f = os.path.abspath(filename)
    n = len(f.split('/'))
    assert depth < n, "The maximum depth is exceeded"
    for _ in range(depth): f = os.path.dirname(f)
    return f


def batch_run(f, iterable, *args, **kwargs):
    return [f(x, *args, **kwargs) for x in iterable]


def parallel_run(f, iterable, *args, batch_size=256, n_jobs=-1, verbose=1, **kwargs):
    k = batch_size
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(batch_run)(f, iterable[i:i+k], *args, **kwargs)
            for i in range(0, len(iterable), k))
    res = sum(res, [])
    return res


def map_reduce(f_map, f_reduce, iterable, *args, **kwargs):
    return f_reduce([f_map(x, *args, **kwargs) for x in iterable])


def parallel_map_reduce(f_map, f_reduce, iterable, *args,
                        batch_size=256, n_jobs=-1, verbose=1, **kwargs):
    k = batch_size
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(map_reduce)(f_map, f_reduce, iterable[i:i+k], *args, **kwargs)
            for i in range(0, len(iterable), k))
    res = f_reduce(res)
    return res


def natural_keys(text):
    def atoi(text):
        return int(text) if text.isdigit() else text
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]