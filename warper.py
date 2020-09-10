import numpy as np

import nsdesolve
# from nsdesolve import simulate_2d, simulate_1d

# def apply_kwargs(fun, kwargs):
#     return fun(**kwargs)

# def REP(pool, runs, fun, **kwargs):
#     result = pool.starmap(apply_kwargs, 
#         [(fun, kwargs)]*runs
#     )
#     return result
#     return [np.concatenate(single, axis=0) for single in zip(*result)]

def _simulate_1d_kwargs(kwargs):
    return nsdesolve.simulate_1d(**kwargs)

def simulate_1d(pool=None,runs=1, **kwargs):
    if pool is None:
        assert runs == 1
        return _simulate_1d_kwargs(kwargs)
    
    result = pool.map(_simulate_1d_kwargs, 
        [kwargs]*runs
    )
    return [np.concatenate(single, axis=0) for single in zip(*result)]

def _simulate_2d_kwargs(kwargs):
    return nsdesolve.simulate_2d(**kwargs)

def simulate_2d(pool=None,runs=1, **kwargs):
    if pool is None:
        assert runs == 1
        return _simulate_2d_kwargs(kwargs)
    
    result = pool.map(_simulate_2d_kwargs, 
        [kwargs]*runs
    )
    return [np.concatenate(single, axis=0) for single in zip(*result)]


# if __name__ == "__main__":
#     simulate_1d()