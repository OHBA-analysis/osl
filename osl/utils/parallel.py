#import multiprocessing as mp
#import multiprocess as mp
from functools import partial
from dask.distributed import Client, LocalCluster, wait
import dask

# Housekeeping for logging
import logging
osl_logger = logging.getLogger(__name__)

def initialise_pool(nprocesses=1):
    """Set-up a python multiprocessing pool that shouldn't randomly stall.

    Notes
    -----
    Multiprocessing has a common bug described here:
    - https://pythonspeed.com/articles/python-multiprocessing/
    - https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/
    Use of this function should avoid this issue.

    Could use joblib for this but joblib.Parallel can't handle errors in child processes.

    """
    # Ensure we have a resonable number of parallel processes compared to number of CPUs
    cpus = mp.cpu_count()
    if nprocesses > cpus:
        msg = 'Requested {0} processes but only {1} CPUs available'
        raise ValueError(msg.format(nprocesses, cpus))

    P = mp.get_context('spawn').Pool(processes=nprocesses)

    return P


def dask_parallel_bag(func, iter_args,
                      func_args=None, func_kwargs=None):
    """A maybe more consistent alternative to dask_parallel."""

    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs

    client = dask.distributed.default_client()

    osl_logger.info('Dask Client : {0}'.format(client.__repr__()))
    osl_logger.info('Dask Client dashboard link: {0}'.format(client.dashboard_link))

    osl_logger.debug('Running function : {0}'.format(func.__repr__()))
    osl_logger.debug('User args : {0}'.format(func_args))
    osl_logger.debug('User kwargs : {0}'.format(func_kwargs))

    # Set kwargs - need to handle args on function call to preserve order.
    run_func = partial(func, **func_kwargs)
    osl_logger.info('Function defined : {0}'.format(run_func))

    import dask.bag as db
    b = db.from_sequence(iter_args)

    #bm = db.map(run_func, b, *func_args)
    bm = b.starmap(run_func, *func_args)
    flags = bm.compute()

    osl_logger.info('Computation complete')

    return flags


def dask_parallel(client, func, iter_args,
                  func_args=None, func_kwargs=None,
                  block_console=True,
                  ret_results=False):
    """Run a function across many sets of inputs in parallel using Dask.

    Parameters
    ----------
    client : dask.distributed.Client instance
        Initalised dask client to handle job running
    func : function
        Function to be executed
    iter_args : list, tuple
        Input arguments to be run one at a time. eg func will be called once
        for each item in iter_args. These are assumed to be positional
        arguments of func.
    func_args : list, tuple
        Optional positional arguments that are fixed across all jobs
    func_kwargs : dict
        Optional keyword arguments that are fixed across all jobs

    Returns
    -------
    Dask futures or list of results of func depending on value of ret_results.
    """

    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs

    osl_logger.info('Dask Client : {0}'.format(client.__repr__()))
    osl_logger.info('Dask Client dashboard link: {0}'.format(client.dashboard_link))

    osl_logger.debug('Running function : {0}'.format(func.__repr__()))
    osl_logger.debug('User args : {0}'.format(func_args))
    osl_logger.debug('User kwargs : {0}'.format(func_kwargs))

    # Set kwargs - need to handle args on function call to preserve order.
    run_func = partial(func, **func_kwargs)

    # Cue up jobs
    lazy_results = []
    for idx, aa in enumerate(iter_args):
        if isinstance(aa, (list, tuple)):
            # Assume user has provided ALL positional args
            lazy_results.append(dask.delayed(run_func)(*aa))
        else:
            # Assume user has provided ONLY FIRST positional arg - rest are static
            lazy_results.append(dask.delayed(run_func)(aa, *func_args))
    osl_logger.info('Prepared {0} jobs for processing'.format(len(lazy_results)))

    #persist or compute? probably persist
    # https://distributed.dask.org/en/latest/manage-computation.html#dask-collections-to-futures
    osl_logger.info('Starting computation')
    futures = dask.persist(*lazy_results)  # trigger computation in the background

    # This should probably always be True
    if block_console:
        wait(futures)

    osl_logger.info('Computation complete')

    if ret_results:
        osl_logger.info('Gathering results')
        results = client.gather(futures)
        results = dask.compute(results)[0]
        return results

    return futures
