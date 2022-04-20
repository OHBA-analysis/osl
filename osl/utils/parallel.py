#import multiprocessing as mp
#import multiprocess as mp
from functools import partial
from dask.distributed import Client, LocalCluster
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


def initialise_client(**kwargs):
    """Initialise a Dask client for massive parallel processing.

    This will always start a new cluster and client connection.

    """
    # In general, we probably want lots of processes with relatively few theads
    # per process - this is as we're probably running complex code on MNE data
    # objects rather than pure numpy array operations.
    #
    # If address is not passed, Client creates a cluster using LocalCluster
    # otherwise client will connect to whatever is at the end of the address.
    #
    # Optional environment config to limit threads in processes
    #export OMP_NUM_THREADS=1
    #export MKL_NUM_THREADS=1
    #export OPENBLAS_NUM_THREADS=1

    kwargs.setdefault('processes', True)
    kwargs.setdefault('threads_per_worker', 1)
    kwargs.setdefault('name', 'osl-cluster')
    kwargs.setdefault('scheduler_port', 8786)  # This is the dask-default

    if _dask_find_existing(scheduler_port=kwargs['scheduler_port']) is not None:
        osl_logger.error('Found Dask client running in this session')
        osl_logger.error('Either use existing client/cluster or close before starting a new instance')

    osl_logger.info('Starting new Dask Cluster : {0}'.format(kwargs))
    cluster = LocalCluster(scheduler_port=kwargs['scheduler_port'])
    client = Client(cluster)

    osl_logger.info('Cluster dashboard link: {0}'.format(client.dashboard_link))

    return client


def _dask_find_existing(scheduler_port=8786):
    try:
        client = Client(f'tcp://localhost:{scheduler_port}', timeout='2s')
    except OSError:
        client = None
    return client


def _close_client_and_cluster(client):
    osl_logger.info('Closing dask client: {0}'.format(client))
    if client.cluster is not None:
        client.cluster.close()
    client.close()


def cluster_run_delayed(client, func, iter_args, *args, **kwargs):

    osl_logger.debug('User args : {0}'.format(args))
    osl_logger.debug('User kwargs : {0}'.format(kwargs))

    # Set kwargs - need to handle args on function call to preserve order.
    run_func = partial(func, **kwargs)

    # Cue up jobs
    lazy_results = []
    for idx, aa in enumerate(iter_args):
        lazy_results.append(dask.delayed(run_func)(aa, *args))
    osl_logger.info('Prepared {0} jobs for processing'.format(len(lazy_results)))

    #persist or compute? probably persist
    # https://distributed.dask.org/en/latest/manage-computation.html#dask-collections-to-futures
    osl_logger.info('Starting computation')
    futures = dask.persist(*lazy_results)  # trigger computation in the background

    osl_logger.info('Complete')
    return futures
