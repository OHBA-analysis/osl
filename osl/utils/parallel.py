import multiprocessing


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
    cpus = multiprocessing.cpu_count()
    if nprocesses > cpus:
        msg = 'Requested {0} processes but only {1} CPUs available'
        raise ValueError(msg.format(nprocesses, cpus))

    P = multiprocessing.get_context('spawn').Pool(processes=nprocesses)

    return P
