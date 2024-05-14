Example scripts for parallelising batch processing with Dask
------------------------------------------------------------

If you're using a computer with multiple cores, we can batch processes (preprocess, source reconstruct) in parallel. The scripts `serial_*.py` use code that runs serially (i.e. without parallel workers) and the `parallel_*.py` scripts show the changes that need to be made to run the same processing in parallel.

There are 3 things you need to do parallelise a script:
 
1. Add 

```
if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
```

at the top of the script.

2. Setup a Dask client with

```
client = Client(n_workers=2, threads_per_worker=1)
```

3. Pass

```
dask_client=True
```

to the osl batch processing function.


Note, running a script in parallel with Dask will create a `dask-worker-space` directory. This can be safely deleted after the script has finished.
