# Conda environments

One environment file per platform, named as conda names them (`conda info`
shows yours under "platform"). Each installs the compiled dependencies (Qt,
VTK, ONNX Runtime, ...) from conda-forge and osl-ephys itself from PyPI, which
is the closest to what a conda package of osl-ephys would install.

- `linux-64.yml`: Linux on x86-64.
- `linux-64-headless.yml`: Linux x86-64 servers without a display
  (software-rendering VTK, set `PYVISTA_OFF_SCREEN=true`).
- `linux-aarch64.yml`: Linux on ARM64.
- `osx-64.yml`: Intel Macs.
- `osx-arm64.yml`: Apple silicon Macs.
- `win-64.yml`: Windows (source reconstruction needs FSL, so WSL with
  `linux-64.yml`).

Create and activate the environment with:

```
conda env create -f envs/<platform>.yml
conda activate osle
```

See the [installation instructions](https://osl-ephys.readthedocs.io/en/latest/install.html)
for getting conda and the alternatives.
