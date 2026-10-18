# Conda environments

One environment file per platform, each installing the compiled dependencies
(Qt, VTK, ONNX Runtime, ...) from conda-forge and osl-ephys itself from PyPI.
This is the closest to what a conda package of osl-ephys would install.

- `linux.yml`: Linux with a display.
- `linux-headless.yml`: Linux servers without a display (software-rendering
  VTK, set `PYVISTA_OFF_SCREEN=true`).
- `macos.yml`: macOS, Intel or Apple silicon.
- `windows.yml`: Windows (source reconstruction needs FSL, so WSL with
  `linux.yml`).

Create and activate the environment with:

```
conda env create -f envs/<platform>.yml
conda activate osle
```

See the [installation instructions](https://osl-ephys.readthedocs.io/en/latest/install.html)
for getting conda and the alternatives.
