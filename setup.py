from setuptools import setup

# Requirement categories
reqs = ['numpy', 'scipy', 'matplotlib', 'mne>=1.0.0', 'sklearn', 'fslpy',
        'sails', 'tabulate', 'pyyaml>=5.1', 'neurokit2', 'jinja2',
        'glmtools', 'numba', 'nilearn', 'dask', 'distributed', 'parse',
        'opencv-python', 'pythreejs']
doc_reqs = ['sphinx==4.0.2', 'numpydoc', 'sphinx_gallery', 'pydata-sphinx-theme']
dev_reqs = ['setuptools>=41.0.1', 'pytest', 'pytest-cov', 'coverage', 'flake8']

name = 'osl'

setup(name=name,
      version='0.1.0',
      description='OHBA Software Library',
      author='OHBA Analysis Group',
      license='MIT',
      packages=['osl', 'osl.report', 'osl.maxfilter',
                'osl.preprocessing', 'osl.utils', 'osl.utils.spmio',
                'osl.source_recon'],
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'osl_batch = osl.preprocessing.batch:main',
              'osl_maxfilter = osl.maxfilter.maxfilter:main',
              'osl_report = osl.report.raw_report:main',
          ]},

      python_requires='>=3.7',
      install_requires=reqs,
      extras_require={
          'dev': dev_reqs,
          'doc': doc_reqs,
          'full': dev_reqs + doc_reqs,
      },

      package_data={'osl': ['utils/*tree',
                            'utils/simulation_config/*npy',
                            'utils/simulation_config/*fif',
                            'report/templates/*']},

      command_options={
          'build_sphinx': {
              'project': ('setup.py', name),
              'version': ('setup.py', name),
              'release': ('setup.py', name)}},
      )
