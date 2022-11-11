import pathlib

from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Requirement categories
reqs = ['numpy', 'scipy', 'matplotlib', 'mne>=1.0.0', 'sklearn', 'fslpy',
        'sails', 'tabulate', 'pyyaml>=5.1', 'neurokit2', 'jinja2',
        'glmtools', 'numba', 'nilearn', 'dask', 'distributed', 'parse',
        'opencv-python', 'pythreejs']
doc_reqs = ['sphinx==4.0.2', 'numpydoc', 'sphinx_gallery', 'pydata-sphinx-theme']
dev_reqs = ['setuptools>=41.0.1', 'pytest', 'pytest-cov', 'coverage', 'flake8']

name = 'osl'

setup(name=name,
      version='0.1.1',
      description='OHBA Software Library',
      long_description=README,
      long_description_content_type="text/markdown",
      author='OHBA Analysis Group',
      license='MIT',

      # Choose your license
      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
          'Development Status :: 4 - Beta',

          # Indicate who your project is intended for
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Mathematics',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
      ],

      python_requires='>=3.7',
      install_requires=reqs,
      extras_require={
          'dev': dev_reqs,
          'doc': doc_reqs,
          'full': dev_reqs + doc_reqs,
      },

      zip_safe=False,
      entry_points={
          'console_scripts': [
              'osl_batch = osl.preprocessing.batch:main',
              'osl_maxfilter = osl.maxfilter.maxfilter:main',
              'osl_report = osl.report.raw_report:main',
          ]},

      packages=['osl', 'osl.report', 'osl.maxfilter',
                'osl.preprocessing', 'osl.utils', 'osl.utils.spmio',
                'osl.source_recon', 'osl.source_recon.rhino',
                'osl.source_recon.parcellation'],


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
