from setuptools import setup

# Requirement categories
reqs = ['numpy', 'scipy', 'matplotlib', 'mne', 'sklearn', 'fslpy', 'sails', 'tabulate', 'pyyaml>=5.1', 'neurokit2', 'jinja2']
dev_reqs = ['setuptools>=41.0.1', 'pytest', 'pytest-cov', 'coverage', 'flake8']


setup(name='osl',
      version='0.0.1.dev',
      description='OHBA Software Library',
      author='OHBA Analysis Group',
      license='MIT',
      packages=['osl', 'osl.report', 'osl.maxfilter',
                'osl.preprocessing', 'osl.utils', 'osl.utils.spmio'],
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'osl_batch = osl.preprocessing.batch:main',
              'osl_maxfilter = osl.maxfilter.maxfilter:main',
              'osl_report = osl.report.raw_report:main',
              ]},
      install_requires=reqs + dev_reqs,
      package_data={'osl': ['utils/*tree', 'report/templates/*']},
      )
