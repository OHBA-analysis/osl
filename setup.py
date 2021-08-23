from setuptools import setup

name = 'osl'

setup(name=name,
      version='0.0.1.dev',
      description='OHBA Software Library',
      author='OHBA Analysis Group',
      license='MIT',
      packages=['osl', 'osl.report', 'osl.maxfilter',
                'osl.preprocessing', 'osl.utils'],
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'osl_batch = osl.preprocessing.batch:main',
              'osl_maxfilter = osl.maxfilter.maxfilter:main',
              'osl_report = osl.report.raw_report:main',
              ]},
      install_requires=['mne', 'sklearn', 'fslpy', 'sails', 'tabulate', 'PyYAML', 'pydata-sphinx-theme'],
      package_data={'osl': ['utils/*tree',  'utils/reduced_mvar_*', 'utils/megin*fif']},

      command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', name),
            'release': ('setup.py', name)}},

      )
