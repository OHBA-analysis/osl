from setuptools import setup

setup(name='osl',
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
              ],
      install_requires = ['fslpy', 'sails', 'tabulate'],
      },)
