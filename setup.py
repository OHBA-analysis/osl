from setuptools import setup

setup(name='osl',
      version='0.0.1.dev',
      description='OHBA Software Library',
      author='OHBA Analysis Group',
      license='MIT',
      packages=['osl'],
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'osl_batch = osl.preprocessing:main',
              ],
      },)
