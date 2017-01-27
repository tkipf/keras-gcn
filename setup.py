from setuptools import setup
from setuptools import find_packages

setup(name='kegra',
      version='0.0.1',
      description='Deep Learning on Graphs with Keras',
      author='Thomas Kipf',
      author_email='thomas.kipf@gmail.com',
      url='https://tkipf.github.io',
      download_url='...',
      license='MIT',
      install_requires=['keras'],
      extras_require={
          'model_saving': ['json', 'h5py'],
      },
      package_data={'kegra': ['README.md']},
      packages=find_packages())