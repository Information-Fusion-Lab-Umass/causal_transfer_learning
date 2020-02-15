from setuptools import setup
from setuptools import find_packages

setup(name='mazelab', version='1.0.0',
     install_requires=['gym',
                        'numpy',
                        'matplotlib',
                        'scikit-image'],
      tests_require=['pytest'],
      python_requires='>=3',
      # List all packages (folder with __init__.py), useful to distribute a release
      packages=find_packages(),
      # tell pip some metadata (e.g. Python version, OS etc.)

)
