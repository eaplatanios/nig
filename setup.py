import os
from setuptools import setup, find_packages


def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()

setup(
    name='nig',
    version='0.1dev',
    license='Apache License 2.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    description='Nature\'s Imitation Game',
    long_description=read('README.md'),
    url='https://github.com/eaplatanios/nig',
    install_requires=['cython', 'numpy', 'pandas', 'pyyaml', 'six',
                      'tensorflow', 'jnius==1.1-dev'],
    package_data={
        'nig': ['evaluation/*.jar', 'logging.yaml'],
    }
)
