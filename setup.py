import os
from setuptools import setup, find_packages


def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()

setup(
    name='nig',
    version='0.1dev',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    description='Nature\'s Imitation Game',
    long_description=read('README.md'),
    url='https://github.com/eaplatanios/nig',
    license='MIT',
    install_requires=['cython', 'enum34', 'numpy>=1.5', 'pandas', 'six',
                      'tensorflow>=0.9', 'jnius==1.1-dev'],
    package_data={
        'nig': ['evaluation/*.jar'],
    }
)
