from setuptools import setup, find_packages

setup(
    name='RL environments',
    version='0.1.0',
    packages=find_packages(include=['environments', 'environments.*']),
    install_requires=[
        'numpy~=1.22.3',
        'polars~=0.13.34',
        'jupyter'
    ]
)
