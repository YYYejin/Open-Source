
from setuptools import setup, find_packages

setup(
    name='K_fold',
    version='0.1',
    packages=find_packages(),
    license='MIT',
    description='A Python implementation of K-Fold Cross-Validation',
    author='Kim Yejin',
    author_email='vvvsfhk@naver.com',
    url='https://github.com/YYYejin/Open-Source.git',
    install_requires=[
        'scikit-learn'
    ],
)