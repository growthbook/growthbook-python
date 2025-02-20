#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

requirements = [
    'cryptography', 
    'typing_extensions', 
    'urllib3',
    'dataclasses;python_version<"3.7"',  # Add dataclasses backport for Python 3.6
    'async-generator;python_version<"3.7"',  # For asynccontextmanager in Python 3.6
    'aiohttp>=3.6.0',  # For async HTTP support
    'importlib-metadata;python_version<"3.8"',  # For metadata in Python 3.6-3.7
]

test_requirements = [
    'pytest>=3',
    'pytest-asyncio>=0.10.0',
    'mock;python_version<"3.8"',  # Only install mock for Python < 3.8
]

setup(
    name='growthbook',
    author="GrowthBook",
    author_email='hello@growthbook.io',
    python_requires='>=3.6',
    setup_requires=['setuptools_scm'],
    use_scm_version={
        'write_to': 'growthbook/_version.py',
        'write_to_template': '__version__ = "{version}"',
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    description="Powerful Feature flagging and A/B testing for Python apps",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    license="MIT",
    include_package_data=True,
    packages=find_packages(include=['growthbook', 'growthbook.*']),
    package_data={"growthbook": ["py.typed"]},
    keywords='growthbook',
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/growthbook/growthbook-python',
)
