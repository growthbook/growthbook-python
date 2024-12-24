#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

requirements = ['cryptography', 'typing_extensions', 'urllib3', ]

test_requirements = ['pytest>=3', ]

setup(
    name='growthbook',
    version='1.2.0',
    author="GrowthBook",
    author_email='hello@growthbook.io',
    python_requires='>=3.6',
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
