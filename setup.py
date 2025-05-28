#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path
import re

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

requirements = [
    'cryptography', 
    'typing_extensions', 
    'urllib3',
    'aiohttp>=3.6.0',  # For async HTTP support
]

test_requirements = [
    'pytest>=3',
    'pytest-asyncio>=0.10.0',
]

def get_version():
    with open('growthbook/__init__.py', 'r') as f:
        content = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name='growthbook',
    author="GrowthBook",
    author_email='hello@growthbook.io',
    python_requires='>=3.7',
    version=get_version(),  # Read version from __init__.py
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
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
