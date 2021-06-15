#!/usr/bin/env python

from setuptools import setup

requirements = []

test_requirements = ['pytest>=3', ]

setup(
    name='growthbook',
    version='0.1.0',
    author="Growth Book",
    author_email='hello@growthbook.io',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Powerful A/B testing for Python apps",
    install_requires=requirements,
    license="MIT",
    include_package_data=True,
    keywords='growthbook',
    py_modules=['growthbook'],
    scripts=['growthbook.py'],
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/growthbook/growthbook-python',
)
