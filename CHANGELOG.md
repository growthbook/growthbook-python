# Changelog

## [1.3.0](https://github.com/growthbook/growthbook-python/compare/v1.2.0...v1.3.0) (2025-06-11)


### Features

* tracking plugins ([fde4d42](https://github.com/growthbook/growthbook-python/commit/fde4d4283343758ca1ec034052b8bdb2c0639b22))

## [1.2.1](https://github.com/growthbook/growthbook-python/compare/v1.1.0...v) (2024-XX-XX)

### Bug Fixes

* PEP-561 compatibility
* Fix zero value evaluation for _getOrigHashValue
* Saved groups, $inGroup $notInGroup operators, versionCompare
* Added SSE client
* Support for multi-context and Enhanced GrowthBookClient with Async processing
* Update to test spec 0.7.1

### Features

* Enhanced async client with better error handling and retry logic
* Added comprehensive test coverage for async functionality

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

## **1.1.0** - Apr 11, 2024

- Support for prerequisite feature flags
- Optional Sticky Bucketing for experiment variation assignments
- SemVer targeting support
- Fixed multiple bugs and edge cases when comparing different data types
- Fixed bugs with the $in and $nin operators
- Now, we ignore unknown fields in feature definitions instead of throwing Exceptions
- Support for feature rule ids (for easier debugging)

## **1.0.0** - Apr 23, 2023

- Update to the official 0.4.1 GrowthBook SDK spec version
- Built-in fetching and caching of feature flags from the GrowthBook API
- Added detailed logging for easier debugging
- Support for new feature/experiment properties that enable holdout groups, meta info, better hashing algorithms, and more

## **0.3.1** - Aug 1, 2022

- Bug fix - skip experiment when the hashAttribute's value is `None`

## **0.3.0** - May 24, 2022

- Bug fix - don't skip feature rules when experiment variation is forced

## **0.2.0** - Feb 13, 2022

- Support for Feature Flags

## **0.1.1** - Jun 15, 2021

- Initial release (inline experiments only)
