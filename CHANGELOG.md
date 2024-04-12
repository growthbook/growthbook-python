# Changelog

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
