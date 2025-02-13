import inspect
import logging
from typing import Callable, List, Optional, Union, Awaitable

from growthbook.common_types import (
    AsyncEvaluationContext,
    Experiment,
    Result,
    FeatureResult,
)
from growthbook.core import (
    evalCondition,
    _isIncludedInRollout,
    _isFilteredOut,
    _getExperimentResult,
    getQueryStringOverride,
    _getHashValue,
    _get_sticky_bucket_variation,
    inNamespace,
    _urlIsValid,
    gbhash,
    getBucketRanges,
    chooseVariation,
    _get_sticky_bucket_experiment_key,
    _generate_sticky_bucket_assignment_doc,
)

logger = logging.getLogger("growthbook.core")


async def eval_prereqs_async(
    parentConditions: List[dict], evalContext: AsyncEvaluationContext
) -> str:
    for parentCondition in parentConditions:
        parentRes = await eval_feature_async(
            key=parentCondition.get("id", None), evalContext=evalContext
        )

        if parentRes.source == "cyclicPrerequisite":
            return "cyclic"

        if not evalCondition(
            {"value": parentRes.value},
            parentCondition.get("condition", None),
            evalContext.global_ctx.saved_groups,
        ):
            if parentCondition.get("gate", False):
                return "gate"
            return "fail"
    return "pass"


async def eval_feature_async(
    key: str,
    evalContext: AsyncEvaluationContext = None,
    callback_subscription: Callable[
        [Experiment, Result], Union[Awaitable[None] | None]
    ] = None,
):
    """Core feature evaluation logic as a standalone function"""

    if evalContext is None:
        raise ValueError("evalContext is required - eval_feature")

    if key not in evalContext.global_ctx.features:
        logger.warning("Unknown feature %s", key)
        return FeatureResult(None, "unknownFeature")

    if key in evalContext.stack.evaluted_features:
        logger.warning(
            "Cyclic prerequisite detected, stack: %s",
            evalContext.stack.evaluted_features,
        )
        return FeatureResult(None, "cyclicPrerequisite")

    evalContext.stack.evaluted_features.add(key)

    feature = evalContext.global_ctx.features[key]

    for rule in feature.rules:
        if rule.parentConditions:
            prereq_res = await eval_prereqs_async(
                parentConditions=rule.parentConditions, evalContext=evalContext
            )
            if prereq_res == "gate":
                logger.debug(
                    "Top-level prerequisite failed, return None, feature %s", key
                )
                return FeatureResult(None, "prerequisite")
            if prereq_res == "cyclic":
                # Warning already logged in this case
                return FeatureResult(None, "cyclicPrerequisite")
            if prereq_res == "fail":
                logger.debug(
                    "Skip rule because of failing prerequisite, feature %s", key
                )
                continue

        if rule.condition:
            if not evalCondition(
                evalContext.user.attributes,
                rule.condition,
                evalContext.global_ctx.saved_groups,
            ):
                logger.debug("Skip rule because of failed condition, feature %s", key)
                continue
        if rule.filters:
            if _isFilteredOut(rule.filters, evalContext):
                logger.debug("Skip rule because of filters/namespaces, feature %s", key)
                continue
        if rule.force is not None:
            if not _isIncludedInRollout(
                seed=rule.seed or key,
                hashAttribute=rule.hashAttribute,
                fallbackAttribute=rule.fallbackAttribute,
                range=rule.range,
                coverage=rule.coverage,
                hashVersion=rule.hashVersion,
                eval_context=evalContext,
            ):
                logger.debug(
                    "Skip rule because user not included in percentage rollout, feature %s",
                    key,
                )
                continue

            logger.debug("Force value from rule, feature %s", key)
            return FeatureResult(rule.force, "force", ruleId=rule.id)

        if rule.variations is None:
            logger.warning("Skip invalid rule, feature %s", key)
            continue

        exp = Experiment(
            key=rule.key or key,
            variations=rule.variations,
            coverage=rule.coverage,
            weights=rule.weights,
            hashAttribute=rule.hashAttribute,
            fallbackAttribute=rule.fallbackAttribute,
            namespace=rule.namespace,
            hashVersion=rule.hashVersion,
            meta=rule.meta,
            ranges=rule.ranges,
            name=rule.name,
            phase=rule.phase,
            seed=rule.seed,
            filters=rule.filters,
            condition=rule.condition,
            disableStickyBucketing=rule.disableStickyBucketing,
            bucketVersion=rule.bucketVersion,
            minBucketVersion=rule.minBucketVersion,
        )

        result = await run_experiment_async(
            experiment=exp, featureId=key, evalContext=evalContext
        )

        if callback_subscription:
            res = callback_subscription(exp, result)
            if inspect.isawaitable(res):
                await res

        if not result.inExperiment:
            logger.debug(
                "Skip rule because user not included in experiment, feature %s", key
            )
            continue

        if result.passthrough:
            logger.debug("Continue to next rule, feature %s", key)
            continue

        logger.debug("Assign value from experiment, feature %s", key)
        return FeatureResult(result.value, "experiment", exp, result, ruleId=rule.id)

    logger.debug("Use default value for feature %s", key)
    return FeatureResult(feature.defaultValue, "defaultValue")


async def run_experiment_async(
    experiment: Experiment,
    featureId: Optional[str] = None,
    evalContext: AsyncEvaluationContext = None,
    tracking_cb: Callable[[Experiment, Result], Union[None, Awaitable[None]]] = None,
) -> Result:
    if evalContext is None:
        raise ValueError("evalContext is required - run_experiment")
    # 1. If experiment has less than 2 variations, return immediately
    if len(experiment.variations) < 2:
        logger.warning("Experiment %s has less than 2 variations, skip", experiment.key)
        return _getExperimentResult(
            experiment=experiment, featureId=featureId, evalContext=evalContext
        )
    # 2. If growthbook is disabled, return immediately
    if not evalContext.global_ctx.options.enabled:
        logger.debug(
            "Skip experiment %s because GrowthBook is disabled", experiment.key
        )
        return _getExperimentResult(
            experiment=experiment, featureId=featureId, evalContext=evalContext
        )
    # 2.5. If the experiment props have been overridden, merge them in
    if evalContext.user.overrides.get(experiment.key, None):
        experiment.update(evalContext.user.overrides[experiment.key])
    # 3. If experiment is forced via a querystring in the url
    qs = getQueryStringOverride(
        experiment.key, evalContext.user.url, len(experiment.variations)
    )
    if qs is not None:
        logger.debug(
            "Force variation %d from URL querystring, experiment %s",
            qs,
            experiment.key,
        )
        return _getExperimentResult(
            experiment=experiment,
            variationId=qs,
            featureId=featureId,
            evalContext=evalContext,
        )
    # 4. If variation is forced in the context
    if evalContext.user.forced_variations.get(experiment.key, None) is not None:
        logger.debug(
            "Force variation %d from GrowthBook context, experiment %s",
            evalContext.user.forced_variations[experiment.key],
            experiment.key,
        )
        return _getExperimentResult(
            experiment=experiment,
            variationId=evalContext.user.forced_variations[experiment.key],
            featureId=featureId,
            evalContext=evalContext,
        )
    # 5. If experiment is a draft or not active, return immediately
    if experiment.status == "draft" or not experiment.active:
        logger.debug("Experiment %s is not active, skip", experiment.key)
        return _getExperimentResult(
            experiment=experiment, featureId=featureId, evalContext=evalContext
        )

    # 6. Get the user hash attribute and value
    (hashAttribute, hashValue) = _getHashValue(
        attr=experiment.hashAttribute,
        fallbackAttr=experiment.fallbackAttribute,
        eval_context=evalContext,
    )
    if not hashValue:
        logger.debug(
            "Skip experiment %s because user's hashAttribute value is empty",
            experiment.key,
        )
        return _getExperimentResult(
            experiment=experiment, featureId=featureId, evalContext=evalContext
        )

    assigned = -1

    found_sticky_bucket = False
    sticky_bucket_version_is_blocked = False
    if (
        evalContext.global_ctx.sticky_bucket_service
        and not experiment.disableStickyBucketing
    ):
        sticky_bucket = _get_sticky_bucket_variation(
            experiment_key=experiment.key,
            bucket_version=experiment.bucketVersion,
            min_bucket_version=experiment.minBucketVersion,
            meta=experiment.meta,
            hash_attribute=experiment.hashAttribute,
            fallback_attribute=experiment.fallbackAttribute,
            evalContext=evalContext,
        )
        found_sticky_bucket = sticky_bucket.get("variation", 0) >= 0
        assigned = sticky_bucket.get("variation", 0)
        sticky_bucket_version_is_blocked = sticky_bucket.get("versionIsBlocked", False)

    if found_sticky_bucket:
        logger.debug(
            "Found sticky bucket for experiment %s, assigning sticky variation %s",
            experiment.key,
            assigned,
        )

    # Some checks are not needed if we already have a sticky bucket
    if not found_sticky_bucket:
        # 7. Filtered out / not in namespace
        if experiment.filters:
            if _isFilteredOut(experiment.filters, evalContext):
                logger.debug(
                    "Skip experiment %s because of filters/namespaces", experiment.key
                )
                return _getExperimentResult(
                    experiment=experiment, featureId=featureId, evalContext=evalContext
                )
        elif experiment.namespace and not inNamespace(hashValue, experiment.namespace):
            logger.debug("Skip experiment %s because of namespace", experiment.key)
            return _getExperimentResult(
                experiment=experiment, featureId=featureId, evalContext=evalContext
            )

        # 7.5. If experiment has an include property
        if experiment.include:
            try:
                if not experiment.include():
                    logger.debug(
                        "Skip experiment %s because include() returned false",
                        experiment.key,
                    )
                    return _getExperimentResult(
                        experiment=experiment,
                        featureId=featureId,
                        evalContext=evalContext,
                    )
            except Exception:
                logger.warning(
                    "Skip experiment %s because include() raised an Exception",
                    experiment.key,
                )
                return _getExperimentResult(
                    experiment=experiment, featureId=featureId, evalContext=evalContext
                )

        # 8. Exclude if condition is false
        if experiment.condition and not evalCondition(
            evalContext.user.attributes,
            experiment.condition,
            evalContext.global_ctx.saved_groups,
        ):
            logger.debug(
                "Skip experiment %s because user failed the condition", experiment.key
            )
            return _getExperimentResult(
                experiment=experiment, featureId=featureId, evalContext=evalContext
            )

        # 8.05 Exclude if parent conditions are not met
        if experiment.parentConditions:
            prereq_res = await eval_prereqs_async(
                parentConditions=experiment.parentConditions, evalContext=evalContext
            )
            if prereq_res == "gate" or prereq_res == "fail":
                logger.debug(
                    "Skip experiment %s because of failing prerequisite", experiment.key
                )
                return _getExperimentResult(
                    experiment=experiment, featureId=featureId, evalContext=evalContext
                )
            if prereq_res == "cyclic":
                logger.debug(
                    "Skip experiment %s because of cyclic prerequisite", experiment.key
                )
                return _getExperimentResult(
                    experiment=experiment, featureId=featureId, evalContext=evalContext
                )

        # 8.1. Make sure user is in a matching group
        if experiment.groups and len(experiment.groups):
            expGroups = evalContext.user.groups or {}
            matched = False
            for group in experiment.groups:
                if expGroups[group]:
                    matched = True
            if not matched:
                logger.debug(
                    "Skip experiment %s because user not in required group",
                    experiment.key,
                )
                return _getExperimentResult(
                    experiment=experiment, featureId=featureId, evalContext=evalContext
                )

    # The following apply even when in a sticky bucket

    # 8.2. If experiment.url is set, see if it's valid
    if experiment.url:
        if not _urlIsValid(
            url=evalContext.global_ctx.options.url, pattern=experiment.url
        ):
            logger.debug(
                "Skip experiment %s because current URL is not targeted",
                experiment.key,
            )
            return _getExperimentResult(
                experiment=experiment, featureId=featureId, evalContext=evalContext
            )

    # 9. Get bucket ranges and choose variation
    n = gbhash(
        experiment.seed or experiment.key, hashValue, experiment.hashVersion or 1
    )
    if n is None:
        logger.warning(
            "Skip experiment %s because of invalid hashVersion", experiment.key
        )
        return _getExperimentResult(
            experiment=experiment, featureId=featureId, evalContext=evalContext
        )

    if not found_sticky_bucket:
        c = experiment.coverage
        ranges = experiment.ranges or getBucketRanges(
            len(experiment.variations), c if c is not None else 1, experiment.weights
        )
        assigned = chooseVariation(n, ranges)

    # Unenroll if any prior sticky buckets are blocked by version
    if sticky_bucket_version_is_blocked:
        logger.debug(
            "Skip experiment %s because sticky bucket version is blocked",
            experiment.key,
        )
        return _getExperimentResult(
            experiment=experiment,
            featureId=featureId,
            stickyBucketUsed=True,
            evalContext=evalContext,
        )

    # 10. Return if not in experiment
    if assigned < 0:
        logger.debug(
            "Skip experiment %s because user is not included in the rollout",
            experiment.key,
        )
        return _getExperimentResult(
            experiment=experiment, featureId=featureId, evalContext=evalContext
        )

    # 11. If experiment is forced, return immediately
    if experiment.force is not None:
        logger.debug(
            "Force variation %d in experiment %s", experiment.force, experiment.key
        )
        return _getExperimentResult(
            experiment=experiment,
            variationId=experiment.force,
            featureId=featureId,
            evalContext=evalContext,
        )

    # 12. Exclude if in QA mode
    if evalContext.global_ctx.options.qa_mode:
        logger.debug("Skip experiment %s because of QA Mode", experiment.key)
        return _getExperimentResult(
            experiment=experiment, featureId=featureId, evalContext=evalContext
        )

    # 12.5. If experiment is stopped, return immediately
    if experiment.status == "stopped":
        logger.debug("Skip experiment %s because it is stopped", experiment.key)
        return _getExperimentResult(
            experiment=experiment, featureId=featureId, evalContext=evalContext
        )

    # 13. Build the result object
    result = _getExperimentResult(
        experiment=experiment,
        variationId=assigned,
        hashUsed=True,
        featureId=featureId,
        bucket=n,
        stickyBucketUsed=found_sticky_bucket,
        evalContext=evalContext,
    )

    # 13.5 Persist sticky bucket
    if (
        evalContext.global_ctx.sticky_bucket_service
        and not experiment.disableStickyBucketing
    ):
        assignment = {}
        assignment[
            _get_sticky_bucket_experiment_key(experiment.key, experiment.bucketVersion)
        ] = result.key

        data = _generate_sticky_bucket_assignment_doc(
            attribute_name=hashAttribute,
            attribute_value=hashValue,
            assignments=assignment,
            evalContext=evalContext,
        )
        doc = data.get("doc", None)
        if doc and data.get("changed", False):
            if not evalContext.user.sticky_bucket_assignment_docs:
                evalContext.user.sticky_bucket_assignment_docs = {}
            evalContext.user.sticky_bucket_assignment_docs[data.get("key")] = doc
            await evalContext.global_ctx.sticky_bucket_service.save_assignments(doc)

    # 14. Fire the tracking callback if set
    if tracking_cb:
        res = tracking_cb(experiment, result)
        if inspect.isawaitable(res):
            await res

    # 15. Return the result
    logger.debug("Assigned variation %d in experiment %s", assigned, experiment.key)
    return result
