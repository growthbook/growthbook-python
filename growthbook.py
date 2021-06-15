import re


def fnv1a32(str):
    hval = 0x811c9dc5
    prime = 0x01000193
    uint32_max = 2 ** 32
    for s in str:
        hval = hval ^ ord(s)
        hval = (hval * prime) % uint32_max
    return hval


class Experiment(object):
    def __init__(
        self,
        key: str,
        variations: list,
        weights: list = None,
        status: str = "running",
        coverage: int = 1,
        url: str = "",
        include=None,
        groups: list = None,
        force: int = None,
        hashAttribute: str = "id"
    ) -> None:
        self.key = key
        self.variations = variations
        self.weights = weights
        self.status = status
        self.coverage = coverage
        self.url = url
        self.include = include
        self.groups = groups
        self.force = force
        self.hashAttribute = hashAttribute

    def _equalWeights(self):
        return [1/len(self.variations)]*len(self.variations)

    def getWeights(self):
        weights = self.weights or self._equalWeights()

        if len(weights) != len(self.variations):
            weights = self._equalWeights()

        s = sum(weights)
        if s < 0.98 or s > 1.02:
            weights = self._equalWeights()

        if self.coverage is None:
            return weights

        coverage = self.coverage
        if coverage < 0:
            coverage = 0
        if coverage > 1:
            coverage = 1

        return list(map(lambda x: x*coverage, weights))

    def update(self, data: dict):
        weights = data.get("weights", None)
        status = data.get("status", None)
        coverage = data.get("coverage", None)
        url = data.get("url", None)
        groups = data.get("groups", None)
        force = data.get("force", None)

        if weights is not None:
            self.weights = weights
        if status is not None:
            self.status = status
        if coverage is not None:
            self.coverage = coverage
        if url is not None:
            self.url = url
        if groups is not None:
            self.groups = groups
        if force is not None:
            self.force = force


class Result(object):
    def __init__(
        self,
        variationId,
        inExperiment,
        value,
        hashAttribute,
        hashValue
    ) -> None:
        self.variationId = variationId
        self.inExperiment = inExperiment
        self.value = value
        self.hashAttribute = hashAttribute
        self.hashValue = hashValue


class GrowthBook(object):
    def __init__(
        self,
        user,
        trackingCallback=None,
        enabled=True,
        url="",
        groups={},
        overrides={},
        forcedVariations={},
        qaMode=False,
    ):
        self._user = user
        self._trackingCallback = trackingCallback
        self._enabled = enabled
        self._url = url
        self._groups = groups
        self._overrides = overrides
        self._forcedVariations = forcedVariations
        self._qaMode = qaMode

        self._tracked = {}
        self._assigned = {}
        self._subscriptions = set()

    def destroy(self):
        self._subscriptions.clear()
        self._tracked.clear()
        self._assigned.clear()
        self._trackingCallback = None
        self._forcedVariations.clear()
        self._overrides.clear()
        self._groups.clear()

    def getAllResults(self):
        return self._assigned.copy()

    def run(self, experiment: Experiment) -> Result:
        result = self._run(experiment)

        prev = self._assigned.get(experiment.key, None)
        if (not prev or prev["result"].inExperiment != result.inExperiment
                or prev["result"].variationId != result.variationId):
            self._assigned[experiment.key] = {
                "experiment": experiment,
                "result": result
            }
            for cb in self._subscriptions:
                try:
                    cb(experiment, result)
                except Exception:
                    pass

        return result

    def subscribe(self, callback):
        self._subscriptions.add(callback)
        return lambda: self._subscriptions.remove(callback)

    def _run(self, experiment: Experiment) -> Result:
        # 1. If experiment is invalid, return immediately
        if len(experiment.variations) < 2:
            return self._getResult(experiment=experiment)
        # 2. If growthbook is disabled, return immediately
        if not self._enabled:
            return self._getResult(experiment=experiment)
        # 3. If the experiment props have been overridden, merge them in
        if self._overrides.get(experiment.key, None):
            experiment.update(self._overrides[experiment.key])
        # 4. If experiment is forced via a querystring in the url
        qs = self._getQueryStringOverride(experiment.key)
        if qs is not None:
            return self._getResult(experiment=experiment, variationId=qs)
        # 5. If variation is forced in the context
        if self._forcedVariations.get(experiment.key, None) is not None:
            return self._getResult(
                experiment=experiment,
                variationId=self._forcedVariations[experiment.key]
            )
        # 6. If experiment is a draft, return immediately
        if experiment.status == "draft":
            return self._getResult(experiment=experiment)
        # 7. Get the user hash attribute and value
        hashAttribute = experiment.hashAttribute or "id"
        hashValue = self._user.get(hashAttribute, None)
        if not hashValue:
            return self._getResult(experiment=experiment)
        # 8. If experiment has an include property
        if experiment.include:
            try:
                if not experiment.include():
                    return self._getResult(experiment=experiment)
            except Exception:
                return self._getResult(experiment=experiment)
        # 9. If experiment.groups is set, make sure user is in a matching group
        if experiment.groups and len(experiment.groups):
            expGroups = self._groups or {}
            matched = False
            for group in experiment.groups:
                if expGroups[group]:
                    matched = True
            if not matched:
                return self._getResult(experiment=experiment)
        # 10. If experiment.url is set, see if it's valid
        if experiment.url:
            if not self._urlIsValid(experiment.url):
                return self._getResult(experiment=experiment)
        # 11. If experiment is forced, return immediately
        if experiment.force is not None:
            return self._getResult(
                experiment=experiment,
                variationId=experiment.force
            )
        # 12. If experiment is stopped, return immediately
        if experiment.status == "stopped":
            return self._getResult(experiment=experiment)
        # 13. If in qa mode, return immediately
        if self._qaMode:
            return self._getResult(experiment=experiment)
        # 14. Compute a hash for variation assignment
        n = (fnv1a32(hashValue + experiment.key) % 1000)/1000
        # 15. Get variation weights
        weights = experiment.getWeights()
        # 16. Loop through weights until we reach the hash value
        cumulative = 0
        assigned = -1
        for i, weight in enumerate(weights):
            cumulative += weight
            if n < cumulative and assigned < 0:
                assigned = i
        # 17. If not assigned, return immediately
        if assigned < 0:
            return self._getResult(experiment=experiment)
        result = self._getResult(
            experiment=experiment,
            variationId=assigned,
            inExperiment=True
        )
        # 18. Fire the tracking callback if set
        self._track(experiment, result)
        # 19. Return the result
        return result

    def _track(self, experiment: Experiment, result: Result):
        if not self._trackingCallback:
            return None
        key = result.hashAttribute + \
            str(result.hashValue) + experiment.key + str(result.variationId)
        if not self._tracked.get(key):
            try:
                self._trackingCallback(
                    experiment=experiment,
                    result=result
                )
                self._tracked[key] = True
            except Exception:
                pass

    def _urlIsValid(self, pattern):
        if not self._url:
            return False

        try:
            r = re.compile(pattern)
            if r.search(self._url):
                return True

            pathOnly = re.sub(
                r'^[^/]*/', '/', re.sub(r'^https?:\/\/', '', self._url))
            if r.search(pathOnly):
                return True
            return False
        except Exception:
            return True

    def _getQueryStringOverride(self, key):
        if not self._url:
            return None

        # Look for the experiment key in the querystring
        try:
            match = re.search("[?&]"+key+"=([0-9]+)($|&|#)", self._url)
            # If a match is found, return the variation, otherwise return None
            if not match:
                return None
            return int(match.group(1))
        except Exception:
            return None

    def _getResult(
        self,
        experiment,
        variationId=0,
        inExperiment=False
    ) -> Result:
        hashAttribute = experiment.hashAttribute or "id"

        if variationId < 0 or variationId > len(experiment.variations)-1:
            variationId = 0

        return Result(
            inExperiment=inExperiment,
            variationId=variationId,
            value=experiment.variations[variationId],
            hashAttribute=hashAttribute,
            hashValue=self._user.get(hashAttribute)
        )
