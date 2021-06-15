#!/usr/bin/env python

from growthbook import GrowthBook, Experiment


def test_default_weights():
    gb = GrowthBook(user={})
    exp = Experiment(
        key="my-test",
        variations=[0, 1]
    )
    expected = [1, 0, 0, 1, 1, 1, 0, 1, 0]

    for i, v in enumerate(expected):
        gb._user = {"id": str(i+1)+""}
        assert gb.run(exp).value == v

    gb.destroy()


def test_uneven_weights():
    gb = GrowthBook(user={})
    exp = Experiment(
        key="my-test",
        variations=[0, 1],
        weights=[0.1, 0.9]
    )
    expected = [1, 1, 0, 1, 1, 1, 0, 1, 1]

    for i, v in enumerate(expected):
        gb._user = {"id": str(i+1)+""}
        assert gb.run(exp).value == v


def test_coverage():
    gb = GrowthBook(user={})
    exp = Experiment(
        key="my-test",
        variations=[0, 1],
        coverage=0.4
    )
    expected = [-1, 0, 0, -1, -1, -1, 0, -1, 1]

    for i, v in enumerate(expected):
        gb._user = {"id": str(i+1)+""}
        res = gb.run(exp)
        actual = res.variationId if res.inExperiment else -1
        assert actual == v


def test_threeWayTest():
    gb = GrowthBook(user={})

    exp = Experiment(
        key="my-test",
        variations=[0, 1, 2],
    )

    expected = [2, 0, 0, 2, 1, 2, 0, 1, 0]

    for i, v in enumerate(expected):
        gb._user = {"id": str(i+1)+""}
        assert gb.run(exp).value == v


def test_testName():
    gb = GrowthBook(user={"id": '1'})

    assert gb.run(Experiment(key="my-test", variations=[0, 1])).value == 1
    assert gb.run(Experiment(key="my-test-3", variations=[0, 1])).value == 0

    gb.destroy()


def test_missing_id():
    gb = GrowthBook(user={"id": '1'})

    exp = Experiment(
        key="my-test",
        variations=[0, 1],
    )
    assert gb.run(exp).inExperiment is True
    gb._user = {"id": ''}
    assert gb.run(exp).inExperiment is False

    gb.destroy()


def getTrackingMock(gb: GrowthBook):
    calls = []
    def track(experiment, result): return calls.append([experiment, result])
    gb._trackingCallback = track
    return lambda: calls


def test_tracking():
    gb = GrowthBook(user={"id": '1'})

    getMockedCalls = getTrackingMock(gb)

    exp1 = Experiment(
        key="my-tracked-test",
        variations=[0, 1],
    )
    exp2 = Experiment(
        key="my-other-tracked-test",
        variations=[0, 1],
    )

    res1 = gb.run(exp1)
    gb.run(exp1)
    gb.run(exp1)
    res4 = gb.run(exp2)
    gb._user = {"id": '2'}
    res5 = gb.run(exp2)

    calls = getMockedCalls()
    assert len(calls) == 3
    assert calls[0] == [exp1, res1]
    assert calls[1] == [exp2, res4]
    assert calls[2] == [exp2, res5]

    gb.destroy()


def test_handles_weird_experiment_values():
    gb = GrowthBook(user={"id": '1'})

    assert gb.run(Experiment(
        key="my-test",
        variations=[0],
    )).inExperiment is False

    assert gb.run(Experiment(
        key="my-test",
        variations=[0, 1],
        include=lambda: 1/0,
    )).inExperiment is False

    exp = Experiment(
        key="my-test",
        variations=[0, 1],
        coverage=-0.2
    )
    assert exp.getWeights() == [0, 0]

    exp.coverage = 1.5
    assert exp.getWeights() == [0.5, 0.5]

    exp.coverage = 1
    exp.weights = [0.4, 0.1]
    assert exp.getWeights() == [0.5, 0.5]

    exp.weights = [0.7, 0.6]
    assert exp.getWeights() == [0.5, 0.5]

    exp.variations = [0, 1, 2, 3]
    exp.weights = [0.4, 0.4, 0.2]
    assert exp.getWeights() == [.25, .25, .25, .25]

    res1 = gb.run(Experiment(
        key="my-test",
        variations=[0, 1],
        force=-8,
    ))

    assert res1.inExperiment is False
    assert res1.value == 0

    res2 = gb.run(Experiment(
        key="my-test",
        variations=[0, 1],
        force=25,
    ))

    assert res2.inExperiment is False
    assert res2.value == 0

    # Should fail gracefully
    gb._trackingCallback = lambda experiment, result: 1/0
    assert gb.run(Experiment(key="my-test", variations=[0, 1])).value == 1

    gb.subscribe(lambda: 1/0)
    assert gb.run(Experiment(key="my-new-test", variations=[0, 1])).value == 0

    gb.destroy()


def test_force_variation():
    gb = GrowthBook(user={"id": '6'})
    exp = Experiment(key="forced-test", variations=[0, 1])
    assert gb.run(exp).value == 0

    getMockedCalls = getTrackingMock(gb)

    gb._overrides = {
        'forced-test': {
            "force": 1,
        },
    }
    assert gb.run(exp).value == 1

    calls = getMockedCalls()
    assert len(calls) == 0

    gb.destroy()


def test_uses_overrides():
    gb = GrowthBook(
        user={"id": '1'},
        overrides={
            'my-test': {
                'coverage': 0.01,
            },
        },
    )

    assert gb.run(Experiment(
        key="my-test",
        variations=[0, 1],
    )).inExperiment is False

    gb._overrides = {
        'my-test': {
            'url': r'^\\/path',
        },
    }

    assert gb.run(Experiment(
        key="my-test",
        variations=[0, 1],
    )).inExperiment is False

    gb.destroy()


def test_filters_user_groups():
    gb = GrowthBook(
        user={"id": '123'},
        groups={
            "alpha": True,
            "beta": True,
            "internal": False,
            "qa": False,
        },
    )

    assert gb.run(Experiment(
        key="my-test",
        variations=[0, 1],
        groups=['internal', 'qa'],
    )).inExperiment is False

    assert gb.run(Experiment(
        key="my-test",
        variations=[0, 1],
        groups=['internal', 'qa', 'beta'],
    )).inExperiment is True

    assert gb.run(Experiment(
        key="my-test",
        variations=[0, 1],
    )).inExperiment is True

    gb.destroy()


def test_runs_custom_include_callback():
    gb = GrowthBook(user={"id": '1'})
    assert gb.run(Experiment(
        key="my-test",
        variations=[0, 1],
        include=lambda: False
    )).inExperiment is False

    gb.destroy()


def test_supports_custom_user_hash_keys():
    gb = GrowthBook(user={
        "id": "1",
        "company": "abc"
    })

    exp = Experiment(
        key="my-test",
        variations=[0, 1],
        hashAttribute="company"
    )

    res = gb.run(exp)

    assert res.hashAttribute == "company"
    assert res.hashValue == "abc"

    gb.destroy()


def test_experiments_disabled():
    gb = GrowthBook(user={"id": '1'}, enabled=False)

    getMockedCalls = getTrackingMock(gb)

    assert gb.run(Experiment(key="disabled-test",
                             variations=[0, 1])).inExperiment is False

    calls = getMockedCalls()
    assert len(calls) == 0

    gb.destroy()


def test_querystring_force():
    gb = GrowthBook(user={"id": '1'})
    exp = Experiment(
        key="forced-test-qs",
        variations=[0, 1],
    )
    res1 = gb.run(exp)
    assert res1.value == 0
    assert res1.inExperiment is True

    gb._url = 'http://example.com?forced-test-qs=1#someanchor'

    res2 = gb.run(exp)
    assert res2.value == 1
    assert res2.inExperiment is False

    gb.destroy()


def test_querystring_force_disabled_tracking():
    gb = GrowthBook(
        user={"id": '1'},
        url='http://example.com?forced-test-qs=1',
    )
    getMockedCalls = getTrackingMock(gb)

    exp = Experiment(
        key="forced-test-qs",
        variations=[0, 1],
    )
    gb.run(exp)

    calls = getMockedCalls()
    assert len(calls) == 0


def test_querystring_force_invalid_url():
    gb = GrowthBook(
        user={},
        url=""
    )

    gb._url = ""
    assert gb._getQueryStringOverride('my-test') is None

    gb._url = 'http://example.com'
    assert gb._getQueryStringOverride('my-test') is None

    gb._url = 'http://example.com?'
    assert gb._getQueryStringOverride('my-test') is None

    gb._url = 'http://example.com?somequery'
    assert gb._getQueryStringOverride('my-test') is None

    gb._url = 'http://example.com??&&&?#'
    assert gb._getQueryStringOverride('my-test') is None


def test_url_targeting():
    gb = GrowthBook(
        user={"id": '1'},
        url='http://example.com',
    )

    exp = Experiment(
        key="my-test",
        variations=[0, 1],
        url='^\\/post\\/[0-9]+',
    )

    res = gb.run(exp)
    assert res.inExperiment is False
    assert res.value == 0

    gb._url = 'http://example.com/post/123'
    res = gb.run(exp)
    assert res.inExperiment is True
    assert res.value == 1

    exp.url = 'http:\\/\\/example.com\\/post\\/[0-9]+'
    res = gb.run(exp)
    assert res.inExperiment is True
    assert res.value == 1

    gb.destroy()


def test_invalid_url_regex():
    gb = GrowthBook(
        user={"id": '1'},
        overrides={
            'my-test': {
                'url': '???***[)',
            },
        },
        url='http://example.com',
    )

    assert gb.run(Experiment(
        key="my-test",
        variations=[0, 1],
    )).value == 1

    gb.destroy()


def test_ignores_draft_experiments():
    gb = GrowthBook(user={"id": '1'})
    exp = Experiment(
        key="my-test",
        status='draft',
        variations=[0, 1],
    )

    res1 = gb.run(exp)
    gb._url = 'http://example.com/?my-test=1'
    res2 = gb.run(exp)

    assert res1.inExperiment is False
    assert res1.value == 0
    assert res2.inExperiment is False
    assert res2.value == 1

    gb.destroy()


def test_ignores_stopped_experiments_unless_forced():
    gb = GrowthBook(user={"id": '1'})
    expLose = Experiment(
        key="my-test",
        status='stopped',
        variations=[0, 1, 2],
    )
    expWin = Experiment(
        key="my-test",
        status='stopped',
        variations=[0, 1, 2],
        force=2,
    )

    res1 = gb.run(expLose)
    res2 = gb.run(expWin)

    assert res1.value == 0
    assert res1.inExperiment is False
    assert res2.value == 2
    assert res2.inExperiment is False

    gb.destroy()


fired = {}


def flagSubscription(experiment, result):
    fired["value"] = True


def hasFired():
    return fired.get("value", False)


def resetFiredFlag():
    fired["value"] = False


def test_destroy_removes_subscriptions():
    gb = GrowthBook(user={"id": '1'})

    resetFiredFlag()
    gb.subscribe(flagSubscription)

    gb.run(Experiment(
        key="my-test",
        variations=[0, 1],
    ))

    assert hasFired() is True

    resetFiredFlag()
    gb.destroy()

    gb.run(Experiment(
        key="my-other-test",
        variations=[0, 1],
    ))

    assert hasFired() is False

    gb.destroy()


def test_configData_experiment():
    gb = GrowthBook(user={"id": '1'})
    exp = Experiment(
        key="my-test",
        variations=[
            {
                "color": 'blue',
                "size": 'small',
            },
            {
                "color": 'green',
                "size": 'large',
            },
        ],
    )

    res1 = gb.run(exp)
    assert res1.variationId == 1
    assert res1.value == {
        "color": 'green',
        "size": 'large',
    }

    # Fallback to control config data if not in test
    exp.coverage = 0.01
    res2 = gb.run(exp)
    assert res2.inExperiment is False
    assert res2.variationId == 0
    assert res2.value == {
        "color": 'blue',
        "size": 'small',
    }

    gb.destroy()


def test_does_even_weighting():
    gb = GrowthBook(user={})
    # Full coverage
    exp = Experiment(key="my-test", variations=[0, 1])
    variations = {
        '0': 0,
        '1': 0,
        '-1': 0,
    }
    for i in range(1000):
        gb._user = {"id": str(i) + ''}
        res = gb.run(exp)
        v = res.value if res.inExperiment else -1
        variations[str(v)] += 1

    assert variations['0'] == 503

    # Reduced coverage
    exp.coverage = 0.4
    variations = {
        '0': 0,
        '1': 0,
        '-1': 0,
    }
    for i in range(1000):
        gb._user = {"id": str(i) + ''}
        res = gb.run(exp)
        v = res.value if res.inExperiment else -1
        variations[str(v)] += 1
    assert variations['0'] == 200
    assert variations['1'] == 204
    assert variations['-1'] == 596

    # 3-way
    exp.coverage = 0.6
    exp.variations = [0, 1, 2]
    variations = {
        '0': 0,
        '1': 0,
        '2': 0,
        '-1': 0,
    }
    for i in range(10000):
        gb._user = {"id": str(i) + ''}
        res = gb.run(exp)
        v = res.value if res.inExperiment else -1
        variations[str(v)] += 1
    assert variations['-1'] == 3973
    assert variations['0'] == 2044
    assert variations['1'] == 1992
    assert variations['2'] == 1991

    gb.destroy()


def test_forces_variations_from_the_client():
    gb = GrowthBook(user={"id": '1'})
    exp = Experiment(
        key="my-test",
        variations=[0, 1],
    )
    res1 = gb.run(exp)
    assert res1.inExperiment is True
    assert res1.value == 1

    gb._forcedVariations = {'my-test': 0}
    res2 = gb.run(exp)
    assert res2.inExperiment is False
    assert res2.value == 0

    gb.destroy()


def test_qa_mode():
    gb = GrowthBook(user={"id": '1'}, qaMode=True)
    exp = Experiment(
        key="my-test",
        variations=[0, 1],
    )

    res1 = gb.run(exp)
    assert res1.inExperiment is False
    assert res1.value == 0

    # Still works if explicitly forced
    gb._forcedVariations = {'my-test': 1}
    res2 = gb.run(exp)
    assert res2.inExperiment is False
    assert res2.value == 1

    # Works if the experiment itself is forced
    res3 = gb.run(Experiment(
        key="my-test-2",
        variations=[0, 1],
        force=1,
    ))

    assert res3.inExperiment is False
    assert res3.value == 1

    gb.destroy()


def test_fires_subscriptions_correctly():
    gb = GrowthBook(
        user={
            "id": '1',
        },
    )

    resetFiredFlag()
    unsubscriber = gb.subscribe(flagSubscription)

    assert hasFired() is False

    exp = Experiment(
        key="my-test",
        variations=[0, 1],
    )

    # Should fire when user is put in an experiment
    gb.run(exp)
    assert hasFired() is True

    # Does not fire if nothing has changed
    resetFiredFlag()
    gb.run(exp)
    assert hasFired() is False

    # Does not fire after unsubscribed
    unsubscriber()
    gb.run(Experiment(
        key="other-test",
        variations=[0, 1],
    ))

    assert hasFired() is False

    gb.destroy()


def test_stores_assigned_variations_in_the_user():
    gb = GrowthBook(
        user={
            "id": '1',
        },
    )

    gb.run(Experiment(key="my-test", variations=[0, 1]))
    gb.run(Experiment(key="my-test-3", variations=[0, 1]))

    assigned = gb.getAllResults()
    assignedArr = []

    for e in assigned:
        assignedArr.append({
            "key": e,
            "variation": assigned[e]["result"].variationId
        })

    assert len(assignedArr) == 2
    assert assignedArr[0]["key"] == 'my-test'
    assert assignedArr[0]["variation"] == 1
    assert assignedArr[1]["key"] == 'my-test-3'
    assert assignedArr[1]["variation"] == 0

    gb.destroy()
