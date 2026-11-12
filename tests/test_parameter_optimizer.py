import sys
from types import SimpleNamespace

import pytest

from greenrouting.config import Settings
from greenrouting.parameter_optimizer import run_parameter_optimization


class FakeTrial:
    def __init__(self, values):
        self.values = values
        self.params = {}
        self.user_attrs = {}

    def suggest_float(self, name, low, high):
        value = self.values[name]
        assert low <= value <= high
        self.params[name] = value
        return value

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


class FakeStudy:
    def __init__(self, trials):
        self.trials = trials
        self.values = []

    def optimize(self, objective, n_trials):
        assert n_trials == len(self.trials)
        for number, trial in enumerate(self.trials):
            trial.number = number
        self.values = [objective(trial) for trial in self.trials]


class FakeBenchmarkRunner:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def run_benchmark(self, queries, modes):
        assert queries == ["question"]
        assert modes == ["GREENROUTING"]
        quality = self.orchestrator.settings.routing.complexity_threshold
        energy = 1.0 - self.orchestrator.settings.caching.semantic_similarity_threshold
        return {"mode_summaries": {"GREENROUTING": {
            "avg_quality_score": quality,
            "total_energy_kwh": energy,
        }}}


def test_optimizer_selects_lowest_energy_trial_meeting_quality_floor_and_restores_settings():
    settings = Settings()
    trials = [
        FakeTrial({"complexity_threshold": 0.4, "cache_similarity_threshold": 0.90}),
        FakeTrial({"complexity_threshold": 0.6, "cache_similarity_threshold": 0.95}),
    ]
    study = FakeStudy(trials)
    optuna = SimpleNamespace(
        samplers=SimpleNamespace(TPESampler=lambda **kwargs: kwargs),
        create_study=lambda **kwargs: study,
    )

    result = run_parameter_optimization(
        SimpleNamespace(settings=settings),
        queries=["question"],
        n_trials=2,
        min_quality_score=0.5,
        benchmark_runner_factory=FakeBenchmarkRunner,
        optuna_module=optuna,
    )

    assert result["feasible"] is True
    assert result["recommended"]["cache_similarity_threshold"] == 0.95
    assert result["recommended"]["avg_quality_score"] == 0.6
    assert settings.routing.complexity_threshold == 0.50
    assert settings.caching.semantic_similarity_threshold == 0.88


def test_optimizer_reports_when_no_trial_meets_quality_floor():
    settings = Settings()
    study = FakeStudy([
        FakeTrial({"complexity_threshold": 0.4, "cache_similarity_threshold": 0.90}),
    ])
    optuna = SimpleNamespace(
        samplers=SimpleNamespace(TPESampler=lambda **kwargs: kwargs),
        create_study=lambda **kwargs: study,
    )

    result = run_parameter_optimization(
        SimpleNamespace(settings=settings),
        queries=["question"],
        n_trials=1,
        min_quality_score=0.7,
        benchmark_runner_factory=FakeBenchmarkRunner,
        optuna_module=optuna,
    )

    assert result["feasible"] is False
    assert result["recommended"] is None
    assert result["best_quality_trial"]["avg_quality_score"] == 0.4


def test_optimizer_explains_when_optuna_dependency_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)
    with pytest.raises(RuntimeError, match="pip install -r requirements.txt"):
        run_parameter_optimization(
            SimpleNamespace(settings=Settings()), ["question"], 2, 0.5,
            benchmark_runner_factory=FakeBenchmarkRunner,
        )


@pytest.mark.parametrize("queries,n_trials,min_quality", [([], 3, 0.5), (["q"], 0, 0.5), (["q"], 3, 1.1)])
def test_optimizer_rejects_invalid_inputs(queries, n_trials, min_quality):
    with pytest.raises(ValueError):
        run_parameter_optimization(
            SimpleNamespace(settings=Settings()), queries, n_trials, min_quality,
            benchmark_runner_factory=FakeBenchmarkRunner,
            optuna_module=SimpleNamespace(),
        )
