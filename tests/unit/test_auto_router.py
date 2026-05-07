"""Tests for konjoai.retrieve.auto_router (Sprint 25)."""
import pytest
from konjoai.retrieve.auto_router import AutoRouter, RouteStrategy, RouteDecision


@pytest.fixture
def router():
    return AutoRouter()


def test_correct_routes_direct(router):
    d = router.decide("correct")
    assert d.strategy == RouteStrategy.DIRECT


def test_ambiguous_routes_self_rag(router):
    d = router.decide("ambiguous")
    assert d.strategy == RouteStrategy.SELF_RAG


def test_incorrect_routes_decompose(router):
    d = router.decide("incorrect")
    assert d.strategy == RouteStrategy.DECOMPOSE


def test_case_insensitive_correct(router):
    assert router.decide("CORRECT").strategy == RouteStrategy.DIRECT


def test_case_insensitive_ambiguous(router):
    assert router.decide("Ambiguous").strategy == RouteStrategy.SELF_RAG


def test_case_insensitive_incorrect(router):
    assert router.decide("INCORRECT").strategy == RouteStrategy.DECOMPOSE


def test_unknown_classification_decomposes(router):
    d = router.decide("garbage_value")
    assert d.strategy == RouteStrategy.DECOMPOSE


def test_decision_carries_classification(router):
    d = router.decide("correct")
    assert d.crag_classification == "correct"


def test_decision_carries_score(router):
    d = router.decide("correct", crag_score=0.95)
    assert d.crag_score == 0.95


def test_decision_score_none_by_default(router):
    d = router.decide("ambiguous")
    assert d.crag_score is None


def test_rationale_not_empty(router):
    for cls in ("correct", "ambiguous", "incorrect"):
        d = router.decide(cls)
        assert len(d.rationale) > 0


def test_route_decision_is_frozen(router):
    d = router.decide("correct")
    with pytest.raises((AttributeError, TypeError)):
        d.strategy = RouteStrategy.DECOMPOSE  # type: ignore[misc]


def test_strategy_values_are_strings():
    assert RouteStrategy.DIRECT.value == "direct"
    assert RouteStrategy.SELF_RAG.value == "self_rag"
    assert RouteStrategy.DECOMPOSE.value == "decompose"
