"""
Unit tests for Quality-Aware Cache Validator in GreenRouting.
Ensures validation rejects context mismatches, stale answers, and unsupported claims.
"""
import pytest
from datetime import datetime, timedelta, timezone
from greenrouting.config import CachingConfig
from greenrouting.cache_validator import CacheQualityValidator, CacheValidationStatus


def test_validator_rejects_document_context_mismatch():
    config = CachingConfig(semantic_similarity_threshold=0.85, enforce_document_context_match=True)
    validator = CacheQualityValidator(config)

    candidate = {
        "cache_id": "c_1",
        "answer_text": "Solar energy provides 22% efficiency in commercial silicon panels.",
        "source_doc_ids": ["doc_solar_report"],
        "grounding_score": 0.85,
        "is_active": 1,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    # Querying with active doc_id "doc_nuclear_report" which does NOT match "doc_solar_report"
    is_valid, status, reason = validator.validate_cache_candidate(
        candidate_entry=candidate,
        similarity_score=0.92,
        current_doc_ids=["doc_nuclear_report"]
    )

    assert is_valid is False
    assert status == CacheValidationStatus.HIT_REJECTED_CONTEXT_MISMATCH
    assert "Context mismatch" in reason


def test_validator_rejects_stale_cache():
    config = CachingConfig(semantic_similarity_threshold=0.85, max_cache_age_hours=24)
    validator = CacheQualityValidator(config)

    # 48 hours old entry
    old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    candidate = {
        "cache_id": "c_2",
        "answer_text": "Cached information.",
        "source_doc_ids": ["doc_1"],
        "grounding_score": 0.90,
        "is_active": 1,
        "created_at": old_time
    }

    is_valid, status, reason = validator.validate_cache_candidate(
        candidate_entry=candidate,
        similarity_score=0.95,
        current_doc_ids=["doc_1"]
    )

    assert is_valid is False
    assert status == CacheValidationStatus.HIT_REJECTED_STALE
    assert "expired" in reason.lower()


def test_validator_rejects_low_grounding_cache():
    config = CachingConfig(semantic_similarity_threshold=0.85, min_cache_grounding_score=0.60)
    validator = CacheQualityValidator(config)

    candidate = {
        "cache_id": "c_3",
        "answer_text": "Hallucinated or unsupported claim.",
        "source_doc_ids": ["doc_1"],
        "grounding_score": 0.35, # Below 0.60
        "is_active": 1,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    is_valid, status, reason = validator.validate_cache_candidate(
        candidate_entry=candidate,
        similarity_score=0.96,
        current_doc_ids=["doc_1"]
    )

    assert is_valid is False
    assert status == CacheValidationStatus.HIT_REJECTED_LOW_GROUNDING


def test_validator_rejects_unsupported_claims_against_new_chunks():
    config = CachingConfig(semantic_similarity_threshold=0.85)
    validator = CacheQualityValidator(config)

    candidate = {
        "cache_id": "c_4",
        "answer_text": "Quantum computing cryogenic cooling requires liquid helium at four Kelvin temperature.",
        "source_doc_ids": ["doc_tech"],
        "grounding_score": 0.90,
        "is_active": 1,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    # Retrieved chunks talk about completely different topics (e.g. agriculture)
    retrieved_chunks = [{
        "text_content": "Organic farming and crop rotation enhance soil nutrient absorption in spring."
    }]

    is_valid, status, reason = validator.validate_cache_candidate(
        candidate_entry=candidate,
        similarity_score=0.91,
        current_doc_ids=["doc_tech"],
        retrieved_chunks=retrieved_chunks
    )

    assert is_valid is False
    assert status == CacheValidationStatus.HIT_REJECTED_UNSUPPORTED
