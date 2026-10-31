"""
Configuration loader for GreenRouting.
Loads defaults from config/config.yaml and allows environment variable overrides.
"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


@dataclass
class AppConfig:
    name: str = "GreenRouting"
    version: str = "1.0.0"
    description: str = "An Energy-Aware RAG Framework"
    data_dir: Path = ROOT_DIR / "data"
    db_path: Path = ROOT_DIR / "data" / "greenrouting.db"
    faiss_index_dir: Path = ROOT_DIR / "data" / "faiss_index"


@dataclass
class ModelConfig:
    ollama_base_url: str = "http://localhost:11434"
    lightweight_model: str = "qwen2.5:0.5b"
    capable_model: str = "llama3.2:3b"
    embedding_model: str = "all-MiniLM-L6-v2"
    enable_simulation_fallback: bool = True


@dataclass
class RagConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_retrieval: int = 3
    similarity_metric: str = "cosine"


@dataclass
class RoutingConfig:
    complexity_threshold: float = 0.50
    weights: Dict[str, float] = field(default_factory=lambda: {
        "length": 0.20,
        "reasoning_keywords": 0.35,
        "syntactic_complexity": 0.25,
        "domain_depth": 0.20
    })
    enable_retry_on_low_quality: bool = True
    min_acceptable_quality_score: float = 0.55


@dataclass
class CachingConfig:
    enabled: bool = True
    semantic_similarity_threshold: float = 0.88
    enforce_document_context_match: bool = True
    min_cache_grounding_score: float = 0.60
    max_cache_age_hours: int = 72


@dataclass
class EnergyConfig:
    tracking_mode: str = "auto"
    country_iso_code: str = "IND"
    offline_mode: bool = True
    cpu_tdp_watts: float = 45.0
    gpu_tdp_watts: float = 75.0
    emission_factor_kg_per_kwh: float = 0.708


@dataclass
class Settings:
    app: AppConfig = field(default_factory=AppConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)


def load_settings(custom_path: Path = None) -> Settings:
    """Load settings from yaml file or return defaults."""
    cfg_file = custom_path or CONFIG_PATH
    data: Dict[str, Any] = {}

    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    app_data = data.get("app", {})
    models_data = data.get("models", {})
    rag_data = data.get("rag", {})
    routing_data = data.get("routing", {})
    caching_data = data.get("caching", {})
    energy_data = data.get("energy", {})

    settings = Settings(
        app=AppConfig(
            name=app_data.get("name", "GreenRouting"),
            version=app_data.get("version", "1.0.0"),
            description=app_data.get("description", "Energy-Aware RAG"),
            data_dir=ROOT_DIR / app_data.get("data_dir", "data"),
            db_path=ROOT_DIR / app_data.get("db_path", "data/greenrouting.db"),
            faiss_index_dir=ROOT_DIR / app_data.get("faiss_index_dir", "data/faiss_index"),
        ),
        models=ModelConfig(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", models_data.get("ollama_base_url", "http://localhost:11434")),
            lightweight_model=os.getenv("LIGHTWEIGHT_MODEL", models_data.get("lightweight_model", "qwen2.5:0.5b")),
            capable_model=os.getenv("CAPABLE_MODEL", models_data.get("capable_model", "llama3.2:3b")),
            embedding_model=os.getenv("EMBEDDING_MODEL", models_data.get("embedding_model", "all-MiniLM-L6-v2")),
            enable_simulation_fallback=models_data.get("enable_simulation_fallback", True),
        ),
        rag=RagConfig(
            chunk_size=rag_data.get("chunk_size", 500),
            chunk_overlap=rag_data.get("chunk_overlap", 50),
            top_k_retrieval=rag_data.get("top_k_retrieval", 3),
            similarity_metric=rag_data.get("similarity_metric", "cosine"),
        ),
        routing=RoutingConfig(
            complexity_threshold=routing_data.get("complexity_threshold", 0.50),
            weights=routing_data.get("weights", {
                "length": 0.20,
                "reasoning_keywords": 0.35,
                "syntactic_complexity": 0.25,
                "domain_depth": 0.20
            }),
            enable_retry_on_low_quality=routing_data.get("enable_retry_on_low_quality", True),
            min_acceptable_quality_score=routing_data.get("min_acceptable_quality_score", 0.55),
        ),
        caching=CachingConfig(
            enabled=caching_data.get("enabled", True),
            semantic_similarity_threshold=caching_data.get("semantic_similarity_threshold", 0.88),
            enforce_document_context_match=caching_data.get("enforce_document_context_match", True),
            min_cache_grounding_score=caching_data.get("min_cache_grounding_score", 0.60),
            max_cache_age_hours=caching_data.get("max_cache_age_hours", 72),
        ),
        energy=EnergyConfig(
            tracking_mode=energy_data.get("tracking_mode", "auto"),
            country_iso_code=energy_data.get("country_iso_code", "IND"),
            offline_mode=energy_data.get("offline_mode", True),
            cpu_tdp_watts=energy_data.get("cpu_tdp_watts", 45.0),
            gpu_tdp_watts=energy_data.get("gpu_tdp_watts", 75.0),
            emission_factor_kg_per_kwh=energy_data.get("emission_factor_kg_per_kwh", 0.708),
        )
    )

    # Ensure directories exist
    settings.app.data_dir.mkdir(parents=True, exist_ok=True)
    settings.app.faiss_index_dir.mkdir(parents=True, exist_ok=True)

    return settings
