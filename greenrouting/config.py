"""
Configuration loader for GreenRouting / EcoWave.
Loads defaults from config/config.yaml and allows environment variable overrides.
"""
import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


@dataclass
class AppConfig:
    name: str = "EcoWave"
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
    force_simulation_mode: bool = False


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
class HardwarePowerConfig:
    cpu_tdp_watts: float = 45.0
    gpu_tdp_watts: float = 75.0
    idle_overhead_watts: float = 5.0


@dataclass
class StagePowerConfig:
    query_processing_watts: float = 10.0
    cache_lookup_watts: float = 12.0
    retrieval_embedding_watts: float = 25.0
    routing_analysis_watts: float = 15.0


@dataclass
class ModelPowerConfig:
    lightweight_model_watts: float = 27.0
    capable_model_watts: float = 93.75
    default_model_watts: float = 45.0


@dataclass
class BaselineConfig:
    mode: str = "STANDARD_RAG"
    model: str = "llama3.2:3b"
    enable_cache: bool = False
    top_k_retrieval: int = 3


@dataclass
class EnergyConfig:
    measurement_mode: str = "modeled_tdp_estimate"
    tracking_mode: str = "modeled"
    country_iso_code: str = "IND"
    offline_mode: bool = True
    grid_carbon_intensity_g_per_kwh: float = 708.0
    emission_factor_kg_per_kwh: float = 0.708
    cpu_tdp_watts: float = 45.0
    gpu_tdp_watts: float = 75.0
    hardware: HardwarePowerConfig = field(default_factory=HardwarePowerConfig)
    stage_power: StagePowerConfig = field(default_factory=StagePowerConfig)
    model_power: ModelPowerConfig = field(default_factory=ModelPowerConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)


@dataclass
class Settings:
    app: AppConfig = field(default_factory=AppConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)


def _safe_float(val: Any, default: float) -> float:
    try:
        f = float(val)
        return f if f >= 0 else default
    except (TypeError, ValueError):
        return default


def load_settings(custom_path: Optional[Path] = None) -> Settings:
    """Load settings from yaml file or return defaults with safe validation."""
    cfg_file = custom_path or CONFIG_PATH
    data: Dict[str, Any] = {}

    if cfg_file.exists() and yaml is not None:
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Failed to parse YAML config at %s: %s. Using default settings.", cfg_file, e)

    app_data = data.get("app", {})
    models_data = data.get("models", {})
    rag_data = data.get("rag", {})
    routing_data = data.get("routing", {})
    caching_data = data.get("caching", {})
    energy_data = data.get("energy", {})
    baseline_data = data.get("baseline", {})

    # Energy structured sub-blocks
    hardware_data = energy_data.get("hardware", {})
    cpu_tdp = _safe_float(hardware_data.get("cpu_tdp_watts", energy_data.get("cpu_tdp_watts", 45.0)), 45.0)
    gpu_tdp = _safe_float(hardware_data.get("gpu_tdp_watts", energy_data.get("gpu_tdp_watts", 75.0)), 75.0)
    idle_power = _safe_float(hardware_data.get("idle_overhead_watts", 5.0), 5.0)

    hardware_cfg = HardwarePowerConfig(
        cpu_tdp_watts=cpu_tdp,
        gpu_tdp_watts=gpu_tdp,
        idle_overhead_watts=idle_power
    )

    stage_data = energy_data.get("stage_power", {})
    stage_cfg = StagePowerConfig(
        query_processing_watts=_safe_float(stage_data.get("query_processing_watts", 10.0), 10.0),
        cache_lookup_watts=_safe_float(stage_data.get("cache_lookup_watts", 12.0), 12.0),
        retrieval_embedding_watts=_safe_float(stage_data.get("retrieval_embedding_watts", 25.0), 25.0),
        routing_analysis_watts=_safe_float(stage_data.get("routing_analysis_watts", 15.0), 15.0),
    )

    model_power_data = energy_data.get("model_power", {})
    model_power_cfg = ModelPowerConfig(
        lightweight_model_watts=_safe_float(model_power_data.get("lightweight_model_watts", cpu_tdp * 0.60), 27.0),
        capable_model_watts=_safe_float(model_power_data.get("capable_model_watts", cpu_tdp + gpu_tdp * 0.65), 93.75),
        default_model_watts=_safe_float(model_power_data.get("default_model_watts", cpu_tdp), 45.0),
    )

    baseline_cfg = BaselineConfig(
        mode=str(baseline_data.get("mode", "STANDARD_RAG")),
        model=str(baseline_data.get("model", models_data.get("capable_model", "llama3.2:3b"))),
        enable_cache=bool(baseline_data.get("enable_cache", False)),
        top_k_retrieval=int(baseline_data.get("top_k_retrieval", 3)),
    )

    # Emissions factor
    grid_intensity = _safe_float(
        energy_data.get("grid_carbon_intensity_g_per_kwh",
                        _safe_float(energy_data.get("emission_factor_kg_per_kwh", 0.708), 0.708) * 1000.0),
        708.0
    )
    emission_factor_kg = grid_intensity / 1000.0

    energy_cfg = EnergyConfig(
        measurement_mode=str(energy_data.get("measurement_mode", "modeled_tdp_estimate")),
        tracking_mode=str(energy_data.get("tracking_mode", "modeled")),
        country_iso_code=str(energy_data.get("country_iso_code", "IND")),
        offline_mode=bool(energy_data.get("offline_mode", True)),
        grid_carbon_intensity_g_per_kwh=grid_intensity,
        emission_factor_kg_per_kwh=emission_factor_kg,
        cpu_tdp_watts=cpu_tdp,
        gpu_tdp_watts=gpu_tdp,
        hardware=hardware_cfg,
        stage_power=stage_cfg,
        model_power=model_power_cfg,
        baseline=baseline_cfg
    )

    settings = Settings(
        app=AppConfig(
            name=str(app_data.get("name", "EcoWave")),
            version=str(app_data.get("version", "1.0.0")),
            description=str(app_data.get("description", "Energy-Aware RAG")),
            data_dir=ROOT_DIR / app_data.get("data_dir", "data"),
            db_path=ROOT_DIR / app_data.get("db_path", "data/greenrouting.db"),
            faiss_index_dir=ROOT_DIR / app_data.get("faiss_index_dir", "data/faiss_index"),
        ),
        models=ModelConfig(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", str(models_data.get("ollama_base_url", "http://localhost:11434"))),
            lightweight_model=os.getenv("LIGHTWEIGHT_MODEL", str(models_data.get("lightweight_model", "qwen2.5:0.5b"))),
            capable_model=os.getenv("CAPABLE_MODEL", str(models_data.get("capable_model", "llama3.2:3b"))),
            embedding_model=os.getenv("EMBEDDING_MODEL", str(models_data.get("embedding_model", "all-MiniLM-L6-v2"))),
            enable_simulation_fallback=bool(models_data.get("enable_simulation_fallback", True)),
            force_simulation_mode=bool(models_data.get("force_simulation_mode", False)),
        ),
        rag=RagConfig(
            chunk_size=int(rag_data.get("chunk_size", 500)),
            chunk_overlap=int(rag_data.get("chunk_overlap", 50)),
            top_k_retrieval=int(rag_data.get("top_k_retrieval", 3)),
            similarity_metric=str(rag_data.get("similarity_metric", "cosine")),
        ),
        routing=RoutingConfig(
            complexity_threshold=_safe_float(routing_data.get("complexity_threshold", 0.50), 0.50),
            weights=routing_data.get("weights", {
                "length": 0.20,
                "reasoning_keywords": 0.35,
                "syntactic_complexity": 0.25,
                "domain_depth": 0.20
            }),
            enable_retry_on_low_quality=bool(routing_data.get("enable_retry_on_low_quality", True)),
            min_acceptable_quality_score=_safe_float(routing_data.get("min_acceptable_quality_score", 0.55), 0.55),
        ),
        caching=CachingConfig(
            enabled=bool(caching_data.get("enabled", True)),
            semantic_similarity_threshold=_safe_float(caching_data.get("semantic_similarity_threshold", 0.88), 0.88),
            enforce_document_context_match=bool(caching_data.get("enforce_document_context_match", True)),
            min_cache_grounding_score=_safe_float(caching_data.get("min_cache_grounding_score", 0.60), 0.60),
            max_cache_age_hours=int(caching_data.get("max_cache_age_hours", 72)),
        ),
        energy=energy_cfg,
        baseline=baseline_cfg
    )

    try:
        settings.app.data_dir.mkdir(parents=True, exist_ok=True)
        settings.app.faiss_index_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("Could not create data directories: %s", e)

    return settings
