"""
Unified model type registry used by the new model runtime layer.

This mirrors the previous core-level behavior so existing config remains valid.
"""

from dataclasses import dataclass
import re


MODEL_TYPE_PATTERN = re.compile(r"^(?P<family>yolo[a-z0-9]+)_(?P<precision>int8|fp16|fp32)$")
SUPPORTED_MODEL_TYPE_VALUES = {"yolo26_int8", "yolov8_fp16", "yolov8_int8"}


@dataclass(frozen=True)
class ModelTypeSpec:
    model_key: str
    raw: str
    family: str
    precision: str


def _visible_keys(mapping):
    return {
        key
        for key in (mapping or {})
        if isinstance(key, str) and not key.startswith("_")
    }


def parse_model_type(model_key, raw_value):
    if not isinstance(raw_value, str):
        raise ValueError(f"model_types['{model_key}'] must be a string, got {type(raw_value).__name__}")

    normalized = raw_value.strip().lower()
    match = MODEL_TYPE_PATTERN.fullmatch(normalized)
    if not match:
        raise ValueError(
            f"model_types['{model_key}'] must use family_precision format like "
            f"'yolov8_fp16', 'yolov8_int8', or 'yolo26_int8', got '{raw_value}'"
        )

    return ModelTypeSpec(
        model_key=model_key,
        raw=normalized,
        family=match.group("family"),
        precision=match.group("precision"),
    )


def validate_model_types_config(config):
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")

    models = config.get("models")
    if not isinstance(models, dict):
        raise ValueError("config.models must be a dict")

    model_types = config.get("model_types")
    if not isinstance(model_types, dict):
        raise ValueError("config.model_types must be a dict")

    model_keys = _visible_keys(models)
    declared_keys = _visible_keys(model_types)

    missing = sorted(model_keys - declared_keys)
    extra = sorted(declared_keys - model_keys)
    if missing:
        raise ValueError(f"config.model_types is missing keys for models: {', '.join(missing)}")
    if extra:
        raise ValueError(f"config.model_types contains unknown keys: {', '.join(extra)}")

    for model_key in sorted(model_keys):
        parse_model_type(model_key, model_types[model_key])


def resolve_model_type(model_key, config):
    validate_model_types_config(config)

    models = config.get("models") or {}
    if model_key not in models:
        raise KeyError(f"Unknown model key '{model_key}' in config.models")

    raw_value = config["model_types"][model_key]
    return parse_model_type(model_key, raw_value)


def ensure_supported_model_type(spec):
    if spec.raw in SUPPORTED_MODEL_TYPE_VALUES:
        return spec

    raise NotImplementedError(
        "Unsupported model type for '{}': '{}'. Supported values in this build: {}".format(
            spec.model_key,
            spec.raw,
            ", ".join(sorted(SUPPORTED_MODEL_TYPE_VALUES)),
        )
    )
