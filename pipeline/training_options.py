"""Training options a Vision Label Studio user may change (VLS spec 039).

A closed catalog: every key has a type and a range. The defaults are the
«cassettes» profile (``exps/general/profiles/cassettes.json``), the former
cassette experiment; only the options that differ travel in ``request.json``,
so a request without options trains exactly as before. The agent, the pipeline
and the general experiment (``exps/general/yolox_general.py``) all validate
with ``validate``; nothing reaches a shell.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "exps" / "general" / "profiles" / "cassettes.json"
MODEL_SIZES = ("nano", "tiny", "s", "m", "l", "x")


@dataclass(frozen=True)
class Option:
    key: str
    kind: str  # "int", "float", "bool" or "choice"
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    multiple_of: int | None = None


CATALOG = (
    Option("epochs", "int", 1, 1000),
    Option("batchSize", "int", 1, 64),
    Option("learningRate", "float", 0.000001, 0.1),
    Option("noAugEpochs", "int", 0, 1000),
    Option("modelSize", "choice", choices=MODEL_SIZES),
    Option("inputSize", "int", 320, 1280, multiple_of=32),
    Option("warmupEpochs", "int", 0, 100),
    Option("minLrRatio", "float", 0, 1),
    Option("weightDecay", "float", 0, 0.1),
    Option("momentum", "float", 0, 0.999),
    Option("ema", "bool"),
    Option("evalInterval", "int", 1, 100),
    Option("fp16", "bool"),
    Option("mosaicProb", "float", 0, 1),
    Option("mosaicScaleMin", "float", 0.1, 2),
    Option("mosaicScaleMax", "float", 0.1, 2),
    Option("degrees", "float", 0, 45),
    Option("translate", "float", 0, 0.5),
    Option("shear", "float", 0, 10),
    Option("flipProb", "float", 0, 1),
    Option("mixupProb", "float", 0, 1),
)
BY_KEY = {option.key: option for option in CATALOG}


@lru_cache(maxsize=1)
def _profile_defaults() -> tuple:
    options = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))["options"]
    if set(options) != set(BY_KEY):
        raise ValueError(f"{DEFAULT_PROFILE} debe definir exactamente las opciones del catálogo")
    return tuple(options.items())


def defaults() -> dict:
    return dict(_profile_defaults())


def _checked(option: Option, value):
    if option.kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{option.key} debe ser true o false")
        return value
    if option.kind == "choice":
        if value not in option.choices:
            raise ValueError(f"{option.key} debe ser uno de: {', '.join(option.choices)}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{option.key} debe ser un número")
    if option.kind == "int":
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{option.key} debe ser un número entero")
        value = int(value)
    else:
        if not math.isfinite(value):
            raise ValueError(f"{option.key} debe ser un número finito")
        value = float(value)
    if not option.minimum <= value <= option.maximum:
        raise ValueError(f"{option.key} debe estar entre {option.minimum} y {option.maximum}")
    if option.multiple_of and value % option.multiple_of:
        raise ValueError(f"{option.key} debe ser múltiplo de {option.multiple_of}")
    return value


def validate(options) -> dict:
    """Request options → the same options, typed; ValueError if anything is off."""
    if not isinstance(options, dict):
        raise ValueError("trainingOptions debe ser un objeto")
    unknown = sorted(set(options) - set(BY_KEY))
    if unknown:
        raise ValueError(f"Opción de entrenamiento desconocida: {', '.join(map(str, unknown))}")
    chosen = {key: _checked(BY_KEY[key], value) for key, value in options.items()}
    current = effective(chosen)
    if current["warmupEpochs"] >= current["epochs"]:
        raise ValueError("warmupEpochs debe ser menor que epochs")
    if current["noAugEpochs"] > current["epochs"]:
        raise ValueError("noAugEpochs no puede ser mayor que epochs")
    if current["mosaicScaleMin"] > current["mosaicScaleMax"]:
        raise ValueError("mosaicScaleMin no puede ser mayor que mosaicScaleMax")
    return chosen


def effective(options: dict) -> dict:
    """Every option: the «cassettes» defaults plus the chosen ones."""
    return {**defaults(), **options}


def _text(value) -> str:
    return ("true" if value else "false") if isinstance(value, bool) else str(value)


def describe(options: dict) -> str:
    if not options:
        return "por defecto"
    return ", ".join(f"{key}={_text(value)}" for key, value in options.items())
