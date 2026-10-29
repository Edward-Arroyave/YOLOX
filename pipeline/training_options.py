"""Training options a Vision Label Studio user may change (VLS spec 039).

A closed catalog: every key has a type, a range and a default. The defaults are
the current cassette experiment (``exps/cassette/cassette_yolox.py``) and
``TRAIN_BATCH_SIZE``; only the options that differ travel in ``request.json``,
so a request without options trains exactly as before. The agent and the
pipeline both validate with ``validate``; nothing reaches a shell.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# The experiment's learning rate is written per 64 images ("0.0015 / 64.0").
LR_IMAGES = 64


@dataclass(frozen=True)
class Option:
    key: str
    kind: str  # "int", "float" or "bool"
    default: object
    minimum: float | None = None
    maximum: float | None = None
    attribute: str | None = None  # Exp attribute set through tools/train.py; None = pipeline argument


CATALOG = (
    Option("epochs", "int", 120, 1, 1000, "max_epoch"),
    Option("batchSize", "int", 8, 1, 64),
    Option("learningRate", "float", 0.0015, 0.000001, 0.1, "basic_lr_per_img"),
    Option("noAugEpochs", "int", 60, 0, 1000, "no_aug_epochs"),
    Option("warmupEpochs", "int", 5, 0, 100, "warmup_epochs"),
    Option("minLrRatio", "float", 0.1, 0, 1, "min_lr_ratio"),
    Option("weightDecay", "float", 0.0005, 0, 0.1, "weight_decay"),
    Option("momentum", "float", 0.9, 0, 0.999, "momentum"),
    Option("ema", "bool", True, attribute="ema"),
    Option("evalInterval", "int", 5, 1, 100, "eval_interval"),
    Option("fp16", "bool", True),
    Option("mosaicProb", "float", 0.25, 0, 1, "mosaic_prob"),
    Option("mosaicScaleMin", "float", 0.9, 0.1, 2, "mosaic_scale"),
    Option("mosaicScaleMax", "float", 1.1, 0.1, 2, "mosaic_scale"),
    Option("degrees", "float", 1.0, 0, 45, "degrees"),
    Option("translate", "float", 0.02, 0, 0.5, "translate"),
    Option("shear", "float", 0.0, 0, 10, "shear"),
    Option("flipProb", "float", 0.0, 0, 1, "flip_prob"),
    Option("mixupProb", "float", 0.0, 0, 1, "mixup_prob"),
)
BY_KEY = {option.key: option for option in CATALOG}


def defaults() -> dict:
    return {option.key: option.default for option in CATALOG}


def _checked(option: Option, value):
    if option.kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{option.key} debe ser true o false")
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
    return value


def validate(options) -> dict:
    """Request options → the same options, typed; ValueError if anything is off."""
    if not isinstance(options, dict):
        raise ValueError("trainingOptions debe ser un objeto")
    unknown = sorted(set(options) - set(BY_KEY))
    if unknown:
        raise ValueError(f"Opción de entrenamiento desconocida: {', '.join(map(str, unknown))}")
    chosen = {key: _checked(BY_KEY[key], value) for key, value in options.items()}
    effective = {**defaults(), **chosen}
    if effective["warmupEpochs"] >= effective["epochs"]:
        raise ValueError("warmupEpochs debe ser menor que epochs")
    if effective["noAugEpochs"] > effective["epochs"]:
        raise ValueError("noAugEpochs no puede ser mayor que epochs")
    if effective["mosaicScaleMin"] > effective["mosaicScaleMax"]:
        raise ValueError("mosaicScaleMin no puede ser mayor que mosaicScaleMax")
    return chosen


def _text(value) -> str:
    return ("true" if value else "false") if isinstance(value, bool) else str(value)


def exp_overrides(options: dict) -> list[str]:
    """Validated options → ``key value`` pairs for ``tools/train.py`` (Exp.merge)."""
    pairs: list[str] = []
    effective = {**defaults(), **options}
    scale_done = False
    for key, value in options.items():
        option = BY_KEY[key]
        if option.attribute is None:
            continue
        if option.attribute == "mosaic_scale":
            if not scale_done:
                pairs += ["mosaic_scale", f"{effective['mosaicScaleMin']},{effective['mosaicScaleMax']}"]
                scale_done = True
        elif key == "learningRate":
            pairs += ["basic_lr_per_img", str(value / LR_IMAGES)]
        elif key == "mixupProb":
            pairs += ["mixup_prob", str(value), "enable_mixup", _text(value > 0)]
        else:
            pairs += [option.attribute, _text(value)]
    return pairs


def describe(options: dict) -> str:
    if not options:
        return "por defecto"
    return ", ".join(f"{key}={_text(value)}" for key, value in options.items())
