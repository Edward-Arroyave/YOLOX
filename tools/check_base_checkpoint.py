#!/usr/bin/env python3
"""Comprueba que un checkpoint base tenga el tamaño del modelo del experimento (spec 039 r2 de VLS).

``yolox.utils.load_ckpt`` omite, solo con un aviso, las capas cuya forma no
coincide: un YOLOX-s usado como base de un YOLOX-m entrenaría casi desde cero
sin que nadie lo note. Se comparan todas las capas salvo ``head.cls_preds.*``,
que cambian con el número de clases (un ajuste fino a otras clases es válido).
"""
import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

CLASS_LAYERS = "head.cls_preds."


def mismatched_layers(model_state, checkpoint_state) -> list[str]:
    """Layers of the model missing from the checkpoint or with another shape (class layers excluded)."""
    return [key for key, value in model_state.items()
            if not key.startswith(CLASS_LAYERS)
            and (key not in checkpoint_state or tuple(checkpoint_state[key].shape) != tuple(value.shape))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-file", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    import torch
    from yolox.exp import get_exp

    exp = get_exp(args.exp_file, None)
    model_state = exp.get_model().state_dict()
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    checkpoint_state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else {}
    mismatched = mismatched_layers(model_state, checkpoint_state)
    size = getattr(exp, "model_size", None)
    chosen = f"YOLOX-{size}" if size else f"depth {exp.depth}, width {exp.width}"
    if mismatched:
        print(
            f"ERROR: El modelo base no es del tamaño elegido ({chosen}): {len(mismatched)} capas no coinciden, "
            f"por ejemplo {', '.join(mismatched[:3])}. Elige un modelo base del mismo tamaño o cambia el tamaño.",
            file=sys.stderr,
        )
        return 1
    print(f"Modelo base compatible con {chosen} ({len(model_state)} capas).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
