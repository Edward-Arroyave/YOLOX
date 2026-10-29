"""Experimento general de YOLOX configurable desde Vision Label Studio (spec 039 r2).

La configuración llega en un JSON cuya ruta está en ``YOLOX_TRAINING_CONFIG``
(el pipeline la escribe y la pasa a entrenamiento, test y exportación, así las
tres etapas cargan el mismo modelo). Sin la variable usa el perfil «cassettes»
(``exps/general/profiles/cassettes.json``). El número de clases sale del JSON
de anotaciones de entrenamiento del dataset descargado.
"""

import json
import os
from pathlib import Path

import torch.nn as nn
from dotenv import load_dotenv
from loguru import logger

from pipeline import training_options
from yolox.exp import Exp as MyExp

# Las variables del sistema tienen prioridad sobre el archivo .env.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=False)

# Profundidad y anchura de cada tamaño de YOLOX (exps/default/yolox_*.py).
MODEL_SIZES = {
    "nano": (0.33, 0.25),
    "tiny": (0.33, 0.375),
    "s": (0.33, 0.50),
    "m": (0.67, 0.75),
    "l": (1.0, 1.0),
    "x": (1.33, 1.25),
}
# Sin dataset (por ejemplo, una exportación manual): las clases de cassettes.
FALLBACK_CLASSES = 3


def load_training_config() -> dict:
    """«cassettes» plus the file in YOLOX_TRAINING_CONFIG, validated with the pipeline catalog."""
    path = os.getenv("YOLOX_TRAINING_CONFIG")
    if not path:
        return training_options.effective({})
    chosen = json.loads(Path(path).read_text(encoding="utf-8"))
    return training_options.effective(training_options.validate(chosen))


def count_classes(annotations: Path) -> int:
    if not annotations.is_file():
        logger.warning("No existe {}; se usan {} clases (cassettes).", annotations, FALLBACK_CLASSES)
        return FALLBACK_CLASSES
    categories = json.loads(annotations.read_text(encoding="utf-8")).get("categories") or []
    if not categories:
        raise ValueError(f"{annotations} no tiene categorías")
    return len(categories)


class Exp(MyExp):
    """YOLOX configurable: tamaño del modelo, entrada, aprendizaje y aumentos."""

    def __init__(self):
        super().__init__()
        config = load_training_config()
        self.training_config = config

        self.model_size = config["modelSize"]
        self.depth, self.width = MODEL_SIZES[self.model_size]
        self.depthwise = self.model_size == "nano"
        self.act = "silu"

        size = config["inputSize"]
        self.input_size = (size, size)
        self.test_size = (size, size)
        self.random_size = (size // 32 - 6, size // 32)

        self.max_epoch = config["epochs"]
        self.batch_size = config["batchSize"]
        self.warmup_epochs = config["warmupEpochs"]
        self.no_aug_epochs = config["noAugEpochs"]
        self.eval_interval = config["evalInterval"]
        self.print_interval = 20
        self.save_history_ckpt = True

        # La tasa se expresa por 64 imágenes; YOLOX la multiplica por el tamaño de lote.
        self.basic_lr_per_img = config["learningRate"] / 64.0
        self.min_lr_ratio = config["minLrRatio"]
        self.weight_decay = config["weightDecay"]
        self.momentum = config["momentum"]
        self.ema = config["ema"]

        self.degrees = config["degrees"]
        self.translate = config["translate"]
        self.shear = config["shear"]
        self.perspective = 0.0
        self.flip_prob = config["flipProb"]
        self.mosaic_prob = config["mosaicProb"]
        self.mosaic_scale = (config["mosaicScaleMin"], config["mosaicScaleMax"])
        self.mixup_prob = config["mixupProb"]
        self.enable_mixup = config["mixupProb"] > 0

        # El pipeline define YOLOX_DATA_DIR para el lote descargado.
        self.data_dir = os.getenv("YOLOX_DATA_DIR", "datasets/COCO")
        self.train_image_dir = "training/images"
        self.val_image_dir = "val/images"
        self.test_image_dir = "test/images"
        self.annotations_dir = "."
        self.train_ann = "training/annotations/annotations.json"
        self.val_ann = "val/annotations/annotations.json"
        self.test_ann = "test/annotations/annotations.json"
        self.data_num_workers = 2

        self.num_classes = count_classes(Path(self.data_dir) / self.train_ann)
        self.exp_name = "yolox_general"

    def get_model(self, sublinear=False):
        if not self.depthwise:
            return super().get_model()

        # YOLOX-nano usa convoluciones depthwise (exps/default/yolox_nano.py).
        def init_yolo(module):
            for layer in module.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eps = 1e-3
                    layer.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act, depthwise=True)
            self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
