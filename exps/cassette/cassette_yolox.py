"""Configuración compartida de YOLOX para entrenamiento de cassettes."""

import os
from pathlib import Path

from dotenv import load_dotenv

from exps.cassette.settings import TRAIN_BATCH_SIZE
from yolox.exp import Exp as MyExp


# Las variables del sistema tienen prioridad sobre el archivo .env.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=False)


class Exp(MyExp):
    """YOLOX-S ajustado para detectar las regiones Control y Test."""

    def __init__(self):
        super().__init__()

        self.num_classes = 2

        self.depth = 0.33
        self.width = 0.50
        self.act = "silu"

        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.random_size = (14, 20)

        self.max_epoch = 120
        self.batch_size = TRAIN_BATCH_SIZE
        self.warmup_epochs = 5
        self.no_aug_epochs = 60
        self.eval_interval = 5
        self.print_interval = 20
        self.save_history_ckpt = True

        self.basic_lr_per_img = 0.0015 / 64.0
        self.min_lr_ratio = 0.10
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.ema = True

        self.degrees = 1.0
        self.translate = 0.02
        self.scale = (0.90, 1.10)
        self.shear = 0.0
        self.perspective = 0.0
        self.flip_prob = 0.0
        self.mosaic_prob = 0.25
        self.mosaic_scale = (0.9, 1.1)
        self.enable_mixup = False
        self.mixup_prob = 0.0

        self.hsv_h = 0.01
        self.hsv_s = 0.20
        self.hsv_v = 0.20

        # El pipeline define YOLOX_DATA_DIR para el lote descargado.
        self.data_dir = os.getenv("YOLOX_DATA_DIR", "datasets/COCO")
        self.train_image_dir = "training/images"
        self.val_image_dir = "val/images"
        self.test_image_dir = "val/images"
        self.annotations_dir = "."
        self.train_ann = "training/annotations/annotations.json"
        self.val_ann = "val/annotations/annotations.json"
        self.test_ann = "val/annotations/annotations.json"
        self.data_num_workers = 2

        self.exp_name = "cassette_yolox"

    def get_model(self):
        return super().get_model()
