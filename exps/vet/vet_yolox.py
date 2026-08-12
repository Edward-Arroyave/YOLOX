# Copyright (c) 2026 IT Health. All rights reserved.
# Confidential and proprietary. See LICENSE and NOTICE.

import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    """
    YOLOX-S optimizado para casetes médicos (Control/Test)
    Ajustado para líneas débiles y mejor estabilidad AP75.
    """

    def __init__(self):
        super().__init__()

        # ======================
        # CLASES
        # ======================
        self.num_classes = 2

        # ======================
        # MODELO
        # ======================
        self.depth = 0.33
        self.width = 0.50
        self.act = "silu"

        # ======================
        # INPUT
        # ======================
        self.input_size = (640, 640)
        self.test_size = (640, 640)

        # 🔥 CAMBIO 1:
        # multiscale menos agresivo
        # antes: (10, 20)
        self.random_size = (14, 20)

        # ======================
        # TRAINING
        # ======================

        # 🔥 CAMBIO 2:
        # menos epochs = menos overfitting
        # antes: 250
        self.max_epoch = 120

        self.warmup_epochs = 5

        # 🔥 CAMBIO 3:
        # no_aug más temprano
        # antes: 50
        self.no_aug_epochs = 60

        self.eval_interval = 5
        self.print_interval = 20
        self.save_history_ckpt = True

        # ======================
        # OPTIMIZER
        # ======================

        # mantenemos tu LR porque claramente funciona
        self.basic_lr_per_img = 0.0015 / 64.0

        # 🔥 CAMBIO 4:
        # LR final menos microscópico
        # antes: 0.05
        self.min_lr_ratio = 0.10

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.ema = True

        # ======================
        # AUGMENTATION
        # ======================

        self.degrees = 1.0
        self.translate = 0.02

        # 🔥 CAMBIO 5:
        # menos deformación geométrica
        # antes: (0.85, 1.15)
        self.scale = (0.90, 1.10)

        self.shear = 0.0
        self.perspective = 0.0

        # 🔥 CAMBIO 6:
        # flip horizontal puede romper semántica espacial
        # especialmente en bandas clínicas
        # antes: 0.5
        self.flip_prob = 0.0

        # 🔥 CAMBIO 7:
        # mosaic ligeramente menor
        # antes: 0.4
        self.mosaic_prob = 0.25

        self.mosaic_scale = (0.9, 1.1)

        # correcto
        self.enable_mixup = False
        self.mixup_prob = 0.0

        # ======================
        # COLOR AUGMENTATION
        # ======================

        self.hsv_h = 0.01

        # 🔥 CAMBIO 8:
        # color augmentation más suave
        # evita alterar intensidad real de bandas
        self.hsv_s = 0.20
        self.hsv_v = 0.20

        # ======================
        # DATASET
        # ======================
        self.data_dir = "datasets/COCO"
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        # Carpetas de imágenes relativas a data_dir.
        # También pueden cambiarse desde la línea de comandos con opts.
        self.train_image_dir = "train2017"
        self.val_image_dir = "val2017"
        self.test_image_dir = "test2017"

        self.data_num_workers = 2

        self.exp_name = os.path.splitext(
            os.path.basename(__file__)
        )[0]

    def get_model(self):
        model = super().get_model()

        # backbone SIN freeze
        return model
