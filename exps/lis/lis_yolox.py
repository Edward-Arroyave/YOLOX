"""Experimento LIS basado en la configuración común de entrenamiento."""

from exps.vet.vet_yolox import Exp as VetExp


class Exp(VetExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "lis_yolox"
