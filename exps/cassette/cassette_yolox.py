"""Alias del experimento general con el perfil «cassettes» (spec 039 r2 de Vision Label Studio).

Los valores que antes vivían aquí están en ``exps/general/profiles/cassettes.json``;
el número de clases sale del JSON de anotaciones del dataset (3 para cassettes).
"""

from exps.general.yolox_general import Exp as GeneralExp


class Exp(GeneralExp):
    """YOLOX-S ajustado para detectar las regiones Control y Test."""

    def __init__(self):
        super().__init__()
        self.exp_name = "cassette_yolox"
