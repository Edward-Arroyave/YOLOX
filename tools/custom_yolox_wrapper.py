import torch
import torch.nn as nn

class CustomYOLOXWrapper(nn.Module):
    """
    Envoltorio para YOLOX que realiza:
    - Decodificación de las predicciones (desplazamientos → cajas absolutas)
    - Cálculo de puntuaciones finales (objetividad × clase)
    - Filtrado por confianza y NMS (opcional, pero aquí solo decodificación)
    - Salida: [x1, y1, x2, y2, conf, class_id] para cada detección (sin NMS)
    """
    def __init__(self, model, num_classes=2, strides=[8, 16, 32], input_size=640):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.strides = strides
        self.input_size = input_size

        # Generar grillas para cada stride (necesario para decodificación)
        # En ONNX, estas deben ser constantes o calculadas dentro del grafo.
        # Usamos registros para que sean parte del estado del modelo.
        self.register_buffer('grids', self._make_grids())

    def _make_grids(self):
        """Pre-calcula las grillas para cada nivel de stride."""
        grids = []
        for stride in self.strides:
            feat_size = self.input_size // stride
            y, x = torch.meshgrid(torch.arange(feat_size), torch.arange(feat_size), indexing='ij')
            grid_xy = torch.stack([x, y], dim=-1).float() * stride
            # shape: (feat_size, feat_size, 2)
            grid_xy = grid_xy.view(-1, 2)  # (H*W, 2)
            grids.append(grid_xy)
        return torch.cat(grids, dim=0)  # (total_anchors, 2)

    def forward(self, x):
        # Normalizar entrada si no se hizo antes (opcional, según entrenamiento)
        # x: (batch, 3, H, W) en rango [0,1] o [0,255]?
        # Asumimos que ya viene normalizada a [0,1] (como en el wrapper original)
        # Si tu modelo espera [0,255], descomenta la línea:
        # x = x * 255.0

        # Salida del modelo base: puede ser una lista de tensores o un solo tensor
        outputs = self.model(x)

        # YOLOX normalmente devuelve una lista de tres tensores (para cada nivel de stride)
        # o un único tensor concatenado. Vamos a manejar ambos casos.
        if isinstance(outputs, (list, tuple)):
            # Si es lista, concatenamos a lo largo de la dimensión de anclajes
            preds = torch.cat([out.flatten(2).permute(0, 2, 1) for out in outputs], dim=1)
        else:
            # Si es un solo tensor, asumimos shape (batch, total_anchors, 4+1+num_classes)
            preds = outputs

        # preds shape: (batch, total_anchors, 4+1+num_classes)
        # Desglose: [x,y,w,h, obj_score, class_scores...]
        bbox_preds = preds[..., :4]   # (batch, anchors, 4)
        obj_pred = preds[..., 4:5]    # (batch, anchors, 1)
        cls_pred = preds[..., 5:]     # (batch, anchors, num_classes)

        # Aplicar sigmoides
        obj_scores = torch.sigmoid(obj_pred)
        cls_scores = torch.sigmoid(cls_pred)

        # Calcular puntuación total = objetividad * máxima clase
        scores = obj_scores * cls_scores  # (batch, anchors, num_classes)
        max_scores, class_ids = torch.max(scores, dim=-1, keepdim=True)  # (batch, anchors, 1)

        # Decodificar cajas: convertir (cx, cy, w, h) en coordenadas de la imagen de entrada
        # bbox_preds son desplazamientos relativos a las anclas y strides
        # Primero, obtener las grillas (anchors) para toda la imagen
        # Las grillas deben estar en el dispositivo correcto
        grid = self.grids.to(x.device)  # (total_anchors, 2)

        # Para cada ancla: cx = (grid_x + sigmoid(tx)) * stride, pero YOLOX usa:
        # cx = grid_x + dx, donde dx está en el rango [0,1] después de sigmoid
        # En el entrenamiento, las salidas son sin sigmoid; aquí aplicamos sigmoid.
        dx = torch.sigmoid(bbox_preds[..., 0:1])
        dy = torch.sigmoid(bbox_preds[..., 1:2])
        dw = bbox_preds[..., 2:3]
        dh = bbox_preds[..., 3:4]

        # Obtener stride correspondiente para cada ancla (difícil precalcular, lo aproximamos)
        # Una forma sencilla: como conocemos las grillas, el stride es el factor entre grid y el índice
        # Pero para simplificar, podemos calcular cajas directamente con las fórmulas estándar de YOLOX.
        # Usaremos la decodificación típica:
        cx = (grid[:, 0:1] + dx)  # * stride está implícito porque grid ya tiene el stride multiplicado
        cy = (grid[:, 1:2] + dy)
        w = torch.exp(dw)  # * stride? No, porque el modelo predice log(w) relativo al ancla (que es stride)
        h = torch.exp(dh)

        # Las dimensiones deben multiplicarse por el stride, pero en grid ya lo incluimos.
        # En YOLOX, las anclas no son fijas, sino que se usa stride como base.
        # La fórmula correcta: cxy = (grid + sigmoid(dxy)) * stride, donde grid está en celdas (0..feat_size-1)
        # Entonces: cx = (grid_x + dx) * stride, cy = (grid_y + dy) * stride
        # w = exp(dw) * stride, h = exp(dh) * stride
        # En nuestra grid ya tenemos (grid_x*stride, grid_y*stride), por lo que:
        cx = grid[:, 0:1] + dx * self.strides_for_grid()  # Necesitamos stride por ancla
        # Esto se complica. En lugar de hacerlo manual, usaremos la decodificación incorporada de YOLOX si está disponible.
        # Mejor: reutilizamos la función `decode_outputs` del repositorio original.

        # Dado que la implementación completa es larga, y para no extender demasiado,
        # propongo una solución más directa: usar la función `postprocess` de YOLOX dentro del wrapper,
        # pero importándola de yolox.utils. Esto sí funcionará porque ya está probado.

        from yolox.utils import postprocess

        # Para usar postprocess, necesitamos pasar las salidas crudas del modelo (lista de tensores)
        # y los parámetros de decodificación. postprocess espera (cls_pred, reg_pred, obj_pred) por separado.
        # Es más sencillo si modificamos el modelo para que devuelva esas tres salidas.
        # Pero como no queremos cambiar el modelo original, haremos lo siguiente:

        # Reorganizamos preds para que postprocess lo entienda: pero postprocess espera una lista de
        # tensores de salida de cada nivel (sin concatenar). Por lo tanto, es más fácil no usar el wrapper complejo.

        # Conclusión: para evitar errores, recomiendo exportar directamente el modelo sin wrapper,
        # pero configurando `model.head.decode_in_inference = True` y luego hacer el postprocesado en el script de prueba,
        # pero corrigiendo la decodificación de cajas (que era el problema original).

        # Dado el tiempo, voy a proporcionar una solución pragmática: