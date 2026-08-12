# Copyright (c) 2026 IT Health. All rights reserved.
# Confidential and proprietary. See LICENSE and NOTICE.

import cv2
import numpy as np
import onnxruntime as ort

CLASSES = ["cassette", "test"]
CONF_THRESH = 0.1
NMS_THRESH = 0.1
INPUT_SIZE = 640
STRIDES = [8, 16, 32]

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    left = dw // 2
    right = dw - left
    top = dh // 2
    bottom = dh - top
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    return img_padded, (r, r), (left, top)

def nms(boxes, scores, threshold):
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]
    return keep

def generate_anchors(input_size, strides):
    anchors = []
    for stride in strides:
        feat_size = input_size // stride
        y, x = np.meshgrid(np.arange(feat_size), np.arange(feat_size), indexing='ij')
        grid_xy = np.stack([x, y], axis=-1).reshape(-1, 2).astype(np.float32) * stride
        anchors.append(grid_xy)
    return np.concatenate(anchors, axis=0)  # (total_anchors, 2)

def decode_predictions(predictions, anchors, num_classes):
    """
    predictions: (N, 7) -> [tx, ty, tw, th, obj, cls1, cls2, ...]
    anchors: (N, 2) -> (grid_x, grid_y)
    """
    tx = predictions[:, 0]
    ty = predictions[:, 1]
    tw = predictions[:, 2]
    th = predictions[:, 3]
    obj = predictions[:, 4]
    cls = predictions[:, 5:5+num_classes]

    # Sigmoides
    obj_scores = 1 / (1 + np.exp(-obj))
    cls_scores = 1 / (1 + np.exp(-cls))
    scores = obj_scores[:, np.newaxis] * cls_scores
    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)

    # Decodificar cajas
    grid_x = anchors[:, 0]
    grid_y = anchors[:, 1]
    cx = grid_x + (1 / (1 + np.exp(-tx)))
    cy = grid_y + (1 / (1 + np.exp(-ty)))
    w = np.exp(tw)
    h = np.exp(th)

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Clips
    x1 = np.clip(x1, 0, INPUT_SIZE)
    y1 = np.clip(y1, 0, INPUT_SIZE)
    x2 = np.clip(x2, 0, INPUT_SIZE)
    y2 = np.clip(y2, 0, INPUT_SIZE)

    detections = np.stack([x1, y1, x2, y2, max_scores, class_ids], axis=1)
    return detections

def main():
    img_path = "/home/kravmaga/projects/YOLOX/datasets/COCO/train2017/auto_20260514_030906_cassette_multiple_double_0006.png"
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        print("Error: no se pudo cargar la imagen")
        return
    h0, w0 = img_orig.shape[:2]

    # Preprocesar
    img_padded, (rw, rh), (left, top) = letterbox(img_orig, (INPUT_SIZE, INPUT_SIZE))
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    # IMPORTANTE: prueba con o sin división por 255. Según tu entrenamiento, puede necesitar [0,255]
    img_norm = img_rgb.astype(np.float32)  # sin dividir (rango 0-255)
    # img_norm = img_rgb.astype(np.float32) / 255.0  # descomentar si el modelo espera [0,1]
    img_input = np.transpose(img_norm, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)

    # Cargar modelo ONNX
    session = ort.InferenceSession("vet_yolox_decoded.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_input})
    raw = outputs[0][0]  # (8400, 7)

    # Generar anclas
    anchors = generate_anchors(INPUT_SIZE, STRIDES)

    # Decodificar
    detections = decode_predictions(raw, anchors, len(CLASSES))

    # Filtrar por confianza
    mask = detections[:, 4] >= CONF_THRESH
    filtered = detections[mask]
    if len(filtered) == 0:
        print(f"No detecciones con umbral {CONF_THRESH}. Probando umbral 0.1...")
        mask2 = detections[:, 4] >= 0.1
        filtered = detections[mask2]
        if len(filtered) == 0:
            print("No hay detecciones incluso con umbral 0.1.")
            return
        else:
            print(f"Se encontraron {len(filtered)} detecciones con umbral 0.1.")

    boxes = filtered[:, :4]
    scores = filtered[:, 4]
    class_ids = filtered[:, 5].astype(np.int32)

    # NMS
    keep = nms(boxes, scores, NMS_THRESH)
    final_boxes = boxes[keep]
    final_scores = scores[keep]
    final_classes = class_ids[keep]

    # Dibujar
    for i, box in enumerate(final_boxes):
        x1, y1, x2, y2 = box
        # Revertir letterbox
        x1 = (x1 - left) / rw
        x2 = (x2 - left) / rw
        y1 = (y1 - top) / rh
        y2 = (y2 - top) / rh
        x1 = max(0, min(x1, w0))
        x2 = max(0, min(x2, w0))
        y1 = max(0, min(y1, h0))
        y2 = max(0, min(y2, h0))
        cv2.rectangle(img_orig, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{CLASSES[final_classes[i]]} {final_scores[i]:.2f}"
        cv2.putText(img_orig, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print(f"Detecciones finales: {len(final_boxes)}")
    cv2.imwrite("resultado_manual.jpg", img_orig)
    print("Guardado: resultado_manual.jpg")

if __name__ == "__main__":
    main()
