import cv2
import numpy as np
import onnxruntime as ort

CLASSES = ["cassette", "test"]
CONF_THRESH = 0.3
NMS_THRESH = 0.45
INPUT_SIZE = 640

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
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)

    # Cargar modelo ONNX
    session = ort.InferenceSession("vet_yolox_decoded.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_input})
    
    # Inspeccionar salida
    print(f"Número de salidas: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Salida {i}: shape {out.shape}, dtype {out.dtype}")
    
    predictions = outputs[0]  # asumimos que la primera es la detección
    print(f"Forma de predicciones: {predictions.shape}")
    
    # Si la forma es (1, N, 6) o (N, 6) lo tratamos
    if len(predictions.shape) == 3:
        predictions = predictions[0]  # quitar batch
    print(f"Predictions shape after squeeze: {predictions.shape}")
    
    # Ver rango de confianza
    if predictions.shape[1] >= 5:
        conf_col = 4
        print(f"Rango de confianza: min={predictions[:, conf_col].min():.4f}, max={predictions[:, conf_col].max():.4f}")
        print("Primeras 10 filas:")
        print(predictions[:10])
    else:
        print("Forma inesperada, no hay columna de confianza")
        return

    # Filtrar por confianza
    mask = predictions[:, conf_col] >= CONF_THRESH
    filtered = predictions[mask]
    print(f"Filtradas por confianza > {CONF_THRESH}: {len(filtered)}")
    
    if len(filtered) == 0:
        # Si no hay, probar con umbral bajo para depurar
        mask_low = predictions[:, conf_col] >= 0.01
        filtered_low = predictions[mask_low]
        print(f"Con umbral 0.01 hay {len(filtered_low)} detecciones. Las primeras:")
        print(filtered_low[:5])
        return

    boxes = filtered[:, :4]
    scores = filtered[:, conf_col]
    class_ids = filtered[:, 5].astype(np.int32)

    # NMS
    keep = nms(boxes, scores, NMS_THRESH)
    final_boxes = boxes[keep]
    final_scores = scores[keep]
    final_classes = class_ids[keep]

    # Dibujar
    for i, box in enumerate(final_boxes):
        x1, y1, x2, y2 = box
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
        cv2.putText(img_orig, label, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print(f"Detecciones finales tras NMS: {len(final_boxes)}")
    cv2.imwrite("resultado_onnx_fixed.jpg", img_orig)
    print("Guardado: resultado_onnx_fixed.jpg")

if __name__ == "__main__":
    main()