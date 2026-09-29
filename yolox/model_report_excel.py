"""Fill the supplied training ficha and add editable Excel charts."""

from pathlib import Path

from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.styles import Alignment, Font


TEMPLATE = Path(__file__).parent / "templates" / "training_model_template.xlsx"


def _value(value):
    return "No disponible" if value is None or value == "" else str(value)


def _put(sheet, address, value):
    if value is None or value == "":
        return
    cell = sheet[address]
    cell.value = value
    cell.font = Font(name="Arial", size=10, color="17243B")
    cell.alignment = Alignment(vertical="center", wrap_text=True)


def _split_text(dataset):
    splits = dataset.get("splits") or {}
    counts = {name: (splits.get(name) or {}).get("images") for name in ("train", "val", "test")}
    known = [v for v in counts.values() if isinstance(v, (int, float))]
    if not known or len(known) != 3 or sum(known) == 0:
        return ", ".join(f"{name}: {_value(counts[name])}" for name in counts)
    total = sum(known)
    return " / ".join(f"{name}: {counts[name] / total:.1%}" for name in counts)


def _metric_rows(report):
    best = report.get("best") or {}
    labels = (("map_50_95", "mAP 50:95"), ("ap50", "AP50"),
              ("ap75", "AP75"), ("precision", "Precisión"),
              ("recall", "Recall"), ("average_recall", "AR"))
    rows = []
    for key, label in labels:
        value = best.get(key)
        if value is not None:
            rows.append((label, value, "Validación; mejor época {}".format(_value(best.get("epoch")))))
    if best.get("inference_ms") is not None:
        rows.append(("Latencia (ms)", best["inference_ms"], "Inferencia"))
    return rows


def _chart(sheet, title, records, fields, anchor, percent=False):
    if not records:
        sheet[anchor] = title + ": historial no disponible"
        return
    start = max(sheet.max_column + 2, 18)
    sheet.cell(1, start, "Época")
    for offset, (_, label) in enumerate(fields, 1):
        sheet.cell(1, start + offset, label)
    for row_index, record in enumerate(records, 2):
        sheet.cell(row_index, start, record.get("epoch"))
        for offset, (key, _) in enumerate(fields, 1):
            value = record.get(key)
            if isinstance(value, (int, float)):
                sheet.cell(row_index, start + offset, value)
                if percent:
                    sheet.cell(row_index, start + offset).number_format = "0.0%"
    chart = LineChart()
    chart.title = title
    chart.style = 13
    chart.y_axis.title = "Calidad" if percent else "Valor"
    chart.x_axis.title = "Época"
    chart.width = 19
    chart.height = 9
    chart.add_data(Reference(sheet, min_col=start + 1, max_col=start + len(fields),
                             min_row=1, max_row=len(records) + 1), titles_from_data=True)
    chart.set_categories(Reference(sheet, min_col=start, min_row=2, max_row=len(records) + 1))
    sheet.add_chart(chart, anchor)


def write_excel_report(report, output, template=TEMPLATE):
    """Create the ficha from the report data without replacing template guidance."""
    workbook = load_workbook(template)
    sheet = workbook["Formato"]
    model = report.get("model") or {}
    training = report.get("training") or {}
    hardware = report.get("hardware") or {}
    dataset = report.get("dataset") or {}
    artifacts = report.get("artifacts") or {}
    splits = dataset.get("splits") or {}
    source = ", ".join(str((splits.get(name) or {}).get("annotations"))
                       for name in ("train", "val", "test") if (splits.get(name) or {}).get("annotations"))
    checkpoint = (artifacts.get("best_ckpt.pth") or {}).get("path")
    values = {
        "C10": model.get("project"), "G10": model.get("name"),
        "K10": model.get("version"), "C12": model.get("architecture"),
        "C14": training.get("started_at"),
        "G14": report.get("completed_at") if model.get("base_version") else "No aplica",
        "G18": "Entrenar y evaluar un detector de objetos YOLOX.",
        "G22": "Entrenamiento con el conjunto train y evaluación con validación; clases: {}.".format(
            _value(dataset.get("num_classes"))),
        "G26": "Resultados de detección y pesos del modelo para el proyecto {}.".format(_value(model.get("project"))),
        "G32": source or None, "G36": dataset.get("total_images"),
        "G40": _split_text(dataset),
        "G47": "Entrenamiento supervisado YOLOX; {} épocas; lote {}.".format(
            _value(training.get("max_epoch")), _value(training.get("batch_size"))),
        "G51": "Python, PyTorch {}, YOLOX".format(_value(hardware.get("torch"))),
        "G55": "GPU: {}; CUDA: {}; {} dispositivo(s).".format(
            _value(hardware.get("gpu")), _value(hardware.get("cuda")), _value(hardware.get("world_size"))),
        "G59": model.get("experiment"), "G63": checkpoint,
    }
    for address, value in values.items():
        _put(sheet, address, value)
    first = workbook["Ficha Técnica"]
    first_values = {
        "B10": values["C10"], "B14": values["G10"], "H14": values["K10"],
        "B18": values["C12"], "H22": values["C14"], "B26": values["G14"],
        "B32": values["G18"], "B35": values["G22"], "B38": values["G26"],
        "B43": values["G32"], "B46": values["G36"], "B49": values["G40"],
        "B54": values["G47"], "B57": values["G51"], "B60": values["G55"],
        "B63": values["G59"], "B66": values["G63"],
        "B71": "Validación en el conjunto val y evaluación por clase.",
    }
    for address, value in first_values.items():
        _put(first, address, value)
    for row, (label, value, note) in enumerate(_metric_rows(report)[:6], 70):
        _put(sheet, f"B{row}", label)
        _put(sheet, f"D{row}", value)
        _put(sheet, f"H{row}", note)
    for row, (label, value, note) in enumerate(_metric_rows(report)[:6], 77):
        _put(first, f"B{row}", label)
        _put(first, f"D{row}", value)
        _put(first, f"H{row}", "Validación")
        _put(first, f"K{row}", note)
    observations = []
    if len(_metric_rows(report)) > 6:
        observations.append("Métricas adicionales: " + ", ".join(
            f"{name}={value}" for name, value, _ in _metric_rows(report)[6:]))
    observations.append("Campos sin datos de origen: completar manualmente (responsables, cliente, metas y mantenimiento).")
    _put(sheet, "B92", " ".join(observations))

    charts = workbook.create_sheet("Gráficas")
    charts["A1"] = "Curvas del entrenamiento"
    charts["A1"].font = Font(name="Arial", size=16, bold=True, color="17243B")
    charts["A2"] = "Fuente: historial y evaluaciones de esta ejecución; se conservan los valores por época."
    charts.column_dimensions["A"].width = 18
    charts.column_dimensions["L"].width = 18
    evaluations = report.get("evaluations") or []
    history = report.get("training_history") or []
    _chart(charts, "Calidad en validación", evaluations,
           (("map_50_95", "mAP 50:95"), ("ap50", "AP50"),
            ("ap75", "AP75"), ("average_recall", "AR")), "A4", True)
    _chart(charts, "Pérdida total", history, (("total_loss", "Total"),), "L4")
    _chart(charts, "Componentes de la pérdida", history,
           (("iou_loss", "IoU"), ("cls_loss", "Clasificación"),
            ("conf_loss", "Confianza"), ("l1_loss", "L1")), "A24")
    _chart(charts, "Tasa de aprendizaje", history,
           (("learning_rate", "Learning rate"),), "L24")
    charts.freeze_panes = "A3"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
