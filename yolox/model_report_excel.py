"""Fill the supplied training ficha from Markdown and measured training data."""

from pathlib import Path

TEMPLATE = Path(__file__).parent / "templates" / "training_model_template.xlsx"
MARKDOWN_TEMPLATE = Path(__file__).resolve().parents[1] / "pipeline" / "ficha_modelo.md"


def _value(value):
    return "No disponible" if value is None or value == "" else str(value)


def _put(sheet, address, value):
    from openpyxl.styles import Alignment, Font

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


METRICS = (("map_50_95", "mAP 50:95"), ("ap50", "AP50"),
           ("ap75", "AP75"), ("average_recall", "AR"),
           ("inference_ms", "Latencia (ms)"),
           ("precision", "Precisión"), ("recall", "Recall"))


def read_markdown(path):
    """Read simple `- key: value` fields under the two supported headings."""
    sections = {"Campos editables": {}, "Observaciones de métricas": {}}
    current = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip() if line[3:].strip() in sections else None
        elif current and line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            sections[current][key.strip()] = value.strip()
    return sections


def write_markdown_report(report, output, template=MARKDOWN_TEMPLATE):
    """Write an editable run snapshot plus measured values for review."""
    model = report.get("model") or {}
    training = report.get("training") or {}
    hardware = report.get("hardware") or {}
    artifacts = report.get("artifacts") or {}
    best = report.get("best") or {}
    notes = read_markdown(template)["Observaciones de métricas"]
    text = Path(template).read_text(encoding="utf-8").rstrip()
    lines = ["", "## Datos medidos del entrenamiento (no editables)",
             "Estos valores se toman de `metrics.json` al generar el Excel; los cambios aquí no los reemplazan.", "",
             "| Campo | Valor |", "| --- | --- |"]
    facts = (
        ("Proyecto", model.get("project")), ("Modelo", model.get("name")),
        ("Versión", model.get("version")), ("Épocas", training.get("max_epoch")),
        ("Lote", training.get("batch_size")), ("PyTorch", hardware.get("torch")),
        ("GPU", hardware.get("gpu")), ("CUDA", hardware.get("cuda")),
        ("Dispositivos", hardware.get("world_size")),
        ("Repositorio del código", model.get("experiment")),
        ("Artefacto", (artifacts.get("best_ckpt.pth") or {}).get("path")),
    )
    lines.extend(f"| {label} | {_value(value)} |" for label, value in facts)
    lines.extend(["", "| Métrica | Valor obtenido | Meta / umbral | Observaciones |",
                  "| --- | ---: | --- | --- |"])
    for key, label in METRICS:
        if best.get(key) is not None:
            lines.append(f"| {label} | {best[key]} | 90 % | {notes.get(key, '')} |")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _metric_rows(report, notes):
    best = report.get("best") or {}
    rows = []
    for key, label in METRICS:
        value = best.get(key)
        if value is not None:
            rows.append((label, value, notes.get(key, ""), key))
    return rows


def write_excel_report(report, output, markdown, template=TEMPLATE):
    """Create the ficha from the report data without replacing template guidance."""
    from openpyxl import load_workbook

    content = read_markdown(markdown)
    editable = content["Campos editables"]
    notes = content["Observaciones de métricas"]
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
        "G12": editable.get("responsable_tecnico"),
        "K12": editable.get("product_owner"), "M12": editable.get("scrum_master"),
        "K14": editable.get("cliente"), "M14": editable.get("empresa"),
        "G18": editable.get("objetivo"), "G22": editable.get("alcance"),
        "G26": editable.get("mejoras"),
        "G32": editable.get("origen_datos") or source or None, "G36": dataset.get("total_images"),
        "G40": _split_text(dataset),
        "G47": "Entrenamiento supervisado YOLOX; {} épocas; lote {}.".format(
            _value(training.get("max_epoch")), _value(training.get("batch_size"))),
        "G51": "Python, PyTorch {}, YOLOX".format(_value(hardware.get("torch"))),
        "G55": "GPU: {}; CUDA: {}; {} dispositivo(s).".format(
            _value(hardware.get("gpu")), _value(hardware.get("cuda")), _value(hardware.get("world_size"))),
        "G59": model.get("experiment"), "G63": checkpoint,
        "G79": editable.get("plan_reentrenamiento"),
        "G83": editable.get("criterios_reentrenamiento"),
        "G87": editable.get("responsable_monitoreo"),
    }
    for address, value in values.items():
        _put(sheet, address, value)
    first = workbook["Ficha Técnica"]
    first_values = {
        "B10": values["C10"], "H10": values["K14"],
        "B14": values["G10"], "H14": values["K10"],
        "B18": values["C12"], "H18": values["G12"],
        "B22": values["K12"] or values["M12"],
        "H22": values["C14"], "B26": values["G14"],
        "B32": values["G18"], "B35": values["G22"], "B38": values["G26"],
        "B43": values["G32"], "B46": values["G36"], "B49": values["G40"],
        "B54": values["G47"], "B57": values["G51"], "B60": values["G55"],
        "B63": values["G59"], "B66": values["G63"],
        "B71": editable.get("pruebas"),
        "B87": values["G79"], "B90": values["G83"], "B93": values["G87"],
    }
    for address, value in first_values.items():
        _put(first, address, value)
    rows = _metric_rows(report, notes)
    for row, (label, value, note, key) in enumerate(rows[:6], 70):
        _put(sheet, f"B{row}", label)
        _put(sheet, f"D{row}", value)
        _put(sheet, f"F{row}", "90 %")
        _put(sheet, f"H{row}", note)
    for row, (label, value, note, key) in enumerate(rows[:6], 77):
        _put(first, f"B{row}", label)
        _put(first, f"D{row}", value)
        _put(first, f"F{row}", "90 %")
        _put(first, f"H{row}", "Inferencia" if key == "inference_ms" else "Validación")
        _put(first, f"K{row}", note)
    observations = []
    if len(rows) > 6:
        observations.append("Métricas adicionales: " + ", ".join(
            f"{name}={value} ({note})" for name, value, note, _ in rows[6:]))
    observations.append("La meta del 90 % es una referencia del formato; la latencia se mide en ms y requiere un umbral propio.")
    _put(sheet, "B92", " ".join(observations))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
