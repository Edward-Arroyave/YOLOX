"""Portable HTML report with embedded vector charts; no network dependencies."""

from datetime import datetime
from html import escape
import math


MONTHS_ES = ('enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio',
             'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre')

# key -> (etiqueta técnica, explicación en lenguaje llano). Única fuente de verdad
# para las curvas, las tarjetas de resumen, la comparación visual y el glosario.
METRIC_INFO = {
    'map_50_95': ('mAP 50:95',
                  'Precisión general de detección: promedia los aciertos exigiendo niveles crecientes de '
                  'coincidencia entre el recuadro predicho y el real (del 50% al 95% de superposición). Es '
                  'la métrica más completa y más exigente; 100% sería una detección perfecta.'),
    'ap50': ('AP50',
             'Porcentaje de aciertos cuando basta con que el recuadro predicho coincida al menos un 50% con '
             'el real. Más permisiva que el mAP 50:95.'),
    'ap75': ('AP75',
             'Igual que AP50 pero exige un 75% de coincidencia con el recuadro real: mide una ubicación más '
             'fina del objeto.'),
    'average_recall': ('AR (recall promedio)',
                        'De todos los objetos presentes en las imágenes evaluadas, qué porcentaje logró '
                        'encontrar el modelo, aunque el recuadro no sea perfecto. Una AP alta con AR baja '
                        'indica que el modelo acierta cuando detecta, pero se le escapan objetos.'),
    'total_loss': ('Pérdida total',
                    'Indicador interno de cuánto se equivoca el modelo durante el entrenamiento. No es un '
                    'porcentaje ni mide calidad directamente: solo debe bajar y estabilizarse a medida que '
                    'avanzan las épocas.'),
    'iou_loss': ('Pérdida de ubicación (IoU)',
                 'Penaliza cuando el recuadro predicho no coincide bien con la posición y el tamaño reales '
                 'del objeto. Es una pérdida de entrenamiento, no el IoU de detección.'),
    'cls_loss': ('Pérdida de clasificación',
                 'Penaliza cuando el modelo confunde una clase de objeto con otra.'),
    'conf_loss': ('Pérdida de confianza',
                   'Penaliza cuando el modelo está muy seguro de una detección que no existe, o inseguro de '
                   'una que sí existe.'),
    'l1_loss': ('Pérdida L1',
                'Ajuste fino adicional de la posición del recuadro; solo se activa en las últimas épocas '
                'del entrenamiento (sin mosaico).'),
    'learning_rate': ('Tasa de aprendizaje (learning rate)',
                       'Controla qué tan grandes son los ajustes internos que hace el modelo en cada paso. '
                       'Sigue un programa predefinido (sube al inicio y luego baja); no mide la calidad del '
                       'modelo, solo cómo aprende.'),
}

FRIENDLY_NAME = {
    'map_50_95': 'Precisión general',
    'ap50': 'Precisión con margen amplio',
    'ap75': 'Precisión con margen estricto',
    'average_recall': 'Objetos encontrados',
}

CORE_METRICS = ('map_50_95', 'ap50', 'ap75', 'average_recall')

EXTRA_GLOSSARY = (
    ('Época', 'Una vuelta completa de entrenamiento sobre todas las imágenes del conjunto de entrenamiento.'),
    ('Checkpoint', 'Una copia guardada de los pesos del modelo en un momento del entrenamiento. '
                    '"best_ckpt.pth" es la copia de la época con mejor resultado de validación; '
                    '"latest_ckpt.pth" es la más reciente, sin importar si fue la mejor.'),
    ('Conjunto de entrenamiento', 'Las imágenes que el modelo usa para aprender.'),
    ('Conjunto de validación', 'Imágenes que el modelo NO usa para aprender, solo para medir qué tan bien '
                                'generaliza a imágenes que no ha visto.'),
    ('N/A', 'Información no disponible o no calculada por este pipeline. No significa que el valor sea cero.'),
)


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else "N/A"
    return escape(str(value))


def pct(value, decimals=1):
    if not isinstance(value, float) or not math.isfinite(value):
        return 'N/A'
    return f'{value * 100:.{decimals}f}%'


def format_datetime(value):
    if not value:
        return 'N/A'
    try:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return f'{dt.day} de {MONTHS_ES[dt.month - 1]} de {dt.year}, {dt.hour:02d}:{dt.minute:02d} UTC'
    except (ValueError, TypeError, IndexError):
        return fmt(value)


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(
        f'<th>{escape(h)}</th>' for h in headers
    ) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{fmt(v)}</td>' for v in row) + '</tr>' for row in rows
    ) + '</tbody></table></div>'


def per_class_table(per_class):
    rows = [(name, pct(values.get('ap')), pct(values.get('ar')))
            for name, values in (per_class or {}).items()]
    return table(['Clase', 'AP', 'AR'], rows)


def delta_badge(diff, higher_is_better=True):
    if not isinstance(diff, float) or not math.isfinite(diff):
        return '<span class="delta unknown">Sin modelo base para comparar</span>'
    if abs(diff) < 1e-9:
        return '<span class="delta flat">Igual que el modelo base</span>'
    improved = (diff > 0) == higher_is_better
    arrow = '▲' if diff > 0 else '▼'
    cls = 'up' if improved else 'down'
    word = 'mejora' if improved else 'retrocede'
    return f'<span class="delta {cls}">{arrow} {abs(diff) * 100:.1f} pts frente al modelo base ({word})</span>'


def stat_card(key, value, diff):
    tech_label, hint = METRIC_INFO.get(key, (key, ''))
    friendly = FRIENDLY_NAME.get(key)
    friendly_html = f'<br><span class="friendly">{escape(friendly)}</span>' if friendly else ''
    value_html = pct(value) if isinstance(value, float) else 'N/A'
    raw_html = (f'<span class="raw">valor exacto: {fmt(value)}</span>'
                if isinstance(value, float) and math.isfinite(value) else '')
    return (f'<section class="stat"><p>{escape(tech_label)} '
            f'<span class="hint" title="{escape(hint, quote=True)}">ⓘ</span>{friendly_html}</p>'
            f'<strong>{value_html}</strong>{raw_html}{delta_badge(diff)}</section>')


def bar_line(tag, value, css_class):
    width = max(0.0, min(100.0, value * 100))
    return (f'<div class="bar-line"><span class="bar-tag">{escape(tag)}</span>'
            f'<div class="bar-track"><div class="bar {css_class}" style="width:{width:.1f}%"></div></div>'
            f'<span class="bar-value">{pct(value)}</span></div>')


def compare_bars(metrics):
    rows = []
    for key in CORE_METRICS:
        row = metrics.get(key, {}) or {}
        new = row.get('new')
        if not isinstance(new, float) or not math.isfinite(new):
            continue
        label, hint = METRIC_INFO.get(key, (key, ''))
        base = row.get('base')
        if isinstance(base, float) and math.isfinite(base):
            base_html = bar_line('Base', base, 'base')
        else:
            base_html = '<p class="no-base">Sin modelo base con este dato.</p>'
        rows.append('<div class="bar-row"><p class="bar-label">' + escape(label) +
                     f' <span class="hint" title="{escape(hint, quote=True)}">ⓘ</span></p>' +
                     base_html + bar_line('Nuevo', new, 'new') + delta_badge(row.get('difference')) + '</div>')
    if not rows:
        return '<p>Todavía no hay métricas de calidad para comparar visualmente.</p>'
    return '<div class="bars">' + ''.join(rows) + '</div>'


def executive_summary(report):
    model = report.get('model', {}) or {}
    best = report.get('best') or {}
    comp = report.get('comparison', {}) or {}
    dataset = report.get('dataset', {}) or {}
    training = report.get('training', {}) or {}
    metrics = comp.get('metrics', {}) or {}
    map_row = metrics.get('map_50_95', {}) or {}

    project = model.get('project') or 'YOLOX'
    version = fmt(model.get('version'))
    base_version = model.get('base_version')
    sentence1 = f'Se entrenó <strong>{escape(project)} versión {escape(version)}</strong>'
    if base_version and base_version != 'none':
        sentence1 += f' a partir del modelo base <strong>{escape(str(base_version))}</strong>'
    else:
        sentence1 += ' como primer modelo publicado de este proyecto (sin modelo base previo)'

    total_images = dataset.get('total_images')
    splits = dataset.get('splits', {}) or {}
    train_images = (splits.get('train') or {}).get('images')
    val_images = (splits.get('val') or {}).get('images')
    if isinstance(total_images, (int, float)):
        sentence1 += f', usando {fmt(total_images)} imágenes'
        if isinstance(train_images, (int, float)) and isinstance(val_images, (int, float)):
            sentence1 += f' ({fmt(train_images)} para entrenar y {fmt(val_images)} para validar)'
    sentence1 += '.'

    new_map = map_row.get('new')
    if isinstance(new_map, float) and math.isfinite(new_map):
        sentence2 = (f'El mejor resultado se obtuvo en la época {fmt(best.get("epoch"))} de '
                     f'{fmt(training.get("max_epoch"))}: en promedio, el modelo identifica correctamente el '
                     f'{pct(new_map)} de los objetos (mAP 50:95)')
        base_map, diff = map_row.get('base'), map_row.get('difference')
        if isinstance(base_map, float) and math.isfinite(base_map) and isinstance(diff, float):
            if diff > 1e-9:
                trend = f'una mejora de {diff * 100:.1f} puntos porcentuales'
            elif diff < -1e-9:
                trend = f'una reducción de {abs(diff) * 100:.1f} puntos porcentuales'
            else:
                trend = 'sin cambio'
            sentence2 += f', frente al {pct(base_map)} del modelo base — {trend}'
        sentence2 += '.'
    else:
        sentence2 = 'Todavía no hay evaluaciones de calidad registradas para este modelo.'

    warning = ''
    has_base_comparison = any(isinstance(row.get('base'), float) for row in metrics.values())
    if has_base_comparison and not comp.get('same_validation'):
        warning = ('<p class="notice">Esta comparación es histórica: no se confirmó que el modelo base se '
                    'haya evaluado con exactamente el mismo conjunto de validación (mismas imágenes y '
                    'anotaciones) que este modelo. La diferencia mostrada no demuestra, por sí sola, que un '
                    'modelo sea mejor que el otro.</p>')

    return f'<p class="lead">{sentence1} {sentence2}</p>{warning}'


def glossary_section():
    order = ('map_50_95', 'ap50', 'ap75', 'average_recall', 'total_loss',
              'iou_loss', 'cls_loss', 'conf_loss', 'l1_loss', 'learning_rate')
    items = [METRIC_INFO[key] for key in order] + list(EXTRA_GLOSSARY)
    entries = ''.join(f'<dt>{escape(label)}</dt><dd>{escape(desc)}</dd>' for label, desc in items)
    return ('<h2>Glosario de términos</h2>'
            '<p>Resultados reales, sin suavizado. Las pérdidas son el promedio por iteración de cada época '
            'del proceso principal (rank 0). El historial de curvas pertenece a esta ejecución; las épocas '
            'previas a una reanudación pueden no estar incluidas. Este pipeline no calcula validation loss, '
            'precision ni recall puntual, ni IoU de detección — esos campos se muestran como N/A, que no '
            'equivale a un resultado de cero. Las métricas de referencia y los hashes de integridad se '
            'conservan en metrics.json.</p>'
            f'<dl class="glossary">{entries}</dl>')


def chart(title, records, keys, unit, best_epoch=None, explanation=None,
          value_fmt=None, axis_fmt=None, y_max=None):
    colors = ['#2563eb', '#059669', '#d97706', '#db2777', '#7c3aed', '#475569']
    value_fmt = value_fmt or (lambda y: f'{y:.6g}')
    axis_fmt = axis_fmt or value_fmt
    series = []
    for key, label in keys:
        points = [(r.get('epoch'), r.get(key)) for r in records]
        points = [(x, y) for x, y in points if isinstance(x, (int, float))
                  and isinstance(y, (int, float)) and math.isfinite(x) and math.isfinite(y)]
        if points:
            series.append((key, label, points))
    head = f'<section><h3>{escape(title)}</h3>'
    if explanation:
        head += f'<p class="explain">{escape(explanation)}</p>'
    if not series:
        return head + '<p>No hay historial disponible para esta curva.</p></section>'
    xs = [x for _, _, points in series for x, _ in points]
    ys = [y for _, _, points in series for _, y in points]
    xmin, xmax = min(xs), max(xs)
    ymax = y_max if y_max is not None else (max(ys) * 1.08 or 1)
    xmap = lambda x: 65 + (x - xmin) / (xmax - xmin or 1) * 665
    ymap = lambda y: 275 - min(y, ymax) / ymax * 235
    svg = [f'<svg viewBox="0 0 780 340" role="img" aria-label="{escape(title, quote=True)}">',
           f'<text x="65" y="20">{escape(unit)}</text>']
    for i in range(6):
        y = ymax * i / 5
        svg += [f'<path d="M65 {ymap(y)} H730" stroke="#e2e8f0"/>',
                f'<text x="58" y="{ymap(y) + 4}" text-anchor="end">{axis_fmt(y)}</text>']
    for x in sorted(set((xmin, (xmin + xmax) / 2, xmax))):
        svg.append(f'<text x="{xmap(x)}" y="296" text-anchor="middle">{x:g}</text>')
    if best_epoch is not None and xmin <= best_epoch <= xmax:
        x = xmap(best_epoch)
        svg.append(f'<path d="M{x} 35 V275" stroke="#64748b" stroke-dasharray="5 5"/>')
    for idx, (key, label, points) in enumerate(series):
        color = colors[idx % len(colors)]
        coords = ' '.join(f'{xmap(x):.2f},{ymap(y):.2f}' for x, y in points)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in points:
            svg.append(f'<circle cx="{xmap(x)}" cy="{ymap(y)}" r="2.5" fill="{color}">'
                       f'<title>{escape(label)} · época {x}: {value_fmt(y)}</title></circle>')
    svg += ['<text x="395" y="325" text-anchor="middle">Época</text>', '</svg>']
    legend = ' '.join(
        f'<span style="color:{colors[i % len(colors)]}" '
        f'title="{escape(METRIC_INFO.get(key, ("", ""))[1], quote=True)}">● {escape(label)}</span>'
        for i, (key, label, _) in enumerate(series)
    )
    return head + ''.join(svg) + f'<div class="legend">{legend}</div></section>'


def render_html(report):
    model, best = report.get('model', {}), report.get('best') or {}
    comp = report.get('comparison', {})
    history = report.get('training_history', [])
    evaluations = report.get('evaluations', [])
    dataset = report.get('dataset', {})
    training = report.get('training', {})
    metrics = comp.get('metrics', {})
    title = f"{model.get('project', 'YOLOX')} · {model.get('version', 'N/A')}"

    body = [f'<header><p>INFORME DE ENTRENAMIENTO · YOLOX</p><h1>{escape(title)}</h1>',
            f'<p>Mejor checkpoint: época {fmt(best.get("epoch"))} de {fmt(training.get("max_epoch"))}'
            f' · Finalizado: {format_datetime(report.get("completed_at"))}</p></header>',
            '<main>', executive_summary(report), '<h2>Resumen</h2><div class="cards">']
    for key in CORE_METRICS:
        row = metrics.get(key, {})
        body.append(stat_card(key, row.get('new'), row.get('difference')))
    body.append('</div>')

    body += ['<h2>Comparación visual con el modelo base</h2>',
             '<p class="explain">Cada fila compara el resultado del modelo anterior (gris) con el nuevo '
             '(azul) en la misma métrica; barras más largas son mejores.</p>',
             compare_bars(metrics)]

    body += ['<h2>Curvas del entrenamiento</h2>',
             '<p class="explain">Cada gráfica muestra la evolución de un indicador a lo largo de las épocas. '
             'La línea vertical gris marca la época del mejor resultado. Pasa el cursor sobre cualquier punto '
             'para ver su valor exacto.</p>',
             '<div class="charts">',
             chart('Calidad en validación', evaluations,
                   [('map_50_95', 'mAP 50:95'), ('ap50', 'AP50'), ('ap75', 'AP75'), ('average_recall', 'AR')],
                   'Calidad (0-100%)', best.get('epoch'),
                   explanation='Mide sobre imágenes que el modelo no usó para aprender (validación). Más '
                               'arriba es mejor; lo ideal es que las líneas suban y luego se mantengan '
                               'estables cerca del final.',
                   value_fmt=lambda y: f'{y * 100:.2f}%', axis_fmt=lambda y: f'{y * 100:.0f}%', y_max=1.0),
             chart('Pérdida total', history, [('total_loss', 'Total')], 'Pérdida (sin unidad fija)',
                   best.get('epoch'),
                   explanation='Resume cuánto se equivoca el modelo en cada época. No es un porcentaje: lo '
                               'esperado es que baje y se estabilice; no se compara directamente contra las '
                               'métricas de calidad de arriba.'),
             chart('Componentes de la pérdida', history,
                   [(k, v) for k, v in [('iou_loss', 'IoU'), ('cls_loss', 'Clasificación'),
                                        ('conf_loss', 'Confianza'), ('l1_loss', 'L1')]],
                   'Pérdida (sin unidad fija)', best.get('epoch'),
                   explanation='Desglosa la pérdida total por causa: ubicación (IoU), clasificación, '
                               'confianza y ajuste fino (L1). Sirve para diagnosticar en qué falla más el '
                               'modelo.'),
             chart('Tasa de aprendizaje (learning rate)', history, [('learning_rate', 'LR al cierre de época')],
                   'Valor programado',
                   explanation='Configuración interna que controla el tamaño de los ajustes durante el '
                               'entrenamiento (calentamiento inicial y luego descenso programado). No mide '
                               'la calidad del modelo ni requiere acción del usuario.'),
             '</div>']

    body += ['<h2>Calidad por clase · mejor checkpoint</h2>',
             '<p class="explain">AP y AR se calculan por separado para cada tipo de objeto que el modelo '
             'debe reconocer, en la época del mejor checkpoint (0% a 100%, más alto es mejor).</p>',
             per_class_table(best.get('per_class', {}))]

    body += ['<h2>Dataset</h2>',
             '<p class="explain">Imágenes usadas para entrenar y validar el modelo. El conjunto de '
             'entrenamiento le enseña al modelo; el de validación mide su desempeño con imágenes que no usó '
             'para aprender.</p>',
             table(['Versión', 'Imágenes totales', 'Clases'],
                   [(dataset.get('version'), dataset.get('total_images'), dataset.get('num_classes'))]),
             table(['Conjunto', 'Imágenes', 'Anotaciones por clase'],
                   [(name, (values or {}).get('images'), (values or {}).get('annotations_per_class'))
                    for name, values in dataset.get('splits', {}).items()]),
             '<p class="explain">Crecimiento del dataset frente al modelo base, contando solo imágenes '
             'nuevas por contenido (N/A si no hay modelo base o no se pudo verificar identidad de '
             'imágenes).</p>',
             table(['Cambio del dataset', 'Valor'], comp.get('dataset', {}).items())]

    body.append('<details class="tech"><summary>Detalles técnicos avanzados</summary>')
    for key, label in [('model', 'Modelo'), ('training', 'Configuración de entrenamiento'),
                        ('hardware', 'Hardware'), ('base_model', 'Modelo base'), ('artifacts', 'Artefactos'),
                        ('losses', 'Pérdidas de la última época')]:
        body += [f'<h3>{label}</h3>', table(['Campo', 'Valor'], report.get(key, {}).items())]
    body += ['<h3>Comparación completa</h3>',
             '<p class="explain">Incluye métricas que este pipeline no calcula (N/A); no significa un '
             'resultado de cero — ver Glosario.</p>',
             table(['Métrica', 'Base', 'Nuevo', 'Diferencia absoluta'],
                   [(key, row.get('base'), row.get('new'), row.get('difference'))
                    for key, row in metrics.items()]),
             '</details>']

    body += ['<h2>Datos de las curvas</h2><details><summary>Ver evaluaciones</summary>',
             table(['Época', 'mAP 50:95', 'AP50', 'AP75', 'AR'],
                   [(r.get(k) for k in ('epoch', 'map_50_95', 'ap50', 'ap75', 'average_recall'))
                    for r in evaluations]),
             '</details><details><summary>Ver historial de entrenamiento</summary>',
             table(['Época', 'Total', 'IoU', 'Clasificación', 'Confianza', 'L1', 'LR'],
                   [(r.get(k) for k in ('epoch', 'total_loss', 'iou_loss', 'cls_loss', 'conf_loss', 'l1_loss',
                                        'learning_rate')) for r in history]),
             '</details>']

    body.append(glossary_section())
    body.append('</main>')

    css = '''body{margin:0;background:#f1f5f9;color:#17243b;font:16px system-ui,sans-serif}header{background:#14243d;color:white;padding:40px max(24px,calc((100% - 1160px)/2))}header p{color:#cbd5e1}h1{font-size:36px}main{max-width:1160px;margin:auto;padding:24px}h2{margin-top:36px}section{background:white;border:1px solid #dbe3ed;border-radius:12px;padding:20px;min-width:0}section h3{margin-top:0}strong{font-size:30px}.cards,.charts{display:grid;gap:16px;grid-template-columns:repeat(2,minmax(0,1fr))}.cards{grid-template-columns:repeat(4,minmax(0,1fr))}.notice{background:#fff3d6;border-left:4px solid #c78100;padding:18px}svg{width:100%;height:auto}svg text{font:13px system-ui;fill:#475569}.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:14px}.scroll{overflow-x:auto;background:white;border-radius:8px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #e2e8f0;overflow-wrap:anywhere}th{background:#e8eef6}summary{cursor:pointer;padding:14px}p{line-height:1.6}.lead{font-size:17px;line-height:1.7;margin-top:4px}.hint{cursor:help;color:#64748b;font-size:13px;border-bottom:1px dotted #94a3b8}.explain{color:#52525b;font-size:14px;line-height:1.55}.friendly{color:#64748b;font-size:13px;font-weight:400}.stat .raw{display:block;font-size:12px;color:#64748b;margin-top:4px}.delta{display:inline-block;margin-top:10px;font-size:13px;font-weight:600;padding:3px 9px;border-radius:999px}.delta.up{color:#006300;background:#e7f6e7}.delta.down{color:#a3241c;background:#fdecec}.delta.flat,.delta.unknown{color:#52514e;background:#eef1f5;font-weight:400}.bars{display:flex;flex-direction:column;gap:18px}.bar-row{border-bottom:1px solid #e2e8f0;padding-bottom:14px}.bar-row:last-child{border-bottom:none}.bar-label{margin:0 0 8px;font-weight:600}.bar-line{display:flex;align-items:center;gap:10px;margin-bottom:6px}.bar-tag{width:44px;flex:none;font-size:12px;color:#64748b}.bar-track{flex:1;background:#eef1f5;border-radius:999px;height:16px;overflow:hidden}.bar{height:100%;border-radius:999px}.bar.base{background:#94a3b8}.bar.new{background:#2563eb}.bar-value{width:64px;flex:none;font-size:13px;font-weight:600;text-align:right}.no-base{color:#94a3b8;font-size:13px;margin:0 0 6px}details.tech summary{font-weight:700;font-size:18px}dl.glossary{display:grid;grid-template-columns:max-content 1fr;gap:8px 20px;font-size:14px;margin-top:12px}dl.glossary dt{font-weight:600;white-space:nowrap}dl.glossary dd{margin:0;color:#334155;line-height:1.5}@media(max-width:700px){.cards,.charts{grid-template-columns:1fr}main{padding:16px}dl.glossary{grid-template-columns:1fr}dl.glossary dd{margin-bottom:8px}}@media print{body{background:white}header{background:white;color:black;padding:0}header p{color:black}section{break-inside:avoid}.charts{display:block}section{margin-bottom:16px}details{display:block}.scroll{overflow:visible}}'''
    return f'<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{css}</style></head><body>{"".join(body)}</body></html>'
