"""Portable HTML report with embedded vector charts; no network dependencies."""

from html import escape
import math


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else "N/A"
    return escape(str(value))


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(
        f'<th>{escape(h)}</th>' for h in headers
    ) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{fmt(v)}</td>' for v in row) + '</tr>' for row in rows
    ) + '</tbody></table></div>'


def chart(title, records, keys, unit, best_epoch=None):
    colors = ['#2563eb', '#059669', '#d97706', '#db2777', '#7c3aed', '#475569']
    series = []
    for key, label in keys:
        points = [(r.get('epoch'), r.get(key)) for r in records]
        points = [(x, y) for x, y in points if isinstance(x, (int, float))
                  and isinstance(y, (int, float)) and math.isfinite(x) and math.isfinite(y)]
        if points:
            series.append((label, points))
    if not series:
        return f'<section><h3>{escape(title)}</h3><p>No hay historial disponible para esta curva.</p></section>'
    xs = [x for _, points in series for x, _ in points]
    ys = [y for _, points in series for _, y in points]
    xmin, xmax = min(xs), max(xs)
    ymax = max(ys) * 1.08 or 1
    xmap = lambda x: 65 + (x - xmin) / (xmax - xmin or 1) * 665
    ymap = lambda y: 275 - y / ymax * 235
    svg = [f'<svg viewBox="0 0 780 340" role="img" aria-label="{escape(title, quote=True)}">',
           f'<text x="65" y="20">{escape(unit)}</text>']
    for i in range(6):
        y = ymax * i / 5
        svg += [f'<path d="M65 {ymap(y)} H730" stroke="#e2e8f0"/>',
                f'<text x="58" y="{ymap(y)+4}" text-anchor="end">{y:.3g}</text>']
    for x in sorted(set((xmin, (xmin + xmax) / 2, xmax))):
        svg.append(f'<text x="{xmap(x)}" y="296" text-anchor="middle">{x:g}</text>')
    if best_epoch is not None and xmin <= best_epoch <= xmax:
        x = xmap(best_epoch)
        svg.append(f'<path d="M{x} 35 V275" stroke="#64748b" stroke-dasharray="5 5"/>')
    for idx, (label, points) in enumerate(series):
        color = colors[idx % len(colors)]
        coords = ' '.join(f'{xmap(x):.2f},{ymap(y):.2f}' for x, y in points)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in points:
            svg.append(f'<circle cx="{xmap(x)}" cy="{ymap(y)}" r="2.5" fill="{color}"><title>{escape(label)} · época {x}: {y:.6g}</title></circle>')
    svg += ['<text x="395" y="325" text-anchor="middle">Época</text>', '</svg>']
    legend = ' '.join(f'<span style="color:{colors[i % len(colors)]}">● {escape(label)}</span>'
                      for i, (label, _) in enumerate(series))
    return f'<section><h3>{escape(title)}</h3>{"".join(svg)}<div class="legend">{legend}</div></section>'


def render_html(report):
    model, best = report.get('model', {}), report.get('best') or {}
    comp = report.get('comparison', {})
    history = report.get('training_history', [])
    evaluations = report.get('evaluations', [])
    title = f"{model.get('project', 'YOLOX')} · {model.get('version', 'N/A')}"
    body = [f'<header><p>INFORME DE ENTRENAMIENTO · YOLOX</p><h1>{escape(title)}</h1>',
            f'<p>Mejor checkpoint: época {fmt(best.get("epoch"))}. Finalizado: {fmt(report.get("completed_at"))}</p></header>',
            '<main><h2>Resumen</h2><div class="cards">']
    for label, value in [('mAP 50:95', best.get('map_50_95')), ('AP50', best.get('ap50')),
                         ('AP75', best.get('ap75')), ('Average Recall', best.get('average_recall'))]:
        body.append(f'<section><p>{label}</p><strong>{fmt(value)}</strong></section>')
    body.append('</div>')
    if not comp.get('same_validation'):
        body.append('<p class="notice">No se puede concluir que el nuevo modelo mejora o empeora: no se ha verificado la misma validación para ambos modelos. La comparación siguiente es histórica; no corresponde a una prueba independiente común.</p>')
    body += ['<h2>Curvas del entrenamiento</h2><p>Resultados reales, sin suavizado. AP y AR en escala 0–1. Pérdidas: promedio por iteración de cada época, proceso rank 0. La línea vertical señala la mejor época. Pasa el cursor sobre los puntos para ver valores.</p>',
             '<div class="charts">',
             chart('Calidad en validación', evaluations, [('map_50_95', 'mAP 50:95'), ('ap50', 'AP50'), ('ap75', 'AP75'), ('average_recall', 'AR')], 'AP / AR', best.get('epoch')),
             chart('Pérdida total', history, [('total_loss', 'Total')], 'Loss', best.get('epoch')),
             chart('Componentes de la pérdida', history, [(k, v) for k, v in [('iou_loss', 'IoU'), ('cls_loss', 'Clasificación'), ('conf_loss', 'Confianza'), ('l1_loss', 'L1')]], 'Loss', best.get('epoch')),
             chart('Learning rate', history, [('learning_rate', 'LR al cierre de época')], 'Learning rate'), '</div>',
             '<h2>Comparación con el modelo base</h2>',
             table(['Métrica', 'Base', 'Nuevo', 'Diferencia absoluta'],
                   [(key, row.get('base'), row.get('new'), row.get('difference')) for key, row in comp.get('metrics', {}).items()]),
             '<h2>Calidad por clase · mejor checkpoint</h2>',
             table(['Clase', 'AP', 'AR'], [(name, values.get('ap'), values.get('ar')) for name, values in best.get('per_class', {}).items()]),
             '<h2>Dataset</h2>']
    dataset = report.get('dataset', {})
    body.append(table(['Versión', 'Imágenes totales', 'Clases'], [(dataset.get('version'), dataset.get('total_images'), dataset.get('num_classes'))]))
    body.append(table(['Conjunto', 'Imágenes', 'Anotaciones por clase'], [(name, (values or {}).get('images'), (values or {}).get('annotations_per_class')) for name, values in dataset.get('splits', {}).items()]))
    body.append(table(['Cambio del dataset', 'Valor'], comp.get('dataset', {}).items()))
    for key, label in [('model', 'Modelo'), ('training', 'Configuración de entrenamiento'), ('hardware', 'Hardware'), ('base_model', 'Modelo base'), ('artifacts', 'Artefactos'), ('losses', 'Pérdidas de la última época')]:
        body += [f'<h2>{label}</h2>', table(['Campo', 'Valor'], report.get(key, {}).items())]
    body += ['<h2>Datos de las curvas</h2><details><summary>Ver evaluaciones</summary>',
             table(['Época', 'mAP 50:95', 'AP50', 'AP75', 'AR'], [(r.get(k) for k in ('epoch', 'map_50_95', 'ap50', 'ap75', 'average_recall')) for r in evaluations]),
             '</details><details><summary>Ver historial de entrenamiento</summary>',
             table(['Época', 'Total', 'IoU', 'Clasificación', 'Confianza', 'L1', 'LR'], [(r.get(k) for k in ('epoch', 'total_loss', 'iou_loss', 'cls_loss', 'conf_loss', 'l1_loss', 'learning_rate')) for r in history]),
             '</details><h2>Cómo interpretar este informe</h2><p>N/A indica información no disponible; no significa cero. No se calcula validation loss, precision, recall puntual ni IoU de detección. IoU loss es una pérdida de entrenamiento. El historial pertenece a esta ejecución; las épocas previas a una reanudación pueden no estar incluidas. Las métricas de referencia y los hashes se conservan en metrics.json.</p></main>']
    css = '''body{margin:0;background:#f1f5f9;color:#17243b;font:16px system-ui,sans-serif}header{background:#14243d;color:white;padding:40px max(24px,calc((100% - 1160px)/2))}header p{color:#cbd5e1}h1{font-size:36px}main{max-width:1160px;margin:auto;padding:24px}h2{margin-top:36px}section{background:white;border:1px solid #dbe3ed;border-radius:12px;padding:20px;min-width:0}section h3{margin-top:0}strong{font-size:30px}.cards,.charts{display:grid;gap:16px;grid-template-columns:repeat(2,minmax(0,1fr))}.cards{grid-template-columns:repeat(4,minmax(0,1fr))}.notice{background:#fff3d6;border-left:4px solid #c78100;padding:18px}svg{width:100%;height:auto}svg text{font:13px system-ui;fill:#475569}.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:14px}.scroll{overflow-x:auto;background:white;border-radius:8px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #e2e8f0;overflow-wrap:anywhere}th{background:#e8eef6}summary{cursor:pointer;padding:14px}p{line-height:1.6}@media(max-width:700px){.cards,.charts{grid-template-columns:1fr}main{padding:16px}}@media print{body{background:white}header{background:white;color:black;padding:0}header p{color:black}section{break-inside:avoid}.charts{display:block}section{margin-bottom:16px}details{display:block}.scroll{overflow:visible}}'''
    return f'<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{css}</style></head><body>{"".join(body)}</body></html>'
