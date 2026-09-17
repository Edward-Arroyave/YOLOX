"""Evaluate final and base checkpoints on the same labeled holdout, before publication."""

import argparse
import copy
import json
from pathlib import Path

from yolox.model_card import dataset_snapshot, digest
from yolox.model_report_html import render_html


def inspect_test(exp):
    snapshot = dataset_snapshot(exp)
    test = snapshot['splits'].get('test')
    root = Path(exp.data_dir)
    if not test:
        if (root / exp.test_image_dir).exists() or (root / exp.annotations_dir / exp.test_ann).exists():
            raise ValueError('Test presente pero sus anotaciones no son válidas o repiten train/val.')
        return snapshot, None
    if not test['images'] or not all(test['image_hashes']):
        raise ValueError('Test vacío o con imágenes faltantes.')
    data = json.loads(Path(test['annotations']).read_text(encoding='utf-8'))
    if len(data['categories']) != exp.num_classes:
        raise ValueError('Las clases de test no coinciden con num_classes del experimento.')
    categories = sorted((c['id'], c['name']) for c in data['categories'])
    overlap = {}
    for split in ('train', 'val'):
        other = snapshot['splits'].get(split)
        if other:
            other_data = json.loads(Path(other['annotations']).read_text(encoding='utf-8'))
            if sorted((c['id'], c['name']) for c in other_data['categories']) != categories:
                raise ValueError(f'Las categorías de test no coinciden con {split}.')
            overlap[split] = len(set(test['image_hashes']) & set(other['image_hashes']))
    return snapshot, overlap


def metric_rows(base, new):
    return {key: {'base': (base or {}).get(key), 'new': value,
                  'difference': value - base[key] if base and base.get(key) is not None and value is not None else None}
            for key, value in new.items() if key != 'per_class'}


def evaluate_checkpoints(exp, checkpoint, base_checkpoint, batch_size, half):
    import torch
    # Use ordinary labeled COCO evaluation, not COCO's unlabeled test-dev export.
    test_exp = copy.deepcopy(exp)
    test_exp.val_ann = exp.test_ann
    test_exp.val_image_dir = exp.test_image_dir
    evaluator = test_exp.get_evaluator(batch_size, False)
    results = {}
    for label, path in (('base', base_checkpoint), ('new', checkpoint)):
        if not path:
            results[label] = None
            continue
        model = copy.deepcopy(exp.get_model()).cuda().float()
        state = torch.load(path, map_location='cpu', weights_only=False)
        # Partial loading would compare random or incompatible detection heads.
        model.load_state_dict(state.get('model', state), strict=True)
        evaluator.evaluate(model, distributed=False, half=half)
        results[label] = copy.deepcopy(evaluator.last_metrics)
        ms = results[label].get('inference_ms')
        results[label]['fps'] = 1000 / ms if ms and ms > 0 else None
        del model, state
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--base-checkpoint')
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()
    from yolox.exp import get_exp
    exp = get_exp(args.exp_file, None)
    directory = Path(args.report_dir)
    report = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    # Preserve effective training overrides (architecture and inference settings).
    for key in ('depth', 'width', 'act', 'test_size', 'test_conf', 'nmsthre'):
        if report.get('training', {}).get(key) is not None:
            setattr(exp, key, report['training'][key])
    snapshot, overlap = inspect_test(exp)
    snapshot['version'] = report['dataset'].get('version')
    report['dataset'] = snapshot
    result = {'status': 'unavailable', 'reason': 'No se encontró un conjunto test independiente.'}
    if overlap is not None:
        results = evaluate_checkpoints(exp, args.ckpt, args.base_checkpoint, args.batch_size, args.fp16)
        result = {'status': 'completed', **results, 'metrics': metric_rows(results['base'], results['new']),
                  'overlap_images': overlap, 'annotation_sha256': snapshot['splits']['test']['annotation_sha256'],
                  'checkpoints': {label: {'path': str(path), 'sha256': digest(path)} if path else None
                                  for label, path in (('base', args.base_checkpoint), ('new', args.ckpt))},
                  'settings': {'size': exp.test_size, 'confidence': exp.test_conf, 'nms_iou': exp.nmsthre,
                               'batch_size': args.batch_size, 'fp16': args.fp16}}
    report['test_evaluation'] = result
    # Finish inference and rendering before replacing either deliverable.
    outputs = {'metrics.json': json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
               'model_report.html': render_html(report)}
    for name, content in outputs.items():
        temporary = directory / (name + '.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(directory / name)
    print('Evaluación test:', result['status'])


if __name__ == '__main__':
    main()
