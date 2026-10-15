"""Extract validation-set features with an explicitly provisioned SlerpFace model.

Run from the project root: ``python -m extract_template.verification --help``.
This exports research features; it does not enroll protected credentials.
"""
import argparse
from pathlib import Path
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True, help='Compatible model checkpoint path')
    parser.add_argument('--backbone', default='IR_50',
                        choices=['IR_18', 'IR_34', 'IR_50', 'IR_100', 'IR_152', 'IR_200'])
    parser.add_argument('--gpu_ids', default='0', type=int, help='Single nonnegative GPU device index')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_root', required=True, help='Directory containing val/ annotations and image folders')
    parser.add_argument('--output_dir', default='./templates', help='Research feature output directory')
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error('--batch_size must be greater than zero')
    if args.gpu_ids < 0:
        parser.error('--gpu_ids must be nonnegative')
    return args


def _extract(args):
    checkpoint = Path(args.model_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'model checkpoint not found: {checkpoint}')
    if not (Path(args.data_root) / 'val').is_dir():
        raise FileNotFoundError(f'validation data directory not found: {Path(args.data_root) / "val"}')
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == 'torch':
            raise ImportError(
                'Template extraction requires PyTorch and the pinned local TFace backbone '
                'compatible with your checkpoint. See README.md model setup.'
            ) from exc
        raise
    import numpy as np
    from extract_template.utils import get_templates
    from extract_template.data_loader import get_val_data_from_folder
    from modules.model import SlerpFace

    device = torch.device(f'cuda:{args.gpu_ids}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = SlerpFace(input_size=[112, 112],
                      num_layers=int(args.backbone.split('_')[-1]), group_size=16).to(device)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    datasets = get_val_data_from_folder(args.data_root)
    dataset_names = ('LFW', 'CFP', 'AGEDB', 'CPLFW', 'CALFW')
    if not any(len(data) for data in datasets[:5]):
        raise ValueError('no image pairs found in validation annotations; no features were written')

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in zip(dataset_names, datasets[:5]):
        if not len(data):
            print(f'Skipping {name}: no image pairs')
            continue
        gallery, query = get_templates(512, args.batch_size, model, data,
                                       name=name, gpu_ids=args.gpu_ids, device=device)
        for kind, templates in (('gallery', gallery), ('query', query)):
            np.save(output / f'{name}_{kind}_templates.npy', templates['group_feature'])
            np.save(output / f'{name}_{kind}_attention_map.npy', templates['attention_map'])
        print(f'{name}: extracted {len(gallery["group_feature"])} research pairs')
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        return _extract(args)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f'Template extraction unavailable: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
