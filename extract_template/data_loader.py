"""
Replacement for missing test.utils.get_val_data_from_bin
Loads face verification datasets from image folders + annotation txt files.
Format of ann.txt: `label img1.bmp img2.bmp` per line
Returns: numpy array of images, shape (N, 3, 112, 112) in range [-1, 1]
         and pair issame labels
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image


def load_dataset_from_folder(data_root, img_folder, ann_file):
    """
    Load image pairs listed as ``0|1 image1 image2`` in an annotation file.

    Image names are relative to ``img_folder``. Dataset-prefixed names such as
    ``faces/a.jpg`` are also accepted when ``img_folder`` is ``faces``. Paths
    must stay inside that folder. Blank lines are ignored; malformed rows are
    errors rather than silently dropping pairs. Empty input returns empty arrays.
    """
    root = Path(data_root).resolve()
    image_root = (root / img_folder).resolve()
    ann_path = root / ann_file
    
    images = []
    issame_list = []
    
    with ann_path.open('r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    for line_number, line in enumerate(lines, 1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 3:
            raise ValueError(f"{ann_path}:{line_number}: expected '0|1 image1 image2'")
        label, img1_rel, img2_rel = parts
        if label not in {'0', '1'}:
            raise ValueError(f"{ann_path}:{line_number}: pair label must be 0 or 1")

        for img_rel in [img1_rel, img2_rel]:
            relative = Path(img_rel)
            if relative.is_absolute() or '..' in relative.parts or relative.drive:
                raise ValueError(f"{ann_path}:{line_number}: image path must stay inside {image_root}")
            try:
                folder_prefix = image_root.relative_to(root)
            except ValueError:
                folder_prefix = None
            if (folder_prefix and folder_prefix.parts and
                    relative.parts[:len(folder_prefix.parts)] == folder_prefix.parts):
                relative = Path(*relative.parts[len(folder_prefix.parts):])
            img_path = (image_root / relative).resolve()
            if not img_path.is_relative_to(image_root):
                raise ValueError(f"{ann_path}:{line_number}: image path must stay inside {image_root}")
            if not img_path.is_file():
                raise FileNotFoundError(f"{ann_path}:{line_number}: image not found: {img_path}")
            with Image.open(img_path) as image:
                img = image.convert('RGB').resize((112, 112))
                arr = np.array(img, dtype=np.float32)
            # Normalize to [-1, 1] and transpose to (C, H, W)
            arr = (arr - 127.5) / 127.5
            arr = arr.transpose(2, 0, 1)
            images.append(arr)
        issame_list.append(label == '1')
    
    images = (np.stack(images, axis=0) if images else
              np.empty((0, 3, 112, 112), dtype=np.float32))
    issame = np.array(issame_list, dtype=bool)
    return images, issame


def get_val_data_from_folder(data_root):
    """
    Load all five datasets from the val/ subfolder.
    Returns: (lfw_data, cfp_data, agedb_data, cplfw_data, calfw_data,
               lfw_issame, cfp_issame, agedb_issame, cplfw_issame, calfw_issame)
    """
    val_dir = os.path.join(data_root, 'val')
    
    datasets = {
        'lfw':      ('lfw_112x112',     'lfw_ann.txt'),
        'cfp':      ('cfp_fp_112x112',  'cfp_fp_ann.txt'),   # may not exist
        'agedb':    ('agedb_30_112x112', 'agedb_30_ann.txt'),
        'cplfw':    ('cplfw_112x112',   'cplfw_ann.txt'),
        'calfw':    ('calfw_112x112',   'calfw_ann.txt'),
    }
    
    results = {}
    for name, (folder, ann) in datasets.items():
        ann_path = os.path.join(val_dir, ann)
        if os.path.exists(ann_path):
            print(f"Loading {name} from {ann}...")
            imgs, issame = load_dataset_from_folder(val_dir, folder, ann)
            results[name] = (imgs, issame)
            print(f"  Loaded {len(issame)} pairs ({imgs.shape[0]} images)")
        else:
            print(f"Warning: {name} annotation file not found at {ann_path}, skipping.")
            results[name] = (np.zeros((0, 3, 112, 112), dtype=np.float32), np.array([], dtype=bool))
    
    lfw    = results['lfw'][0]
    cfp    = results['cfp'][0]
    agedb  = results['agedb'][0]
    cplfw  = results['cplfw'][0]
    calfw  = results['calfw'][0]
    lfw_issame    = results['lfw'][1]
    cfp_issame    = results['cfp'][1]
    agedb_issame  = results['agedb'][1]
    cplfw_issame  = results['cplfw'][1]
    calfw_issame  = results['calfw'][1]
    
    return (lfw, cfp, agedb, cplfw, calfw,
            lfw_issame, cfp_issame, agedb_issame, cplfw_issame, calfw_issame)
