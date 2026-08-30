"""
Replacement for missing test.utils.get_val_data_from_bin
Loads face verification datasets from image folders + annotation txt files.
Format of ann.txt: `label img1.bmp img2.bmp` per line
Returns: numpy array of images, shape (N, 3, 112, 112) in range [-1, 1]
         and pair issame labels
"""
import os
import numpy as np
from PIL import Image


def load_dataset_from_folder(data_root, img_folder, ann_file):
    """
    Load all images listed in ann_file, return as numpy array (N, 3, 112, 112)
    and the issame labels.
    """
    ann_path = os.path.join(data_root, ann_file)
    
    images = []
    issame_list = []
    
    with open(ann_path, 'r') as f:
        lines = f.read().strip().split('\n')
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        label, img1_rel, img2_rel = parts
        issame_list.append(int(label) == 1)
        
        for img_rel in [img1_rel, img2_rel]:
            img_path = os.path.join(data_root, img_rel)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((112, 112))
            arr = np.array(img, dtype=np.float32)
            # Normalize to [-1, 1] and transpose to (C, H, W)
            arr = (arr - 127.5) / 128.0
            arr = arr.transpose(2, 0, 1)
            images.append(arr)
    
    images = np.stack(images, axis=0)  # (N*2, 3, 112, 112)
    issame = np.array(issame_list, dtype=bool)
    return images, issame


def get_val_data_from_folder(data_root):
    """
    Load all 4 datasets from the val/ subfolder.
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
