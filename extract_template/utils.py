import numpy as np
import torch


def l2_norm(input, axis=1):
    """l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def process_batch(batch, backbone, device):
    """Process a single batch of data
    Args:
        batch: Input data batch
        backbone: Model
        device: GPU device
    Returns:
        tuple: (group_features, attention_maps, embeddings)
    """
    # Move data to GPU
    batch = batch.to(device)
    flipped = torch.flip(batch, dims=[3])
    
    # Generate group features
    group_features = backbone.gen_group_feature(batch, flip=True)
    attention_maps = backbone.gen_weight_map(group_features)
    
    # Generate vector features
    vector_features = (
        backbone.gen_vector_feature(batch).cpu() + 
        backbone.gen_vector_feature(flipped).cpu()
    )
    embeddings = l2_norm(vector_features)
    
    return (
        group_features.cpu().detach().numpy(),
        attention_maps.cpu().detach().numpy(),
        embeddings
    )

def get_templates(
    embedding_size,
    batch_size,
    backbone,
    carray,
    group_width=7,
    group_size=16,
    name='LFW',
    gpu_ids=0,
    device=None
):
    """Extract registration and matching templates for SlerpFace model
    Args:
        embedding_size: Feature dimension
        batch_size: Batch size
        backbone: Model
        carray: Input data
        group_width: Width of group
        group_size: Size of group
        name: Dataset name
        gpu_ids: GPU ID
    Returns:
        tuple: (gallery_templates_dict, query_templates_dict)
    """
    backbone.eval()
    if device is None:
        device = torch.device(f'cuda:{gpu_ids}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize storage arrays
    embeddings = np.zeros([len(carray), embedding_size])
    attention_maps = np.zeros([len(carray), group_width, group_width, 1])
    group_features = np.zeros([len(carray), group_size, group_width, group_width])
    
    # Process all batches including remainder
    with torch.no_grad():
        n = len(carray)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            print("S", end='')
            batch = torch.tensor(carray[start:end])
            
            # Process current batch
            g_feat, a_maps, emb = process_batch(batch, backbone, device)
            
            # Store results
            group_features[start:end] = g_feat
            attention_maps[start:end] = a_maps
            embeddings[start:end] = emb
            
            print('=', end='')
    
    # Separate gallery and query templates
    gallery_templates = {
        'vector_feature': embeddings[0::2],
        'group_feature': group_features[0::2],
        'attention_map': attention_maps[0::2]
    }
    
    query_templates = {
        'vector_feature': embeddings[1::2],
        'group_feature': group_features[1::2],
        'attention_map': attention_maps[1::2]
    }
    
    return gallery_templates, query_templates
