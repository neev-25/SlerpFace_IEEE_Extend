"""
SlerpFace - Test with Your Own Images
======================================
Usage:
  # Verify if two images are the SAME person:
  python test_my_image.py --img1 photo1.jpg --img2 photo2.jpg

  # Just extract and show the encrypted template for one image:
  python test_my_image.py --img1 photo1.jpg
"""
import argparse
import numpy as np
import torch
from PIL import Image
from tasks.slerpface.modules.model import SlerpFace


# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = r".\tasks\slerpface\ckpt\Backbone_Epoch_24_checkpoint.pth"
INPUT_SIZE  = [112, 112]
GROUP_SIZE  = 16
SLERP_ALPHA = 0.9
DROP_RATE   = 0.5


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_image(path):
    """Load, resize and normalize an image → tensor (1, 3, 112, 112)"""
    img = Image.open(path).convert("RGB").resize((112, 112))
    arr = np.array(img, dtype=np.float32)
    arr = (arr - 127.5) / 128.0          # normalize to [-1, 1]
    arr = arr.transpose(2, 0, 1)          # HWC → CHW
    return torch.tensor(arr).unsqueeze(0) # add batch dim


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SlerpFace(input_size=INPUT_SIZE, num_layers=50, group_size=GROUP_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
    model.eval()
    print(f"Model loaded on {device}")
    return model, device


def extract_features(model, device, img_tensor):
    """Returns (group_feature, vector_feature)"""
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        group_feat = model.gen_group_feature(img_tensor, flip=True)     # (1, 16, 7, 7)
        vec_feat    = model.gen_vector_feature(img_tensor)               # (1, 512)
    return group_feat.cpu().numpy(), vec_feat.cpu().numpy()


def slerp_encrypt(group_feat, alpha=SLERP_ALPHA, drop_rate=DROP_RATE):
    """Apply SlerpFace encryption to group features"""
    feat = group_feat[0].transpose(1, 2, 0).reshape(-1, GROUP_SIZE)     # (49, 16)
    key  = np.random.randn(*feat.shape)

    norm_f = np.linalg.norm(feat, axis=1, keepdims=True)
    norm_k = np.linalg.norm(key,  axis=1, keepdims=True)
    f_n    = feat / (norm_f + 1e-8)
    k_n    = key  / (norm_k + 1e-8)

    dot    = np.clip((f_n * k_n).sum(axis=1), -1, 1)
    theta  = np.arccos(dot).reshape(-1, 1)
    sin_t  = np.sin(theta) + 1e-8

    encrypted = (np.sin((1 - alpha) * theta) / sin_t * f_n +
                 np.sin(alpha * theta)        / sin_t * k_n)

    # Random dropout
    n_drop = int(encrypted.shape[0] * GROUP_SIZE * drop_rate)
    mask   = np.random.choice(encrypted.size, n_drop, replace=False)
    encrypted.ravel()[mask] = 0

    return encrypted


def cosine_similarity(v1, v2):
    v1, v2 = v1.ravel(), v2.ravel()
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test SlerpFace with your own images")
    parser.add_argument("--img1", required=True, help="Path to first image")
    parser.add_argument("--img2", default=None,  help="Path to second image (for verification)")
    args = parser.parse_args()

    model, device = load_model()

    # ── Image 1 ────────────────────────────────────────────────────────────────
    print(f"\n📸 Processing: {args.img1}")
    t1 = load_image(args.img1)
    g1, v1 = extract_features(model, device, t1)
    enc1   = slerp_encrypt(g1)
    print(f"  Raw group feature shape:      {g1.shape}")
    print(f"  Raw vector feature shape:     {v1.shape}")
    print(f"  Encrypted template shape:     {enc1.shape}")
    print(f"  Encryption non-zero elements: {np.count_nonzero(enc1)}/{enc1.size}")

    if args.img2 is None:
        print("\n✅ Done! (pass --img2 to compare two images)")
        return

    # ── Image 2 + Verification ─────────────────────────────────────────────────
    print(f"\n📸 Processing: {args.img2}")
    t2 = load_image(args.img2)
    g2, v2 = extract_features(model, device, t2)
    enc2   = slerp_encrypt(g2)

    # Raw similarity (unencrypted vectors)
    raw_sim = cosine_similarity(v1, v2)

    # Encrypted group similarity
    enc_sim = cosine_similarity(enc1, enc2)

    THRESHOLD = 0.30  # typical threshold for face verification
    is_same = raw_sim > THRESHOLD

    print("\n" + "="*50)
    print("🔍 VERIFICATION RESULT")
    print("="*50)
    print(f"  Raw cosine similarity:       {raw_sim:.4f}")
    print(f"  Encrypted group similarity:  {enc_sim:.4f}")
    print(f"  Threshold:                   {THRESHOLD}")
    print(f"\n  {'✅ SAME PERSON' if is_same else '❌ DIFFERENT PERSON'}")
    print(f"  Confidence: {abs(raw_sim - THRESHOLD):.3f} away from threshold")
    print("="*50)


if __name__ == "__main__":
    main()
