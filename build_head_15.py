import torch
import torch.nn as nn
from pathlib import Path
from speciesnet.classifier import SpeciesNetClassifier

# ---------- user files ----------
KNOWN_FILE = "labels_known.txt"
NEW_FILE   = "labels_new.txt"
OUT_CKPT   = "speciesnet_15_2init.pth"
OUT_LBL    = "labels_15.txt"
MODEL_ID   = "kaggle:google/speciesnet/pyTorch/v4.0.1a"
# ---------------------------------

load = lambda p: [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]

def gs_map(label_dict):
    m = {}
    for idx, rec in label_dict.items():
        f = rec.split(";")
        if len(f) >= 6 and f[4] and f[5]:
            m[f"{f[4].capitalize()}_{f[5]}"] = idx
    return m

def find_classifier(sd, num_cls):
    """Return (W_key, W, axis, bias_key_or_None, bias_or_None)."""
    # weight: largest 2-D with one dim == num_cls
    W_key, W, axis = None, None, None
    for k, v in sd.items():
        if v.ndim == 2 and (v.size(0) == num_cls or v.size(1) == num_cls):
            if W is None or v.numel() > W.numel():
                W_key, W = k, v
    if W is None:
        raise RuntimeError("Classifier weight not found")
    axis = 0 if W.size(0) == num_cls else 1
    # bias (may be missing)
    b_key = b = None
    for k, v in sd.items():
        if v.ndim == 1 and k == 'initializers.onnx_initializer_136':
            b_key, b = k, v
            break
    return W_key, W, axis, b_key, b

def main():
    known, new = load(KNOWN_FILE), load(NEW_FILE)
    labels15 = known + new
    Path(OUT_LBL).write_text("\n".join(labels15))

    clf = SpeciesNetClassifier(MODEL_ID)
    sd = clf.model.state_dict()
    num_cls = len(clf.labels)

    W_key, W_old, axis, b_key, b_old = find_classifier(sd, num_cls)
    in_feats = W_old.size(1 - axis)
    print(f" weight {W_key}  shape={tuple(W_old.shape)}  axis={axis}")
    if b_key:
        print(f" bias   {b_key}  len={b_old.numel()}")
    else:
        print("no bias tensor found, will create fresh bias")

    # Create new classifier weights ********* for 15 outputs ********
    if axis == 0:
        new_W = torch.empty((15, in_feats))
    else:
        new_W = torch.empty((in_feats, 15))
    nn.init.normal_(new_W, 0, 0.01)
    
    if b_key:
        new_b = torch.empty(15)
        nn.init.constant_(new_b, 0)

    m = gs_map(clf.labels)
    for sp in known:
        src = m[sp]
        tgt = labels15.index(sp)
        if axis == 0:
            new_W[tgt] = W_old[src]
        else:
            new_W[:, tgt] = W_old[:, src]
        if b_key:
            new_b[tgt] = b_old[src]

    # Replace tensors in state dict
    sd[W_key] = new_W
    if b_key:
        sd[b_key] = new_b
    
    # Create a new model with modified classifier
    # This is the critical fix: don't try to load into original model
    # Instead, save just the state dict and labels

    # torch.save({
    #     'state_dict': sd,
    #     'labels': labels15
    # }, OUT_CKPT)

    torch.save(sd, OUT_CKPT)
    
    print(f"Saved {OUT_CKPT} and {OUT_LBL}")

if __name__ == "__main__":
    main()