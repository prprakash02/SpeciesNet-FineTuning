# SpeciesNet Fine-Tuning (15 Species)

This repository provides a PyTorch-based pipeline for fine-tuning [SpeciesNet](https://github.com/google/cameratrapai/tree/main) on a custom subset of 15 species. It includes tools to remap and modify the classifier head, handle partially overlapping label sets, and perform lightweight transfer learning using selected EfficientNetV2-M layers.

---

## Contents

* `build_head_15.py`
  Reconstructs the classifier head for 15 target species. Reuses weights for known species and initializes the rest randomly.

* `finetune_head_15.py`
  End-to-end training loop for fine-tuning the model on the new 15-class dataset. Includes validation, early stopping, and test evaluation.

* `labels_known.txt` and `labels_new.txt`
  Lists of known and new species (in `Genus_species` format) used to construct the new classifier.

* `SpeciesNetClassifier` (imported from official repo)
  Handles preprocessing, label mapping, and inference logic.

---

## Fine-Tuning Strategy

* **Backbone**: EfficientNetV2-M (pretrained on SpeciesNet)
* **Updated classifier head**: 15-class linear layer
* **Trainable components**:

  * Final classifier weights/biases
  * Top convolution layer
  * Final MBConv stage (`block7`)
  * Can also be modified with more trainable parameters
* **Frozen components**: All other parameters
* **Early stopping**: After 5 epochs without validation improvement
* **Learning rate scheduling**: `ReduceLROnPlateau`

---

## Quickstart

### 1. Prepare Environment

```bash
pip install -r requirements.txt
```


### 2. Build New Classifier Head

```bash
python build_head_15.py
```

This generates:

* `speciesnet_15_2init.pth` (new head checkpoint)
* `labels_15.txt` (new label map)

### 3. Fine-Tune Model

```bash
python finetune_head_15.py
```

Outputs:

* `finetuned_models/best_model.pth`
* `confusion_matrix.png`

---

## Data Format

Directory layout should follow:

```
data/
└── train/
    ├── Class 1/
    │   ├── img1.jpg
    │   └── ...
    ├── Class 2/
    └── ...
```

Each subfolder name must match a class label in `labels_known.txt` or `labels_new.txt`.

---

## Outputs

* Training/validation accuracy per epoch
* Final test classification report
* Confusion matrix (saved as PNG)
* Best model weights with optimizer state

---

## Notes

* Images are resized to 480×480 with random augmentations during training (to suit the classifier architecture).
* Invalid samples (e.g., failed loads, wrong shapes) are zero-filled but tracked.
* Classifier head orientation is corrected automatically depending on `torch.load()` format.
