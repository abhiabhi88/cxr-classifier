# Chest X-ray Classification

Binary classifier (e.g. Normal vs Pneumonia) with a **patient-level split**, ResNet18, class weighting, and early stopping.

> Research/learning use only. Not a medical device; not for clinical decisions.

## Setup (local)

```bash
pip install -r requirements.txt
```

CSV must have one row per image:

```text
image_path,label,patient_id
data/img_001.png,0,P001
data/img_002.png,1,P002
```

`label`: 0 = negative, 1 = positive. `patient_id` is required so the same patient never appears in more than one split.

```bash
python cxr_classifier.py --csv labels.csv --epochs 15
python cxr_classifier.py --csv labels.csv --scratch   # random init baseline
```

The best validation checkpoint is saved as `best_model.pt`. Test metrics are reported once at the end.

## Docker

Build:

```bash
docker build -t cxr-classifier .
```

Train (mount a `data/` folder that contains `labels.csv` and the images it points to):

```bash
docker run --rm -v ${PWD}/data:/data -v ${PWD}/outputs:/outputs cxr-classifier --csv /data/labels.csv --epochs 15 --out /outputs/best_model.pt
```

On Windows PowerShell:

```powershell
docker run --rm -v ${PWD}/data:/data -v ${PWD}/outputs:/outputs cxr-classifier --csv /data/labels.csv --epochs 15 --out /outputs/best_model.pt
```

Or:

```bash
docker compose run --rm train
```

GPU (NVIDIA Container Toolkit):

```bash
docker run --rm --gpus all -v ${PWD}/data:/data -v ${PWD}/outputs:/outputs cxr-classifier --csv /data/labels.csv --epochs 15 --out /outputs/best_model.pt
```

## GitHub

Push this folder to GitHub, then anyone can clone and run the same Docker commands.
