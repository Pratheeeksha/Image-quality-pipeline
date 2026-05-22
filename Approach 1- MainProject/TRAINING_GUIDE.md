# Training Guide (High Accuracy Setup)

This guide helps you retrain the classifier to reduce false positives (e.g., BAD predicted as GOOD).

## 1) Where to get images

Use a mix of:

- Your own mission data (best source, same drone/camera conditions)
- Public aerial image sources:
  - [DroneDeploy Public Datasets](https://www.dronedeploy.com/resources/)
  - [OpenAerialMap](https://openaerialmap.org/)
  - [Kaggle drone/aerial datasets](https://www.kaggle.com/)
  - [xView / DOTA](https://xviewdataset.org/) (for aerial variability)

Important: Public data usually does not have your 3 classes, so you must manually label quality as `GOOD`, `RECOVERABLE`, `BAD`.

## 2) Recommended dataset size

Minimum acceptable:

- 500 images per class (`GOOD`, `RECOVERABLE`, `BAD`)
- Total ~1500 images

Good target:

- 1500 to 3000 images per class
- Total ~4500 to 9000 images

Very important:

- Keep classes balanced (difference not more than ~20%)
- Include hard examples (motion blur, haze, overexposed, shadows, low texture)

## 3) Folder structure

Use this exact structure:

```text
data/
  train/
    BAD/
    GOOD/
    RECOVERABLE/
  val/
    BAD/
    GOOD/
    RECOVERABLE/
  test/
    BAD/
    GOOD/
    RECOVERABLE/
```

Split recommendation:

- Train: 70%
- Val: 15%
- Test: 15%

## 4) Training hyperparameters (starting point)

For your project:

- Epochs: 20 to 30
- Batch size: 16 (or 8 if GPU memory is low)
- Learning rate: 1e-4
- Weight decay: 1e-4

If your dataset is under 2000 total images:

- Epochs: 30 to 40
- Keep stronger augmentation enabled

## 5) Run training

From project root:

```bash
python self_learning/train_model.py --epochs 25 --batch-size 16 --lr 1e-4
```

Outputs:

- Model: `classifier/efficientnet_b0_quality.pt`
- Class map: `classifier/class_name.json`
- Metrics report: `data/processed/training_report.json`

## 6) How to improve accuracy quickly

- Increase `RECOVERABLE` diversity (this class is usually hardest)
- Remove wrong labels; bad labels hurt more than small dataset size
- Use mission-like images from your real drone/camera
- Track macro-F1, not only accuracy
- If BAD -> GOOD false positives remain high, raise GOOD acceptance threshold in Streamlit (e.g., 0.70 to 0.80)

## 7) Expo-ready validation checklist

- Check confusion matrix in `training_report.json`
- Ensure BAD recall is high (you do not want bad images accepted)
- Run at least one full mission batch through Streamlit and verify recommendation output
