import os
import sys
import cv2
import pandas as pd
import shutil
import subprocess

# ---------------------------------------------------
# Setup Project Root
# ---------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from classifier.predict import predict_image
from metrics.extract_metrics import extract_metrics
from enhancer.enhance import enhance_image

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "incoming_drone_images")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SELF_POOL_DIR = os.path.join(PROJECT_ROOT, "data", "self_learning_pool")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SELF_POOL_DIR, exist_ok=True)

SELF_LEARNING_THRESHOLD = 0.90
GOOD_ACCEPT_CONFIDENCE = 0.65

results = []

print("\n🚀 Starting Drone Image Quality Pipeline...\n")

for img_name in os.listdir(DATA_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(DATA_DIR, img_name)

    print(f"Processing: {img_name}")

    # -------------------------------
    # Stage 1: Classification
    # -------------------------------
    initial_label, conf, probs = predict_image(img_path)

    metrics = extract_metrics(img_path)

    # Guardrail: low-confidence GOOD is treated as RECOVERABLE for re-check.
    effective_label = initial_label
    forced_recheck = False
    if initial_label == "GOOD" and conf < GOOD_ACCEPT_CONFIDENCE:
        effective_label = "RECOVERABLE"
        forced_recheck = True

    final_label = effective_label
    enhanced = False
    recovered_to_good = False
    final_conf = conf  # Track final confidence

    # -------------------------------
    # Stage 2: Enhancement (if RECOVERABLE)
    # -------------------------------
    if effective_label == "RECOVERABLE":
        enhanced_img = enhance_image(img_path, metrics)
        enh_path = os.path.join(OUT_DIR, "enh_" + img_name)
        cv2.imwrite(enh_path, enhanced_img)

        new_label, new_conf, _ = predict_image(enh_path)

        if new_label == "GOOD" and new_conf > 0.6:
            final_label = "GOOD"
            final_conf = new_conf
            enhanced = True
            recovered_to_good = True
            print(f"  ↳ Enhanced → Upgraded to GOOD ({new_conf:.2f})")
        else:
            print(f"  ↳ Enhancement did not improve classification.")

    # -------------------------------
    # Stage 3: Self-Learning Pooling
    # -------------------------------
    if final_conf >= SELF_LEARNING_THRESHOLD and final_label in ["GOOD", "BAD"]:
        pool_class_dir = os.path.join(SELF_POOL_DIR, final_label)
        os.makedirs(pool_class_dir, exist_ok=True)

        dst_path = os.path.join(pool_class_dir, img_name)
        if not os.path.exists(dst_path):
            shutil.copy(img_path, dst_path)
            print(f"  ↳ Added to self-learning pool ({final_label})")

    # -------------------------------
    # Logging
    # -------------------------------
    results.append({
        "image": img_name,
        "initial_label": initial_label,
        "effective_label": effective_label,
        "final_label": final_label,
        "confidence": final_conf,
        "forced_recheck": forced_recheck,
        "recovered_to_good": recovered_to_good,
        "enhanced": enhanced,
        **metrics
    })

# ---------------------------------------------------
# Save Results
# ---------------------------------------------------
df = pd.DataFrame(results)
csv_path = os.path.join(OUT_DIR, "pipeline_results.csv")
df.to_csv(csv_path, index=False)

print("\n✅ Pipeline finished.")
print(f"Results saved at: {csv_path}")

# ---------------------------------------------------
# Optional: Auto Trigger Self-Learning Update
# ---------------------------------------------------
print("\n🔄 Checking for model update...")

update_script = os.path.join(PROJECT_ROOT, "self_learning", "update_cnn.py")

if os.path.exists(update_script):
    subprocess.run(["python", update_script])
else:
    print("update_cnn.py not found. Skipping retraining.")