import os
import sys
import tempfile
from typing import Dict, List, Optional
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


st.set_page_config(
    page_title="Drone Image Quality Pipeline",
    page_icon="drone",
    layout="wide",
)


@st.cache_resource
def load_pipeline_functions():
    from classifier.predict import predict_image
    from enhancer.enhance import enhance_image, MODE_ADVANCED, MODE_FAST
    from metrics.extract_metrics import extract_metrics

    return predict_image, extract_metrics, enhance_image, MODE_ADVANCED, MODE_FAST


def save_uploaded_file_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def run_single_image(
    image_path: str,
    image_name: str,
    predict_image,
    extract_metrics,
    enhance_image,
    enhancement_mode: str,
    enhancement_enabled: bool,
    upgrade_conf_threshold: float,
    good_accept_threshold: float,
) -> Dict:
    initial_label, initial_conf, _ = predict_image(image_path)
    metrics = extract_metrics(image_path)

    effective_label = initial_label
    forced_recheck = False
    # Guardrail: if model says GOOD but with low confidence, force recovery path.
    if initial_label == "GOOD" and float(initial_conf) < good_accept_threshold:
        effective_label = "RECOVERABLE"
        forced_recheck = True

    final_label = effective_label
    final_conf = initial_conf
    enhanced = False
    enhanced_image_rgb: Optional[np.ndarray] = None
    recovery_attempted = bool(enhancement_enabled and effective_label == "RECOVERABLE")

    if enhancement_enabled and effective_label == "RECOVERABLE":
        enhanced_bgr = enhance_image(image_path, metrics, mode=enhancement_mode)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as enh_tmp:
            cv2.imwrite(enh_tmp.name, enhanced_bgr)
            new_label, new_conf, _ = predict_image(enh_tmp.name)

        if new_label == "GOOD" and new_conf > upgrade_conf_threshold:
            final_label = "GOOD"
            final_conf = new_conf
            enhanced = True

        enhanced_image_rgb = bgr_to_rgb(enhanced_bgr)

    return {
        "image": image_name,
        "initial_label": initial_label,
        "effective_label": effective_label,
        "initial_confidence": round(float(initial_conf), 4),
        "final_label": final_label,
        "final_confidence": round(float(final_conf), 4),
        "forced_recheck": forced_recheck,
        "recovery_attempted": recovery_attempted,
        "recovered_to_good": bool(effective_label == "RECOVERABLE" and final_label == "GOOD"),
        "enhanced": enhanced,
        "lap_var": round(float(metrics["lap_var"]), 4),
        "orb_kpts": int(metrics["orb_kpts"]),
        "brightness": round(float(metrics["brightness"]), 4),
        "saturation": round(float(metrics["saturation"]), 4),
        "haze_score": round(float(metrics["haze_score"]), 4),
        "_enhanced_image_rgb": enhanced_image_rgb,
    }


def main():
    st.title("Drone Image Quality Assessment and Recovery")
    st.caption(
        "Upload drone images to classify quality as GOOD / RECOVERABLE / BAD, "
        "optionally enhance recoverable images, and export mission-level decisions."
    )

    with st.sidebar:
        st.header("Pipeline Controls")
        enhancement_enabled = st.toggle("Enable enhancement for RECOVERABLE", value=True)
        enhancement_mode_label = st.selectbox(
            "Enhancement Mode",
            options=["Advanced Hybrid (Best Quality)", "Fast Classical (Faster)"],
            index=0,
        )
        upgrade_conf_threshold = st.slider(
            "Upgrade to GOOD if confidence >",
            min_value=0.50,
            max_value=0.95,
            value=0.60,
            step=0.01,
        )
        good_accept_threshold = st.slider(
            "Minimum confidence to accept GOOD directly",
            min_value=0.45,
            max_value=0.95,
            value=0.65,
            step=0.01,
            help="GOOD predictions below this are treated as RECOVERABLE and re-checked.",
        )

        st.divider()
        st.header("Mission Decision Rules")
        min_good_ratio = st.slider(
            "Minimum GOOD percentage required",
            min_value=40,
            max_value=95,
            value=70,
            step=1,
            help="If final GOOD percentage is below this, re-fly is recommended.",
        )
        max_bad_ratio = st.slider(
            "Maximum BAD percentage allowed",
            min_value=5,
            max_value=60,
            value=20,
            step=1,
            help="If final BAD percentage exceeds this, re-fly is recommended.",
        )
        min_recovery_success = st.slider(
            "Minimum recovery success for RECOVERABLE (%)",
            min_value=10,
            max_value=95,
            value=35,
            step=1,
            help="Recovery success = recovered_to_good / total_recovery_pool.",
        )

    try:
        predict_image, extract_metrics, enhance_image, MODE_ADVANCED, MODE_FAST = load_pipeline_functions()
    except Exception as exc:
        st.error("Pipeline modules could not be loaded.")
        st.exception(exc)
        st.stop()

    enhancement_mode = (
        MODE_ADVANCED
        if enhancement_mode_label.startswith("Advanced")
        else MODE_FAST
    )

    uploads = st.file_uploader(
        "Upload drone images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploads:
        st.info("Add one or more images to start analysis.")
        return

    if st.button("Run Analysis", type="primary"):
        results: List[Dict] = []
        preview_items = []

        progress = st.progress(0)
        status = st.empty()

        for idx, up in enumerate(uploads, start=1):
            status.write(f"Processing: `{up.name}`")
            temp_path = save_uploaded_file_to_temp(up)

            row = run_single_image(
                image_path=temp_path,
                image_name=up.name,
                predict_image=predict_image,
                extract_metrics=extract_metrics,
                enhance_image=enhance_image,
                enhancement_mode=enhancement_mode,
                enhancement_enabled=enhancement_enabled,
                upgrade_conf_threshold=upgrade_conf_threshold,
                good_accept_threshold=good_accept_threshold,
            )
            results.append(row)
            preview_items.append((up, row.get("_enhanced_image_rgb"), row))

            progress.progress(idx / len(uploads))

        status.success("Analysis completed.")

        rows_for_table = []
        for r in results:
            clean = {k: v for k, v in r.items() if not k.startswith("_")}
            rows_for_table.append(clean)

        df = pd.DataFrame(rows_for_table)

        total = len(df)
        initial_counts = Counter(df["initial_label"].tolist())
        final_counts = Counter(df["final_label"].tolist())

        initial_good = int(initial_counts.get("GOOD", 0))
        initial_rec = int(initial_counts.get("RECOVERABLE", 0))
        initial_bad = int(initial_counts.get("BAD", 0))
        forced_rechecks = int(df["forced_recheck"].sum())

        good_count = int((df["final_label"] == "GOOD").sum())
        rec_count = int((df["final_label"] == "RECOVERABLE").sum())
        bad_count = int((df["final_label"] == "BAD").sum())
        upgraded = int(df["recovered_to_good"].sum())
        recovery_pool_total = int((df["effective_label"] == "RECOVERABLE").sum())
        recoverable_not_recovered = max(recovery_pool_total - upgraded, 0)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total", total)
        c2.metric("GOOD", good_count)
        c3.metric("RECOVERABLE", rec_count)
        c4.metric("BAD", bad_count)
        c5.metric("Upgraded After Enhancement", upgraded)

        st.subheader("Mission Summary")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Initial GOOD", initial_good)
        s2.metric("Initial RECOVERABLE", initial_rec)
        s3.metric("Initial BAD", initial_bad)
        s4.metric("Recovered -> GOOD", upgraded)
        s5.metric("Recoverable Still Not GOOD", recoverable_not_recovered)
        if forced_rechecks > 0:
            st.caption(
                f"Low-confidence GOOD rechecks: `{forced_rechecks}` "
                f"(these are included in recoverable totals)."
            )

        good_ratio = (good_count / total * 100.0) if total else 0.0
        bad_ratio = (bad_count / total * 100.0) if total else 0.0
        recovery_success = (upgraded / recovery_pool_total * 100.0) if recovery_pool_total else 0.0

        rule_good_fail = good_ratio < float(min_good_ratio)
        rule_bad_fail = bad_ratio > float(max_bad_ratio)
        rule_recovery_fail = recovery_pool_total > 0 and recovery_success < float(min_recovery_success)
        refly_needed = rule_good_fail or rule_bad_fail or rule_recovery_fail

        st.markdown(
            f"- Final GOOD: `{good_ratio:.1f}%`  \n"
            f"- Final BAD: `{bad_ratio:.1f}%`  \n"
            f"- Recovery success: `{recovery_success:.1f}%`"
        )

        if refly_needed:
            st.error("Recommendation: Re-fly required for this mission area.")
        else:
            st.success("Recommendation: Sufficient image quality. Proceed to 3D mapping.")

        failed_reasons = []
        if rule_good_fail:
            failed_reasons.append(
                f"GOOD percentage is below threshold ({good_ratio:.1f}% < {min_good_ratio}%)."
            )
        if rule_bad_fail:
            failed_reasons.append(
                f"BAD percentage is above threshold ({bad_ratio:.1f}% > {max_bad_ratio}%)."
            )
        if rule_recovery_fail:
            failed_reasons.append(
                f"Recovery success is below threshold ({recovery_success:.1f}% < {min_recovery_success}%)."
            )

        if failed_reasons:
            st.caption("Why re-fly is recommended:")
            for reason in failed_reasons:
                st.write(f"- {reason}")

        st.subheader("Results Table")
        st.caption(
            f"Enhancement mode used: `{enhancement_mode}`"
        )
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download Results CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="pipeline_results_streamlit.csv",
            mime="text/csv",
        )

        mission_report = pd.DataFrame(
            [
                {
                    "total_images": total,
                    "initial_good": initial_good,
                    "initial_recoverable": initial_rec,
                    "initial_bad": initial_bad,
                    "final_good": int(final_counts.get("GOOD", 0)),
                    "final_recoverable": int(final_counts.get("RECOVERABLE", 0)),
                    "final_bad": int(final_counts.get("BAD", 0)),
                    "low_conf_good_rechecks": forced_rechecks,
                    "recovery_pool_total": recovery_pool_total,
                    "recovered_to_good": upgraded,
                    "recoverable_not_recovered": recoverable_not_recovered,
                    "good_ratio_percent": round(good_ratio, 2),
                    "bad_ratio_percent": round(bad_ratio, 2),
                    "recovery_success_percent": round(recovery_success, 2),
                    "enhancement_mode": enhancement_mode,
                    "recommendation": "RE-FLY" if refly_needed else "PROCEED",
                }
            ]
        )
        st.download_button(
            "Download Mission Report CSV",
            data=mission_report.to_csv(index=False).encode("utf-8"),
            file_name="mission_decision_report.csv",
            mime="text/csv",
        )

        st.subheader("Image Preview")
        for upload, enhanced_rgb, row in preview_items:
            st.markdown(
                f"**{row['image']}** - Initial: `{row['initial_label']}` "
                f"({row['initial_confidence']}) -> Effective: `{row['effective_label']}` -> "
                f"Final: `{row['final_label']}` "
                f"({row['final_confidence']})"
            )
            left, right = st.columns(2)
            left.image(upload, caption="Original", use_container_width=True)
            if enhanced_rgb is not None:
                right.image(enhanced_rgb, caption="Enhanced", use_container_width=True)
            else:
                right.info("No enhancement applied")


if __name__ == "__main__":
    main()
