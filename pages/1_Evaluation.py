"""Evaluation — project-measured detection performance and the
FP32 -> INT8 deployment optimization, plus the model-development
timeline. Every number on this page is loaded from
app/data/evaluation_summary.json (a small, committed, provenance-tracked
file copied verbatim from this project's actual experiment outputs) --
nothing here is typed in from memory. If a section's source data is
missing, that section is omitted rather than shown with placeholder
values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.components import render_section_title, setup_page  # noqa: E402
from app.evaluation_loader import get_section, load_evaluation_summary  # noqa: E402

setup_page("Evaluation")

st.markdown('<div class="adf-headline" style="font-size:1.7rem;">Evaluation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="adf-subtext">Project-measured evaluation on a fixed, held-out, balanced '
    "In-the-Wild subset. These are not the model authors' own benchmark numbers, and are "
    "not a claim of universal accuracy.</div>",
    unsafe_allow_html=True,
)

summary = load_evaluation_summary()
if summary is None:
    st.warning("Evaluation data is not available in this deployment.")
    st.stop()

provenance = summary.get("_provenance", {})
with st.expander("Dataset & methodology"):
    st.write(provenance.get("dataset_methodology", "Not documented."))
    st.caption("Source files: " + ", ".join(provenance.get("sources", [])))

int8_eval = get_section(summary, "int8_evaluation")
fp32_eval = get_section(summary, "fp32_evaluation")

if int8_eval:
    render_section_title("PROJECT-MEASURED EVALUATION — INT8 (deployed model)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("EER", f"{int8_eval['eer'] * 100:.1f}%")
    m2.metric("ROC-AUC", f"{int8_eval['roc_auc']:.4f}")
    m3.metric("F1", f"{int8_eval['f1']:.4f}")
    m4.metric("Bonafide FPR", f"{int8_eval['bonafide_fpr'] * 100:.1f}%")
    m5.metric("Spoof FNR", f"{int8_eval['spoof_fnr'] * 100:.1f}%")
    st.caption(f"n = {int8_eval['n_samples']} evaluation clips, frozen calibrated threshold {int8_eval['threshold_used']:.4f}.")

    cm = int8_eval.get("confusion_matrix")
    if cm:
        import matplotlib.figure

        render_section_title("Confusion matrix (INT8)")
        fig = matplotlib.figure.Figure(figsize=(3.6, 3.2), facecolor="#131922")
        ax = fig.add_subplot(111)
        matrix = [[cm["tn_bonafide_correct"], cm["fp_bonafide_as_spoof"]], [cm["fn_spoof_as_bonafide"], cm["tp_spoof_correct"]]]
        ax.set_facecolor("#131922")
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], labels=["Bonafide", "Spoof"], color="#8b96a8")
        ax.set_yticks([0, 1], labels=["Bonafide", "Spoof"], color="#8b96a8")
        ax.set_xlabel("Predicted", color="#8b96a8")
        ax.set_ylabel("Ground truth", color="#8b96a8")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="white", fontsize=12)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption(
            "Confusion matrix from the frozen 200-clip evaluation set (real counts, not derived "
            "from the headline metrics)."
        )

acceptance = get_section(summary, "fp32_vs_int8_acceptance")
if fp32_eval and int8_eval:
    render_section_title("Deployment optimization — FP32 vs INT8 detection performance")
    st.markdown(
        """
| Metric | FP32 | INT8 |
|---|---|---|
| EER | {fp32_eer:.1f}% | {int8_eer:.1f}% |
| ROC-AUC | {fp32_auc:.4f} | {int8_auc:.4f} |
| F1 | {fp32_f1:.4f} | {int8_f1:.4f} |
| Bonafide FPR | {fp32_fpr:.1f}% | {int8_fpr:.1f}% |
        """.format(
            fp32_eer=fp32_eval["eer"] * 100,
            int8_eer=int8_eval["eer"] * 100,
            fp32_auc=fp32_eval["roc_auc"],
            int8_auc=int8_eval["roc_auc"],
            fp32_f1=fp32_eval["f1"],
            int8_f1=int8_eval["f1"],
            fp32_fpr=fp32_eval["bonafide_fpr"] * 100,
            int8_fpr=int8_eval["bonafide_fpr"] * 100,
        )
    )
    if acceptance:
        st.caption(acceptance.get("interpretation", ""))

resource = get_section(summary, "resource_comparison")
if resource:
    render_section_title("Deployment optimization — model size, memory, and latency")
    size = resource["model_size_bytes"]
    load_time = resource["cold_load_time_s_threads_2"]
    rss = resource["complete_app_peak_rss_mb_after_30s_analysis"]
    latency = resource["real_30s_application_analysis_time_s"]
    st.markdown(
        f"""
| | FP32 | INT8 |
|---|---|---|
| Model size | {size['fp32'] / 1e9:.2f} GB | {size['int8'] / 1e6:.0f} MB |
| Cold load time | {load_time['fp32']:.1f} s | {load_time['int8']:.1f} s |
| Complete-app peak RSS (30s analysis) | {rss['fp32'] / 1000:.2f} GB | {rss['int8']:.0f} MB |
| Real 30-second analysis time | {latency['fp32']:.1f} s | {latency['int8']:.1f} s |
        """
    )
    st.caption(
        f"Size reduction: {resource['size_reduction_percent']:.1f}%. Dynamic INT8 quantization "
        "materially reduced deployment cost while preserving similar detection performance -- "
        "the differences above are within normal small-sample variance, not a claim that "
        "quantization improved accuracy."
    )

sara = get_section(summary, "sara_same_clip_comparison")
timeline = get_section(summary, "model_development_timeline")
if timeline:
    render_section_title("Model development", "How empirical testing led to the deployed detector.")
    for i, step in enumerate(timeline["steps"]):
        st.markdown(f"**{i + 1}. {step['step']}** — {step['detail']}")
    if sara:
        with st.expander("Same-clip Sara vs. Spectra comparison"):
            st.caption(
                f"Both rows below use the exact same {sara['n_shared_evaluation_clips']} evaluation clips."
                if sara.get("same_evaluation_set")
                else "These are separate experiments, not a same-set comparison."
            )
            st.markdown(
                f"""
| Metric | Sara (deployed baseline) | Spectra (FP32) |
|---|---|---|
| Accuracy | {sara['sara']['accuracy'] * 100:.1f}% | {sara['spectra']['accuracy'] * 100:.1f}% |
| Bonafide FPR | {sara['sara']['bonafide_fpr'] * 100:.1f}% | {sara['spectra']['bonafide_fpr'] * 100:.1f}% |
| EER | {sara['sara']['eer'] * 100:.1f}% | {sara['spectra']['eer'] * 100:.1f}% |
                """
            )
            st.caption(sara.get("note", ""))

robustness = get_section(summary, "robustness_fp32_reference")
if robustness:
    render_section_title("Robustness (FP32 reference)", robustness.get("note", ""))
    rows = []
    for cond, m in robustness["conditions"].items():
        rows.append({"Condition": cond, "EER": f"{m['eer'] * 100:.1f}%", "ROC-AUC": f"{m['roc_auc']:.3f}", "Bonafide FPR": f"{m['bonafide_fpr_at_calibration_threshold'] * 100:.1f}%"})
    st.dataframe(rows, width='stretch', hide_index=True)
