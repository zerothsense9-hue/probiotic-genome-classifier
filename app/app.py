import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from collections import Counter
import tempfile

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = BASE_DIR / "models" / "xgboost_k5_balanced_model.pkl"
FEATURE_COLUMNS_FILE = BASE_DIR / "models" / "k5_balanced_feature_columns.pkl"

K = 5
NUCLEOTIDES = ["A", "C", "G", "T"]


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    feature_columns = joblib.load(FEATURE_COLUMNS_FILE)
    return model, feature_columns


def read_fasta_sequence(fasta_path):
    sequence_parts = []

    with open(fasta_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            sequence_parts.append(line.upper())

    return "".join(sequence_parts)


def generate_all_kmers(k):
    return ["".join(p) for p in product(NUCLEOTIDES, repeat=k)]


def extract_kmer_frequencies(sequence, k, all_kmers):
    counts = Counter()
    total = 0

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]

        if set(kmer).issubset({"A", "C", "G", "T"}):
            counts[kmer] += 1
            total += 1

    if total == 0:
        return {kmer: 0 for kmer in all_kmers}

    return {kmer: counts[kmer] / total for kmer in all_kmers}


def predict_genome(fasta_path):
    model, feature_columns = load_model()

    sequence = read_fasta_sequence(fasta_path)
    all_kmers = generate_all_kmers(K)
    features = extract_kmer_frequencies(sequence, K, all_kmers)

    X = pd.DataFrame([features])
    X = X[feature_columns]

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    return prediction, probability, len(sequence)


def plot_probability_chart(non_prob, prob):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(["Non-probiotic", "Probiotic"], [non_prob, prob])
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Prediction Probability")
    st.pyplot(fig)


def plot_class_distribution():
    labels = ["Probiotic", "Non-probiotic / Pathogenic"]
    values = [402, 343]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Number of genomes")
    ax.set_title("Training Dataset Class Distribution")

    for i, value in enumerate(values):
        ax.text(i, value + 5, str(value), ha="center")

    st.pyplot(fig)


def plot_confusion_matrix():
    matrix = [[65, 4], [4, 76]]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Non-probiotic", "Predicted Probiotic"])
    ax.set_yticklabels(["Actual Non-probiotic", "Actual Probiotic"])
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i][j], ha="center", va="center")

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig)


def plot_model_metrics():
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score"],
        "Score": [94.63, 95.00, 95.00, 94.20, 95.00]
    })

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(metrics_df["Metric"], metrics_df["Score"])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Model Performance Metrics")

    for i, value in enumerate(metrics_df["Score"]):
        ax.text(i, value + 1, f"{value:.2f}%", ha="center")

    st.pyplot(fig)


def plot_feature_importance():
    model, feature_columns = load_model()

    if not hasattr(model, "feature_importances_"):
        st.warning("Feature importance is not available for this model.")
        return

    importance_df = pd.DataFrame({
        "k-mer": feature_columns,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    ).head(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["k-mer"], importance_df["importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Important k-mer Features")

    st.pyplot(fig)


st.set_page_config(
    page_title="Probiotic Genome Classifier",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Probiotic Genome Classifier")
st.markdown("**Owner:** Dr. Shakira Ghazanfar")
st.markdown("**Developed by:** Engr. Zaeem Khan")
st.write(
    "A genome-based machine learning tool for predicting probiotic potential "
    "from bacterial genome FASTA files using k-mer frequency features and XGBoost."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Predict",
    "📊 Model Performance",
    "🧬 Feature Importance",
    "🔁 Workflow",
    "ℹ️ About"
])

with tab1:
    st.header("Genome Prediction")

    st.info("Upload a bacterial genome file in `.fna`, `.fa`, or `.fasta` format.")

    uploaded_file = st.file_uploader(
        "Upload genome FASTA file",
        type=["fna", "fa", "fasta"]
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fna") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.success("Genome file uploaded successfully.")

        if st.button("Predict Probiotic Status"):
            with st.spinner("Extracting k-mer features and predicting..."):
                prediction, probability, sequence_length = predict_genome(tmp_path)

            non_probiotic_confidence = probability[0] * 100
            probiotic_confidence = probability[1] * 100

            st.subheader("Prediction Result")
            st.write(f"**Sequence length:** {sequence_length:,} bp")

            col1, col2 = st.columns(2)

            with col1:
                if prediction == 1:
                    st.success("Prediction: Likely Probiotic")
                    st.metric("Probiotic Confidence", f"{probiotic_confidence:.2f}%")
                else:
                    st.error("Prediction: Likely Non-probiotic / Pathogenic")
                    st.metric("Non-probiotic Confidence", f"{non_probiotic_confidence:.2f}%")

            with col2:
                prob_df = pd.DataFrame({
                    "Class": ["Non-probiotic / Pathogenic", "Probiotic"],
                    "Probability (%)": [
                        round(non_probiotic_confidence, 2),
                        round(probiotic_confidence, 2)
                    ]
                })

                st.dataframe(prob_df, use_container_width=True)

            plot_probability_chart(non_probiotic_confidence, probiotic_confidence)

with tab2:
    st.header("Model Performance Visualizations")

    st.subheader("Training Dataset Summary")
    st.write("The model was trained using a balanced genome dataset.")

    summary_df = pd.DataFrame({
        "Class": ["Probiotic", "Non-probiotic / Pathogenic"],
        "Genome Count": [402, 343]
    })

    st.dataframe(summary_df, use_container_width=True)
    plot_class_distribution()

    st.subheader("Evaluation Metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score"],
        "Score": ["94.63%", "95.00%", "95.00%", "94.20%", "95.00%"]
    })

    st.dataframe(metrics_df, use_container_width=True)
    plot_model_metrics()

    st.subheader("Confusion Matrix")
    st.write("Test set results: TN = 65, FP = 4, FN = 4, TP = 76")
    plot_confusion_matrix()

with tab3:
    st.header("Top k-mer Feature Importance")
    st.write(
        "This chart shows the k-mer patterns that contributed most strongly "
        "to the XGBoost prediction model."
    )
    plot_feature_importance()

with tab4:
    st.header("Analysis Workflow")

    st.graphviz_chart("""
        digraph {
            rankdir=LR;
            A [label="Genome Accessions"];
            B [label="Download FASTA Files"];
            C [label="Extract k-mer Features"];
            D [label="Train XGBoost Model"];
            E [label="Evaluate Performance"];
            F [label="Web Prediction Tool"];

            A -> B -> C -> D -> E -> F;
        }
    """)

    st.write(
        "The workflow follows a genome-based machine learning strategy: "
        "whole genome sequences are converted into k-mer frequency features, "
        "then used to train and evaluate a probiotic classification model."
    )

with tab5:
    st.header("About This Tool")

    st.write("""
    This prototype predicts whether a bacterial genome is likely to be probiotic
    or non-probiotic/pathogenic. It uses k-mer frequency features extracted from
    whole genome FASTA files and an XGBoost classifier.

    The tool was developed as a genome-based probiotic screening system inspired
    by published machine learning approaches for probiotic genome classification.
    """)

    st.markdown("**Owner:** Dr. Shakira Ghazanfar")
    st.markdown("**Developed by:** Engr. Zaeem Khan")

st.markdown("---")
st.caption(
    "Owner: Dr. Shakira Ghazanfar | Developed by Zaeem Khan | "
    "Prototype genome-based probiotic classifier using k-mer features and XGBoost."
)