# new_app.py
import streamlit as st
import json
import pandas as pd
import numpy as np
import random
import time
from typing import List, Dict, Optional
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px
import openai

# ----------------------
# Data loading
# ----------------------
def load_data():
    """Load the Customer_Sentiment.csv file expected at data/Customer_Sentiment.csv"""
    try:
        df = pd.read_csv("data/Customer_Sentiment.csv")
        return df
    except FileNotFoundError:
        st.error("Could not find data/Customer_Sentiment.csv in the data/ folder.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

# ----------------------
# Helpers: balanced split and few-shot example creation
# ----------------------
def create_balanced_split_for_label(df: pd.DataFrame, text_col: str, label_col: str, test_size: float = 0.3, random_state: int = 42):
    """Return X_train, X_test, y_train, y_test for the requested label column (stratified)."""
    if label_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"Missing required columns: {text_col} or {label_col}")
    # Filter out rows where label is null
    subset = df[[text_col, label_col]].dropna().copy()
    # If too few examples for stratify, just split randomly
    try:
        from sklearn.model_selection import train_test_split
        X = subset[text_col].reset_index(drop=True)
        y = subset[label_col].reset_index(drop=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test
    except Exception:
        # fallback
        idx = list(range(len(subset)))
        random.seed(random_state)
        random.shuffle(idx)
        split = int((1 - test_size) * len(idx))
        train_idx = idx[:split]
        test_idx = idx[split:]
        X_train = subset[text_col].iloc[train_idx].reset_index(drop=True)
        X_test = subset[text_col].iloc[test_idx].reset_index(drop=True)
        y_train = subset[label_col].iloc[train_idx].reset_index(drop=True)
        y_test = subset[label_col].iloc[test_idx].reset_index(drop=True)
        return X_train, X_test, y_train, y_test

def create_few_shot_examples_from_series(X_train, y_train, labels: List[str], n_examples_per_class: int = 5, random_state: int = 42):
    """Create balanced few-shot example list of dicts {'sentence', 'label'} for the given labels."""
    df_train = pd.DataFrame({'sentence': X_train, 'label': y_train}).reset_index(drop=True)
    few_shot_examples = []
    for label in labels:
        class_examples = df_train[df_train['label'] == label]
        if class_examples.empty:
            continue
        sample_n = min(n_examples_per_class, len(class_examples))
        sampled = class_examples.sample(n=sample_n, random_state=random_state)
        for _, row in sampled.iterrows():
            few_shot_examples.append({'sentence': row['sentence'], 'label': row['label']})
    random.shuffle(few_shot_examples)
    return few_shot_examples

# ----------------------
# Prompt builders for each task
# ----------------------
def build_few_shot_prompt_generic(few_shot_examples: List[Dict], target_text: str, label_name: str, allowed_labels: List[str], guidance: Optional[str] = None) -> str:
    """Generic few-shot prompt builder. label_name is used in the text (e.g., 'Sentiment')."""
    labels_str = ", ".join(allowed_labels)
    prompt = f"""You are a concise classifier that outputs exactly one label (no extra text). The label must be one of: {labels_str}.
Return the label only, exactly matching one of those options.

"""
    if guidance:
        prompt += guidance + "\n\n"
    prompt += "Here are examples:\n\n"
    for ex in few_shot_examples:
        # Example labels may be mixed-case in df; include them verbatim
        prompt += f'Text: "{ex["sentence"]}"\n{label_name}: {ex["label"]}\n\n'
    prompt += f'Text: "{target_text}"\n{label_name}:'
    return prompt

# ----------------------
# LLM call wrapper
# ----------------------
def predict_with_llm_for_task(client, few_shot_examples: List[Dict], texts: List[str], allowed_labels: List[str], label_name: str, model_name: str = "gpt-4.1-nano", temperature: float = 0.1, max_tokens: int = 12) -> List[Dict]:
    """
    Make predictions for a single task (sentiment/gender/age) using few-shot prompting.
    Returns list of {'text','prediction'}.
    """
    if isinstance(texts, str):
        texts = [texts]
    results = []
    for text in texts:
        try:
            # Basic cleaning to avoid bad unicode tokens for prompts
            cleaned_text = text if isinstance(text, str) else ""
            # Build prompt
            # Provide optional small guidance depending on task
            prompt = build_few_shot_prompt_generic(few_shot_examples, cleaned_text, label_name, allowed_labels)
            # Build system instruction to return only exact label
            system_msg = {"role": "system", "content": f"You are a helpful classifier. Respond with exactly one of: {', '.join(allowed_labels)} and nothing else."}
            user_msg = {"role": "user", "content": prompt}
            response = client.chat.completions.create(
                model=model_name,
                messages=[system_msg, user_msg],
                temperature=temperature,
                max_tokens=max_tokens
            )
            raw = response.choices[0].message.content.strip()
            raw_lower = raw.lower()
            # Map returned text to one of allowed labels (case-insensitive matching)
            mapped = None
            for lbl in allowed_labels:
                if lbl.lower() in raw_lower:
                    mapped = lbl
                    break
            # If not matched, try exact match
            if mapped is None:
                for lbl in allowed_labels:
                    if raw.strip() == lbl:
                        mapped = lbl
                        break
            if mapped is None:
                # fallback to the first allowed label (safe default)
                mapped = allowed_labels[0]
            results.append({'text': text, 'prediction': mapped, 'raw': raw})
            time.sleep(0.12)  # small delay
        except Exception as e:
            st.error(f"LLM error for text: {e}")
            results.append({'text': text, 'prediction': allowed_labels[0], 'raw': ''})
    return results

# ----------------------
# Evaluation helper (for any task)
# ----------------------
def evaluate_task(client, few_shot_examples, df_test_texts, df_test_labels, allowed_labels, label_name, model_name):
    """Sample up to N items from test set and predict, returning true labels, preds, and texts."""
    n = min(50, len(df_test_texts))
    if n == 0:
        return [], [], []
    indices = random.sample(range(len(df_test_texts)), n)
    X_sample = [df_test_texts.iloc[i] for i in indices]
    y_sample = [df_test_labels.iloc[i] for i in indices]
    preds = predict_with_llm_for_task(client, few_shot_examples, X_sample, allowed_labels, label_name, model_name)
    y_pred = [p['prediction'] for p in preds]
    return y_sample, y_pred, X_sample

# ----------------------
# UI Helper functions to reduce duplication
# ----------------------
def display_classification_report(y_true, y_pred):
    try:
        report = classification_report(y_true, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3))
    except Exception:
        st.write("Could not produce classification report.")

def display_confusion_matrix(y_true, y_pred, labels):
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig = px.imshow(cm, text_auto=True, labels={'x':'Predicted','y':'Actual'}, x=labels, y=labels)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

# ----------------------
# Streamlit main app
# ----------------------
def main():
    st.set_page_config(page_title="LLM Few-Shot: Sentiment / Gender / Age", page_icon="🤖", layout="wide")
    st.title("LLM Few-Shot Classifiers — Sentiment, Gender, Age")
    st.markdown("This app uses few-shot prompt patterns with an LLM to classify customer `review_text` for sentiment, gender, and age-group.")

    # ----------------------
    # API Configuration (merged from old app.py)
    # ----------------------
    st.sidebar.header("🔧 API Configuration")

    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key"
    )

    model_choice = st.sidebar.selectbox(
        "Model",
        ["gpt-4.1-nano", "gpt-4o-mini", "gpt-3.5-turbo"],
        index=0
    )

    examples_per_class = st.sidebar.slider("Examples per class (few-shot)", 0, 20, 5)
    test_size_pct = st.sidebar.slider("Test set size (%)", 10, 50, 30) / 100.0

    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()

    # Initialize OpenAI client exactly like in old app.py
    try:
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        st.stop()

    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    if df.empty:
        st.stop()

    # Validate required columns and show dataset summary
    required_cols = ["review_text", "sentiment", "gender", "age_group"]
    st.sidebar.header("Dataset Info")
    st.sidebar.metric("Total rows", len(df))
    for col in required_cols:
        if col not in df.columns:
            st.warning(f"Missing column in CSV: {col} (some features won't be available).")

    # Determine allowed labels automatically from dataset where possible
    sentiment_labels = sorted(df['sentiment'].dropna().unique().tolist()) if 'sentiment' in df.columns else ['positive','negative','neutral']
    # normalize to lower-case and consistent values if necessary
    sentiment_labels = [str(s).lower() for s in sentiment_labels]
    # make sure the canonical order is positive/negative/neutral if present
    canonical_sent = []
    for s in ['positive','negative','neutral']:
        if s in sentiment_labels:
            canonical_sent.append(s)
    for s in sentiment_labels:
        if s not in canonical_sent:
            canonical_sent.append(s)
    sentiment_labels = canonical_sent

    gender_labels = sorted(df['gender'].dropna().unique().tolist()) if 'gender' in df.columns else ['male','female','other']
    gender_labels = [str(s).lower() for s in gender_labels]

    age_labels = sorted(df['age_group'].dropna().unique().tolist(), key=lambda x: str(x)) if 'age_group' in df.columns else []
    age_labels = [str(s) for s in age_labels]

    st.sidebar.markdown("**Detected labels**")
    st.sidebar.write("Sentiment:", sentiment_labels)
    st.sidebar.write("Gender:", gender_labels)
    st.sidebar.write("Age groups:", age_labels)

    st.markdown("---")
    st.header("🔧 Prepare Few-Shot Examples (per task)")

    # Prepare each task separately, store in session_state
    if st.button("Prepare all few-shot examples"):
        # Sentiment
        try:
            X_tr_s, X_te_s, y_tr_s, y_te_s = create_balanced_split_for_label(df, "review_text", "sentiment", test_size=test_size_pct)
            fs_sent = create_few_shot_examples_from_series(X_tr_s, y_tr_s, sentiment_labels, n_examples_per_class=examples_per_class)
            st.session_state.few_shot_sentiment = fs_sent
            st.session_state.X_test_sentiment = X_te_s
            st.session_state.y_test_sentiment = y_te_s
            st.success(f"Prepared sentiment few-shot examples: {len(fs_sent)} examples")
        except Exception as e:
            st.error(f"Sentiment few-shot preparation failed: {e}")

        # Gender
        try:
            X_tr_g, X_te_g, y_tr_g, y_te_g = create_balanced_split_for_label(df, "review_text", "gender", test_size=test_size_pct)
            fs_gender = create_few_shot_examples_from_series(X_tr_g, y_tr_g, gender_labels, n_examples_per_class=examples_per_class)
            st.session_state.few_shot_gender = fs_gender
            st.session_state.X_test_gender = X_te_g
            st.session_state.y_test_gender = y_te_g
            st.success(f"Prepared gender few-shot examples: {len(fs_gender)} examples")
        except Exception as e:
            st.error(f"Gender few-shot preparation failed: {e}")

        # Age group
        try:
            # If age labels are many, we still use the existing unique values
            if not age_labels:
                raise ValueError("No age_group labels found in CSV.")
            X_tr_a, X_te_a, y_tr_a, y_te_a = create_balanced_split_for_label(df, "review_text", "age_group", test_size=test_size_pct)
            fs_age = create_few_shot_examples_from_series(X_tr_a, y_tr_a, age_labels, n_examples_per_class=examples_per_class)
            st.session_state.few_shot_age = fs_age
            st.session_state.X_test_age = X_te_a
            st.session_state.y_test_age = y_te_a
            st.success(f"Prepared age-group few-shot examples: {len(fs_age)} examples")
        except Exception as e:
            st.error(f"Age-group few-shot preparation failed: {e}")

    # Display a short preview of examples if prepared
    if 'few_shot_sentiment' in st.session_state:
        st.subheader("Sample Sentiment Examples")
        for ex in st.session_state.few_shot_sentiment[:6]:
            label = ex['label']
            if label.lower() in ['negative','neg','bad']:
                st.error(f"{label}: {ex['sentence'][:140]}...")
            else:
                st.success(f"{label}: {ex['sentence'][:140]}...")

    st.markdown("---")

    # Task evaluation panels
    st.header("📊 Task Evaluations (optional)")
    cols = st.columns(3)
    with cols[0]:
        st.subheader("Sentiment Eval")
        if 'few_shot_sentiment' in st.session_state:
            if st.button("Run Sentiment Evaluation"):
                with st.spinner("Evaluating sentiment..."):
                    y_true, y_pred, texts = evaluate_task(client, st.session_state.few_shot_sentiment, st.session_state.X_test_sentiment, st.session_state.y_test_sentiment, sentiment_labels, "Sentiment", model_choice)
                    if len(y_true) == 0:
                        st.warning("No test data available for sentiment.")
                    else:
                        st.session_state.y_true_sentiment = y_true
                        st.session_state.y_pred_sentiment = y_pred
                        acc = accuracy_score(y_true, y_pred)
                        st.success(f"Sentiment accuracy: {acc:.3f}")
                        # classification report
                        display_classification_report(y_true, y_pred)
                        # confusion matrix
                        display_confusion_matrix(y_true, y_pred, sentiment_labels)
        else:
            st.info("Prepare sentiment few-shot examples first.")

    with cols[1]:
        st.subheader("Gender Eval")
        if 'few_shot_gender' in st.session_state:
            if st.button("Run Gender Evaluation"):
                with st.spinner("Evaluating gender..."):
                    y_true, y_pred, texts = evaluate_task(client, st.session_state.few_shot_gender, st.session_state.X_test_gender, st.session_state.y_test_gender, gender_labels, "Gender", model_choice)
                    if len(y_true) == 0:
                        st.warning("No test data available for gender.")
                    else:
                        st.session_state.y_true_gender = y_true
                        st.session_state.y_pred_gender = y_pred
                        acc = accuracy_score(y_true, y_pred)
                        st.success(f"Gender accuracy: {acc:.3f}")
                        display_classification_report(y_true, y_pred)
        else:
            st.info("Prepare gender few-shot examples first.")

    with cols[2]:
        st.subheader("Age-group Eval")
        if 'few_shot_age' in st.session_state:
            if st.button("Run Age Evaluation"):
                with st.spinner("Evaluating age-group..."):
                    y_true, y_pred, texts = evaluate_task(client, st.session_state.few_shot_age, st.session_state.X_test_age, st.session_state.y_test_age, age_labels, "Age Group", model_choice)
                    if len(y_true) == 0:
                        st.warning("No test data available for age.")
                    else:
                        st.session_state.y_true_age = y_true
                        st.session_state.y_pred_age = y_pred
                        acc = accuracy_score(y_true, y_pred)
                        st.success(f"Age-group accuracy: {acc:.3f}")
                        display_classification_report(y_true, y_pred)
        else:
            st.info("Prepare age-group few-shot examples first.")

    st.markdown("---")
    st.header("🎯 Make Predictions")

    # Single text classification area: allow user to pick which task(s) to run
    st.subheader("Single Review Prediction")
    single_text = st.text_area("Enter a review to classify (single):", height=120)

    task_cols = st.columns(3)
    run_sent = task_cols[0].checkbox("Predict Sentiment", value=True)
    run_gender = task_cols[1].checkbox("Predict Gender", value=False)
    run_age = task_cols[2].checkbox("Predict Age Group", value=False)

    if st.button("Run Prediction on Single Text") and single_text.strip():
        results = {}
        if run_sent:
            if 'few_shot_sentiment' not in st.session_state:
                st.error("Prepare sentiment few-shot examples first.")
            else:
                res = predict_with_llm_for_task(client, st.session_state.few_shot_sentiment, [single_text], sentiment_labels, "Sentiment", model_choice)[0]
                results['sentiment'] = res
        if run_gender:
            if 'few_shot_gender' not in st.session_state:
                st.error("Prepare gender few-shot examples first.")
            else:
                res = predict_with_llm_for_task(client, st.session_state.few_shot_gender, [single_text], gender_labels, "Gender", model_choice)[0]
                results['gender'] = res
        if run_age:
            if 'few_shot_age' not in st.session_state:
                st.error("Prepare age-group few-shot examples first.")
            else:
                res = predict_with_llm_for_task(client, st.session_state.few_shot_age, [single_text], age_labels, "Age Group", model_choice)[0]
                results['age_group'] = res

        st.subheader("Prediction Results")
        for k, v in results.items():
            st.write(f"**{k.capitalize()}**: {v['prediction']}  — raw model output: `{v['raw']}`")

    st.markdown("---")
    st.subheader("📤 Batch Classification (CSV / Text)")

    batch_method = st.radio("Batch input type:", ["Upload CSV with review_text column", "Upload TXT (one per line)"], index=0)

    if batch_method.startswith("Upload CSV"):
        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                if 'review_text' not in batch_df.columns:
                    st.error("CSV must contain a 'review_text' column.")
                else:
                    st.info(f"Loaded {len(batch_df)} rows.")
                    # choose which tasks to run
                    run_sent_b = st.checkbox("Batch: Sentiment", value=True)
                    run_gender_b = st.checkbox("Batch: Gender", value=False)
                    run_age_b = st.checkbox("Batch: Age Group", value=False)
                    if st.button("Run Batch Classification"):
                        texts = batch_df['review_text'].fillna("").astype(str).tolist()
                        results = []
                        if run_sent_b:
                            if 'few_shot_sentiment' not in st.session_state:
                                st.error("Prepare sentiment few-shot examples first.")
                            else:
                                sent_res = predict_with_llm_for_task(client, st.session_state.few_shot_sentiment, texts, sentiment_labels, "Sentiment", model_choice)
                                results.append(pd.DataFrame([{'text': r['text'], 'sentiment_pred': r['prediction'], 'sentiment_raw': r['raw']} for r in sent_res]))
                        if run_gender_b:
                            if 'few_shot_gender' not in st.session_state:
                                st.error("Prepare gender few-shot examples first.")
                            else:
                                gen_res = predict_with_llm_for_task(client, st.session_state.few_shot_gender, texts, gender_labels, "Gender", model_choice)
                                results.append(pd.DataFrame([{'text': r['text'], 'gender_pred': r['prediction'], 'gender_raw': r['raw']} for r in gen_res]))
                        if run_age_b:
                            if 'few_shot_age' not in st.session_state:
                                st.error("Prepare age few-shot examples first.")
                            else:
                                age_res = predict_with_llm_for_task(client, st.session_state.few_shot_age, texts, age_labels, "Age Group", model_choice)
                                results.append(pd.DataFrame([{'text': r['text'], 'age_pred': r['prediction'], 'age_raw': r['raw']} for r in age_res]))
                        # Merge results side-by-side on text
                        if results:
                            out_df = results[0]
                            for other in results[1:]:
                                out_df = out_df.merge(other, on='text', how='outer')
                            st.dataframe(out_df, use_container_width=True)
                            st.download_button("Download results CSV", out_df.to_csv(index=False), file_name="batch_classification_results.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error reading uploaded CSV: {e}")

    else:
        uploaded_txt = st.file_uploader("Upload TXT", type=['txt'])
        if uploaded_txt is not None:
            txt_contents = uploaded_txt.read().decode('utf-8').strip().splitlines()
            texts = [t.strip() for t in txt_contents if t.strip()]
            st.info(f"Loaded {len(texts)} lines.")
            run_sent_b = st.checkbox("Batch: Sentiment", value=True)
            run_gender_b = st.checkbox("Batch: Gender", value=False)
            run_age_b = st.checkbox("Batch: Age Group", value=False)
            if st.button("Run Batch Classification (TXT)"):
                results = {}
                if run_sent_b:
                    if 'few_shot_sentiment' not in st.session_state:
                        st.error("Prepare sentiment few-shot examples first.")
                    else:
                        sent_res = predict_with_llm_for_task(client, st.session_state.few_shot_sentiment, texts, sentiment_labels, "Sentiment", model_choice)
                        results['sentiment'] = sent_res
                if run_gender_b:
                    if 'few_shot_gender' not in st.session_state:
                        st.error("Prepare gender few-shot examples first.")
                    else:
                        gen_res = predict_with_llm_for_task(client, st.session_state.few_shot_gender, texts, gender_labels, "Gender", model_choice)
                        results['gender'] = gen_res
                if run_age_b:
                    if 'few_shot_age' not in st.session_state:
                        st.error("Prepare age few-shot examples first.")
                    else:
                        age_res = predict_with_llm_for_task(client, st.session_state.few_shot_age, texts, age_labels, "Age Group", model_choice)
                        results['age'] = age_res

                # Construct combined DataFrame
                df_out = pd.DataFrame({'text': texts})
                if 'sentiment' in results:
                    df_out['sentiment_pred'] = [r['prediction'] for r in results['sentiment']]
                    df_out['sentiment_raw'] = [r['raw'] for r in results['sentiment']]
                if 'gender' in results:
                    df_out['gender_pred'] = [r['prediction'] for r in results['gender']]
                    df_out['gender_raw'] = [r['raw'] for r in results['gender']]
                if 'age' in results:
                    df_out['age_pred'] = [r['prediction'] for r in results['age']]
                    df_out['age_raw'] = [r['raw'] for r in results['age']]
                st.dataframe(df_out, use_container_width=True)
                st.download_button("Download results CSV", df_out.to_csv(index=False), file_name="batch_text_results.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Raw dataset preview (first 200 rows)")
    st.dataframe(df.head(200), use_container_width=True)

if __name__ == "__main__":
    main()
