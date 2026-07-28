import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback
import altair as alt

st.set_page_config(page_title="Fraud Queue", layout="wide")

st.title("🔴 Ranked Fraud Investigation Queue")

# ---- Load model and test data ----
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('models/rf_model.pkl')
        test_df = pd.read_parquet('test_working_rf.parquet')
        return model, test_df
    except Exception as e:
        st.error(f"Asset load failed:\n{traceback.format_exc()}")
        st.stop()

model, test_df = load_assets()

# ---- Prepare features ----
feature_cols = [c for c in test_df.columns if c not in ['TransactionID', 'isFraud']]
X = test_df[feature_cols].values.astype(np.float32)  # float32 per Rule 4
y_true = test_df['isFraud'].values

# ---- Score ----
try:
    proba = model.predict_proba(X)[:, 1]
except Exception as e:
    st.error(f"Scoring failed:\n{traceback.format_exc()}")
    st.stop()

# ---- Build ranked queue ----
queue = pd.DataFrame({
    'TransactionID': test_df['TransactionID'].values,
    'fraud_probability': proba,
    'isFraud': y_true
})
queue = queue.sort_values('fraud_probability', ascending=False).reset_index(drop=True)
queue.index = queue.index + 1  # 1-based rank

# ---- Display ----
st.caption(f"Model: Random Forest (20k-trained) | Test rows: {len(queue)} | Fraud rate: {y_true.mean():.2%}")

st.dataframe(
    queue,
    use_container_width=True,
    height=600,
    column_config={
        "TransactionID": st.column_config.NumberColumn("Transaction ID", format="%.0f"),
        "fraud_probability": st.column_config.NumberColumn("Fraud Probability", format="%.4f"),
        "isFraud": st.column_config.NumberColumn("Actual Fraud", format="%d"),
    }
)

# Summary stats
fraud_caught_at_100 = queue.head(100)['isFraud'].sum()
total_fraud = queue['isFraud'].sum()
st.metric("Fraud in Top 100", f"{int(fraud_caught_at_100)} / {int(total_fraud)} ({fraud_caught_at_100/total_fraud:.1%})")

# ---- Investigator Capacity Slider ----
st.subheader("Investigator Capacity")
implied_threshold = None
try:
    k = st.slider("Investigator capacity (hours/day)", min_value=10, max_value=2000, value=100, step=10)

    # Compute live from the already-sorted queue
    top_k = queue.head(k)
    tp = int(top_k['isFraud'].sum())
    fp = k - tp
    precision = tp / k if k > 0 else 0.0
    recall = tp / total_fraud if total_fraud > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions Flagged", f"{k}")
    col2.metric("True Positives Caught", f"{tp}")
    col3.metric("Precision", f"{precision:.1%}")
    col4.metric("Recall", f"{recall:.1%}")

    # Threshold implied by current slider position (probability of the
    # k-th ranked transaction) — this connects the slider to the cost curve below.
    k_clamped = min(k, len(queue))
    implied_threshold = float(queue.iloc[k_clamped - 1]['fraud_probability'])

except Exception as e:
    st.error(f"Capacity computation failed:\n{traceback.format_exc()}")

# ---- Cost Curve Chart ----
st.subheader("Cost Curve (Threshold vs Expected Cost)")
try:
    cost_df = pd.read_parquet('outputs/rf_cost_analysis.parquet')
    # columns: threshold, cost, precision, recall

    # Find optimal threshold (minimum cost)
    opt_idx = cost_df['cost'].idxmin()
    opt_threshold = cost_df.loc[opt_idx, 'threshold']
    opt_cost = cost_df.loc[opt_idx, 'cost']

    # Cost at default 0.5 threshold
    cost_at_05 = cost_df.loc[(cost_df['threshold'] - 0.5).abs().idxmin(), 'cost']

    # Savings
    savings_dollars = cost_at_05 - opt_cost
    savings_pct = (savings_dollars / cost_at_05 * 100) if cost_at_05 > 0 else 0.0

    # Base cost curve line
    base_chart = alt.Chart(cost_df).mark_line(color='#1f77b4').encode(
        x=alt.X('threshold:Q', title='Threshold'),
        y=alt.Y('cost:Q', title='Expected Cost ($)')
    )

    layers = [base_chart]

    # Marker for the threshold implied by the current slider position
    if implied_threshold is not None:
        slider_rule = alt.Chart(pd.DataFrame({'threshold': [implied_threshold]})).mark_rule(
            color='orange', strokeDash=[4, 4], size=2
        ).encode(x='threshold:Q')
        layers.append(slider_rule)

    # Marker for the cost-minimizing optimal threshold
    optimal_rule = alt.Chart(pd.DataFrame({'threshold': [opt_threshold]})).mark_rule(
        color='green', strokeDash=[2, 2], size=2
    ).encode(x='threshold:Q')
    layers.append(optimal_rule)

    combined_chart = alt.layer(*layers).properties(height=350)
    st.altair_chart(combined_chart, use_container_width=True)

    st.caption(
        "🟢 dashed green = cost-minimizing optimal threshold &nbsp;&nbsp;|&nbsp;&nbsp; "
        "🟠 dashed orange = threshold implied by your current slider position"
    )

    # Optimal threshold annotation
    st.caption(f"Optimal threshold: **{opt_threshold:.4f}** (min cost = ${opt_cost:,.0f})")

    if implied_threshold is not None:
        st.caption(
            f"Your current capacity (top {k}) corresponds to threshold ≈ **{implied_threshold:.4f}**"
        )

    # Savings metric
    col1, col2 = st.columns(2)
    col1.metric("Cost at 0.5 threshold", f"${cost_at_05:,.0f}")
    col2.metric("Savings at optimal threshold", f"${savings_dollars:,.0f} ({savings_pct:.1f}%)")

except Exception as e:
    st.error(f"Cost curve load failed:\n{traceback.format_exc()}")
