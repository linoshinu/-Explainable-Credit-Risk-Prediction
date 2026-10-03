from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from credit_risk.data import FEATURES, FEATURE_LABELS, TARGET, make_demo_dataset
from credit_risk.explain import explain_default_risk
from credit_risk.modeling import (
    positive_class_probability,
    threshold_curve,
    threshold_summary,
    train_model_suite,
)

st.set_page_config(
    page_title="ClearRisk | Explainable Credit Risk",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#eaf0fb; --muted:#9aa9c3; --line:rgba(157,178,210,.16); --panel:#111c30; --teal:#47d7bd; --blue:#86a9ff; }
    .stApp { background: radial-gradient(ellipse at 8% 0%, #172b47 0%, #0a1220 44%, #080e19 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:rgba(8,15,27,.92); border-right:1px solid var(--line); }
    [data-testid="stSidebar"] * { color:var(--ink); }
    h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.03em; }
    h1 { font-size:2.75rem !important; line-height:1.05 !important; }
    h2 { font-size:1.55rem !important; }
    p,li,label { color:#d4dff1; }
    .hero { padding:1.25rem 0 .55rem; }
    .eyebrow { color:var(--teal); text-transform:uppercase; letter-spacing:.15em; font-size:.72rem; font-weight:700; }
    .hero-copy { color:var(--muted); max-width:780px; font-size:1.06rem; line-height:1.7; margin-top:.75rem; }
    .pill { display:inline-block; border:1px solid rgba(71,215,189,.28); color:#a4f4e5; background:rgba(71,215,189,.08); border-radius:99px; padding:.38rem .75rem; font-size:.76rem; font-weight:600; margin:0 .35rem .35rem 0; }
    .panel { border:1px solid var(--line); background:linear-gradient(145deg,rgba(20,34,56,.9),rgba(13,23,39,.8)); border-radius:18px; padding:1.2rem 1.3rem; }
    .subtle { color:var(--muted); font-size:.9rem; }
    .metric-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.1em; }
    .stMetric { border:1px solid var(--line); background:rgba(15,26,44,.8); border-radius:14px; padding:1rem 1.1rem; }
    .stMetric label { color:var(--muted) !important; }
    .stMetric [data-testid="stMetricValue"] { color:var(--ink); font-family:'Manrope',sans-serif; }
    .stButton>button, .stDownloadButton>button { border-radius:10px; border:1px solid rgba(71,215,189,.35); background:linear-gradient(135deg,#149f91,#087b79); color:white; font-weight:700; }
    .stButton>button:hover, .stDownloadButton>button:hover { border-color:#7df4dd; color:white; }
    .stTabs [data-baseweb="tab-list"] { gap:.5rem; border-bottom:1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { padding:.7rem 1rem; color:var(--muted); }
    .stTabs [aria-selected="true"] { color:#a4f4e5 !important; }
    [data-testid="stForm"] { border:1px solid var(--line); background:rgba(15,26,44,.58); border-radius:16px; padding:1rem; }
    .stAlert { border-radius:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_demo_data() -> pd.DataFrame:
    return make_demo_dataset()


@st.cache_resource(show_spinner="Training the demo models…")
def load_model_suite():
    return train_model_suite(load_demo_data())


def money(value: float) -> str:
    return f"${value:,.0f}"


suite = load_model_suite()
demo_data = load_demo_data()
model_names = list(suite.models)

with st.sidebar:
    st.markdown("### ◈ ClearRisk")
    st.caption("Explainable decision-support demo")
    st.divider()
    selected_model_name = st.selectbox(
        "Scoring model",
        model_names,
        index=model_names.index("XGBoost") if "XGBoost" in model_names else model_names.index("Random Forest"),
    )
    st.markdown("#### Risk appetite")
    risk_cutoff_pct = st.slider(
        "Maximum modeled default risk for demo eligibility",
        min_value=5.0,
        max_value=70.0,
        value=25.0,
        step=1.0,
        format="%.0f%%",
        help="A higher cutoff includes more profiles in the simulated approval group and can also admit more defaults.",
    )
    risk_cutoff = risk_cutoff_pct / 100
    st.divider()
    st.markdown("**Evaluation setup**")
    st.caption("Fixed 75/25 stratified split · random seed 42 · held-out metrics")
    st.markdown("**Data**")
    st.caption("Programmatically generated synthetic profiles; no real applicant data")
    st.download_button(
        "Download synthetic demo data",
        data=demo_data.to_csv(index=False).encode("utf-8"),
        file_name="synthetic_credit_risk_demo.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Model transparency · decision trade-offs · human review</div>
      <h1>Credit risk, made<br><span style="color:#47d7bd">clear enough to question.</span></h1>
      <div class="hero-copy">Explore how a modeled default-risk score changes with an applicant profile, compare three classification approaches, and inspect feature-level SHAP explanations. Every score and outcome in this portfolio demo comes from synthetic data.</div>
      <div style="margin-top:1rem"><span class="pill">Logistic Regression</span><span class="pill">Random Forest</span><span class="pill">XGBoost</span><span class="pill">SHAP explanations</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_review, tab_models, tab_threshold, tab_method = st.tabs(
    ["Applicant review", "Model comparison", "Threshold lab", "Method & limits"]
)

with tab_review:
    st.markdown("## Applicant review studio")
    st.markdown(
        '<p class="subtle">Change a profile to see one model score, a simulated cutoff outcome, and the strongest local feature contributions.</p>',
        unsafe_allow_html=True,
    )
    input_col, result_col = st.columns([1.02, 1], gap="large")

    with input_col:
        st.markdown('<div class="panel"><b>Applicant profile</b><br><span class="subtle">All values are fictional and used only for this demo.</span></div>', unsafe_allow_html=True)
        with st.form("applicant_profile"):
            c1, c2 = st.columns(2)
            with c1:
                annual_income = st.number_input("Annual income ($)", 18_000, 240_000, 68_000, step=1_000)
                loan_amount = st.number_input("Requested loan ($)", 1_000, 250_000, 18_000, step=1_000)
                dti_pct = st.slider("Debt-to-income ratio", 0, 90, 28, 1, format="%d%%")
                employment_years = st.slider("Employment history (years)", 0.0, 35.0, 4.0, 0.5)
            with c2:
                credit_history_years = st.slider("Credit history (years)", 0.0, 35.0, 6.0, 0.5)
                prior_delinquencies = st.selectbox("Prior delinquencies", [0, 1, 2, 3, 4, 5], index=0)
                savings_balance = st.number_input("Savings balance ($)", 0, 180_000, 14_000, step=1_000)
                loan_term_months = st.select_slider("Loan term", options=[12, 24, 36, 48, 60], value=36, format_func=lambda value: f"{value} months")
            submitted = st.form_submit_button("Score this demo profile", use_container_width=True)

    if submitted:
        profile = pd.DataFrame(
            [
                {
                    "annual_income": float(annual_income),
                    "loan_amount": float(loan_amount),
                    "debt_to_income_ratio": float(dti_pct) / 100,
                    "employment_years": float(employment_years),
                    "credit_history_years": float(credit_history_years),
                    "prior_delinquencies": int(prior_delinquencies),
                    "savings_balance": float(savings_balance),
                    "loan_term_months": int(loan_term_months),
                }
            ]
        )
        st.session_state["profile"] = profile
    elif "profile" not in st.session_state:
        st.session_state["profile"] = pd.DataFrame(
            [
                {
                    "annual_income": 68_000.0,
                    "loan_amount": 18_000.0,
                    "debt_to_income_ratio": 0.28,
                    "employment_years": 4.0,
                    "credit_history_years": 6.0,
                    "prior_delinquencies": 0,
                    "savings_balance": 14_000.0,
                    "loan_term_months": 36,
                }
            ]
        )

    profile = st.session_state["profile"]
    profile_probability = float(positive_class_probability(suite.models[selected_model_name], profile)[0])
    simulated_eligible = profile_probability < risk_cutoff
    with result_col:
        st.markdown("### Model readout")
        score_col, cutoff_col = st.columns(2)
        score_col.metric("Estimated default risk", f"{profile_probability:.1%}")
        cutoff_col.metric("Demo cutoff", f"{risk_cutoff:.0%}", delta="More inclusive" if risk_cutoff >= 0.4 else "Risk guardrail", delta_color="off")
        if simulated_eligible:
            st.success("Included in the simulated eligible group at this cutoff")
        else:
            st.warning("Outside the simulated cutoff — refer for human review")
        st.caption("This is a model demonstration, not a credit decision or lending recommendation.")

        cohort = threshold_summary(suite.y_test, suite.probabilities[selected_model_name], risk_cutoff)
        k1, k2 = st.columns(2)
        k1.metric("Holdout profiles eligible", f"{cohort['approval_rate']:.1%}", help="Share of the synthetic holdout set below the selected risk cutoff.")
        k2.metric("Observed defaults in eligible group", f"{cohort['default_rate']:.1%}", help="Actual synthetic default labels among the eligible holdout profiles.")

        st.markdown("#### What influenced this score?")
        try:
            contributions = explain_default_risk(
                suite.models[selected_model_name],
                profile[FEATURES],
                suite.X_train.sample(min(80, len(suite.X_train)), random_state=42),
            )
            explanation = pd.DataFrame(
                {"Feature": [FEATURE_LABELS[name] for name in FEATURES], "Contribution": contributions}
            ).sort_values("Contribution")
            fig, ax = plt.subplots(figsize=(7, 3.7))
            fig.patch.set_facecolor("#111c30")
            ax.set_facecolor("#111c30")
            colors = ["#47d7bd" if value < 0 else "#ff8f70" for value in explanation["Contribution"]]
            ax.barh(explanation["Feature"], explanation["Contribution"], color=colors, height=0.66)
            ax.axvline(0, color="#8292ad", linewidth=0.8)
            ax.tick_params(colors="#d4dff1", labelsize=9)
            ax.set_xlabel("SHAP contribution toward default class output", color="#9aa9c3", fontsize=9)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(axis="x", color="#34445f", alpha=0.35)
            ax.set_axisbelow(True)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            strongest = explanation.iloc[-1]
            direction = "higher" if strongest["Contribution"] > 0 else "lower"
            st.info(
                f"In this synthetic example, **{strongest['Feature']}** has the largest contribution toward {direction} modeled default risk. SHAP describes this model's output relative to its background profiles; it does not establish cause."
            )
        except Exception as exc:
            st.error(f"The score is available, but this SHAP explanation could not be rendered: {exc}")

with tab_models:
    st.markdown("## Compare the classifiers")
    st.markdown("The table uses the same held-out synthetic profiles for every model. Precision, recall, and F1 use a fixed 0.50 default-probability cutoff so model rows are comparable.")
    display_metrics = suite.metrics.copy()
    numeric_cols = ["Precision", "Recall", "F1", "ROC AUC", "Brier score"]
    styled = display_metrics.style.format({column: "{:.3f}" for column in numeric_cols}).highlight_max(
        subset=["Precision", "Recall", "F1", "ROC AUC"], color="#1a544f"
    ).highlight_min(subset=["Brier score"], color="#1a544f")
    st.dataframe(styled, use_container_width=True, hide_index=True)
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown("### Reading the metrics")
        st.markdown(
            "- **Precision:** among profiles flagged as likely to default, how many have a synthetic default label?\n"
            "- **Recall:** among synthetic defaults, how many did the model flag?\n"
            "- **F1:** a single summary of precision and recall.\n"
            "- **ROC AUC:** how well scores rank default cases above non-default cases across cutoffs.\n"
            "- **Brier score:** probability error; lower is better."
        )
    with right:
        st.markdown("### Comparison guardrail")
        st.markdown('<div class="panel"><b>These are learning metrics, not validation evidence.</b><p class="subtle">The target is generated by a known formula that uses the same fields. Performance on this toy distribution cannot predict performance, fairness, or profitability on real lending data.</p></div>', unsafe_allow_html=True)

with tab_threshold:
    st.markdown("## Find a cutoff that fits the stated trade-off")
    st.markdown("Here, a profile is counted as eligible when its modeled default probability is **below** the risk cutoff. Raising the cutoff generally expands the eligible group; inspect the observed default rate alongside it.")
    curve = threshold_curve(suite.y_test, suite.probabilities[selected_model_name])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor("#111c30")
    ax.set_facecolor("#111c30")
    ax.plot(curve["Risk cutoff"], curve["Approval rate"], marker="o", color="#47d7bd", linewidth=2.5, label="Eligible share")
    ax.plot(curve["Risk cutoff"], curve["Default rate among eligible"], marker="o", color="#ff8f70", linewidth=2.5, label="Defaults within eligible group")
    ax.axvline(risk_cutoff, color="#86a9ff", linestyle="--", alpha=0.85, label=f"Selected cutoff: {risk_cutoff:.0%}")
    ax.set(xlabel="Maximum modeled default risk", ylabel="Share of synthetic holdout profiles", ylim=(0, 1))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.tick_params(colors="#d4dff1")
    ax.xaxis.label.set_color("#9aa9c3")
    ax.yaxis.label.set_color("#9aa9c3")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(color="#34445f", alpha=0.35)
    ax.legend(frameon=False, labelcolor="#d4dff1", ncol=3, loc="upper left")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    current = threshold_summary(suite.y_test, suite.probabilities[selected_model_name], risk_cutoff)
    st.markdown(
        f"At **{risk_cutoff:.0%}**, the synthetic holdout simulation includes **{current['approval_rate']:.1%}** of profiles, with a **{current['default_rate']:.1%}** observed default rate inside that group. These values change with the selected model and are not business forecasts."
    )
    st.dataframe(curve.style.format({"Risk cutoff": "{:.0%}", "Approval rate": "{:.1%}", "Default rate among eligible": "{:.1%}"}), use_container_width=True, hide_index=True)

with tab_method:
    st.markdown("## Method, scope, and responsible use")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Modeling workflow")
        st.markdown(
            "1. Generate a reproducible synthetic dataset with a documented target formula.\n"
            "2. Split once into stratified train and holdout groups (75/25, seed 42).\n"
            "3. Fit Logistic Regression, Random Forest, and XGBoost when installed.\n"
            "4. Compare held-out metrics at a fixed 0.50 cutoff.\n"
            "5. Let the user explore a separate risk cutoff and its cohort trade-off.\n"
            "6. Use SHAP to inspect a single profile's feature contributions."
        )
        st.markdown("### Feature definitions")
        definitions = pd.DataFrame(
            [
                ("Annual income", "Gross annual income in USD"),
                ("Requested loan amount", "Requested principal in USD"),
                ("Debt-to-income ratio", "Monthly debt payments divided by monthly gross income"),
                ("Employment history", "Years in employment"),
                ("Credit history length", "Years represented in the synthetic credit history"),
                ("Prior delinquencies", "Count of simulated past delinquencies"),
                ("Savings balance", "Liquid savings in USD"),
                ("Loan term", "Requested duration in months"),
            ],
            columns=["Field", "Demo definition"],
        )
        st.dataframe(definitions, use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Limits to say out loud")
        st.markdown(
            "- The data and labels are synthetic; there is no real borrower sample or external validation.\n"
            "- The generated target reflects an invented formula, so the metrics are illustrative only.\n"
            "- SHAP explains how this fitted model allocates a score relative to reference profiles; it is not causal evidence.\n"
            "- No protected demographic fields are included, but omitting them does not establish fairness.\n"
            "- Before any real lending use, teams would need authorized representative data, leakage checks, calibration, subgroup and fairness analysis, drift monitoring, legal review, and human governance."
        )
        st.warning("This prototype must not be used to approve, decline, price, or rank real applicants.")
        st.markdown("### Demo facts")
        st.metric("Synthetic profiles", f"{len(demo_data):,}")
        st.metric("Synthetic default share", f"{demo_data[TARGET].mean():.1%}")
        st.caption("Recreate the same data and split with random seed 42. See README.md and docs/DATA_CARD.md for the full walkthrough.")

st.markdown("---")
st.markdown('<div class="subtle">ClearRisk is an educational portfolio prototype. Outputs are simulated and require human judgment; no real applicant data is collected or stored.</div>', unsafe_allow_html=True)
