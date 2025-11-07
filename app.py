import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_fscore_support, accuracy_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Personal Loan Propensity – Universal Bank", layout="wide")

# ---------------------- Utilities ----------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={c: c.replace(" ", "") for c in df.columns}, inplace=True)
    return df

def pick(df: pd.DataFrame, name: str):
    targets = [name, name.replace(" ", ""), name.lower(), name.upper(), name.title(), name.replace(" ", "").lower()]
    for v in targets:
        for c in df.columns:
            if c == v or c.lower() == v.lower():
                return c
    for c in df.columns:
        if c.lower().replace("_","").replace(" ","") == name.lower().replace("_","").replace(" ",""):
            return c
    return None

def load_data():
    st.sidebar.markdown("### Data")
    up = st.sidebar.file_uploader("Upload CSV (UniversalBank.csv)", type=["csv"])
    if up is not None:
        return pd.read_csv(up)
    try:
        return pd.read_csv("UniversalBank.csv")
    except Exception:
        st.info("Upload **UniversalBank.csv** (or keep it in repo root) to proceed.")
        return None

def build_xy(df):
    df = normalize_cols(df)
    ID = pick(df, "ID")
    ycol = pick(df, "Personal Loan") or pick(df, "PersonalLoan")
    if ID is None or ycol is None:
        st.error("ID / Personal Loan column not found. Check column names.")
        return None, None, None, None
    X = df.drop(columns=[ID, ycol], errors="ignore")
    y = df[ycol]
    return X, y, ID, ycol

def make_preprocess(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    preprocess = ColumnTransformer([("num", SimpleImputer(strategy="median"), num_cols)], remainder="drop")
    return preprocess, num_cols

def train_all(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    preprocess, num_cols = make_preprocess(X_train)
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    roc_cache, metrics_rows, fitted = {}, [], {}

    for name, base in models.items():
        pipe = Pipeline([("prep", preprocess), ("clf", base)])
        y_train_proba_oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:,1]
        y_train_pred_oof = (y_train_proba_oof >= 0.5).astype(int)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        y_test_proba = pipe.predict_proba(X_test)[:,1]
        y_test_pred = (y_test_proba >= 0.5).astype(int)

        train_acc = accuracy_score(y_train, y_train_pred_oof)
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(y_train, y_train_pred_oof, average='binary', zero_division=0)
        train_auc = roc_auc_score(y_train, y_train_proba_oof)

        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='binary', zero_division=0)
        test_auc = roc_auc_score(y_test, y_test_proba)

        metrics_rows.append({
            "Algorithm": name,
            "Training Accuracy": round(train_acc, 4),
            "Testing Accuracy": round(test_acc, 4),
            "Precision": round(test_prec, 4),
            "Recall": round(test_rec, 4),
            "F1-Score": round(test_f1, 4),
            "AUC": round(test_auc, 4),
        })

        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_cache[name] = (fpr, tpr, test_auc, pipe, y_train_pred_oof, y_test_pred, X_train.columns.tolist())

    metrics_df = pd.DataFrame(metrics_rows).set_index("Algorithm")
    return fitted, metrics_df, roc_cache, (X_train, X_test, y_train, y_test)

def plot_confusion(cm, title):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ticklabels = ["No Loan (0)", "Loan (1)"]
    ax.set_xticks([0,1], ticklabels); ax.set_yticks([0,1], ticklabels)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Count"); fig.tight_layout()
    return fig

# session state
for k in ["models","splits","metrics_df","roc_cache"]:
    if k not in st.session_state: st.session_state[k] = None

# ---------------------- UI ----------------------
st.title("💳 Personal Loan Propensity – Streamlit Dashboard")
st.caption("Explore drivers and predict acceptance of Personal Loans")

df = load_data()
if df is not None:
    st.success(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

tabs = st.tabs(["📊 Insights (5 complex charts)", "🤖 Modeling (DT / RF / GBDT)", "📥 Predict on New Data"])

# ---------------------- Insights ----------------------
with tabs[0]:
    st.subheader("Exploratory insights for conversion strategy")
    if df is None:
        st.info("Upload data to view insights.")
    else:
        X, y, ID, ycol = build_xy(df)
        if X is None: st.stop()
        work = normalize_cols(df).copy()
        tgt = pick(work, "Personal Loan") or pick(work, "PersonalLoan")
        def c(name): return pick(work, name)

        # 1) Acceptance rate by Income decile
        st.markdown("**1) Acceptance rate by Income decile**")
        w = work[[c("Income"), tgt]].dropna()
        w["inc_decile"] = pd.qcut(w[c("Income")].rank(method="first"), 10, labels=[f"D{i}" for i in range(1,11)])
        rate = w.groupby("inc_decile")[tgt].mean().reset_index().rename(columns={tgt:"accept_rate"})
        fig1, ax1 = plt.subplots(); ax1.bar(rate["inc_decile"], rate["accept_rate"])
        ax1.set_title("Acceptance Rate by Income Decile"); ax1.set_xlabel("Income Decile (low → high)"); ax1.set_ylabel("Acceptance Rate")
        st.pyplot(fig1)

        # 2) Heatmap: Income decile × CCAvg decile → acceptance rate
        st.markdown("**2) Heatmap: Income decile × CCAvg decile → acceptance rate**")
        w2 = work[[c("Income"), c("CCAvg"), tgt]].dropna()
        w2["inc_d"] = pd.qcut(w2[c("Income")].rank(method="first"), 10, labels=[f"D{i}" for i in range(1,11)])
        w2["cc_d"]  = pd.qcut(w2[c("CCAvg")].rank(method="first"), 10, labels=[f"D{i}" for i in range(1,11)])
        pivot = w2.pivot_table(index="inc_d", columns="cc_d", values=tgt, aggfunc="mean")
        fig2, ax2 = plt.subplots(); im = ax2.imshow(pivot.values, cmap="viridis")
        ax2.set_title("Acceptance Rate by Income & CCAvg Deciles"); ax2.set_xlabel("CCAvg Decile"); ax2.set_ylabel("Income Decile")
        ax2.set_xticks(range(len(pivot.columns)), pivot.columns); ax2.set_yticks(range(len(pivot.index)), pivot.index)
        for (i,j), v in np.ndenumerate(pivot.values): ax2.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig2.colorbar(im, ax=ax2, label="Accept Rate"); st.pyplot(fig2)

        # 3) Education × CDAccount segmentation
        st.markdown("**3) Education × CDAccount segmentation (acceptance rate)**")
        ed, cd = c("Education"), c("CDAccount")
        w3 = work[[ed, cd, tgt]].dropna()
        seg = w3.groupby([ed, cd])[tgt].mean().reset_index()
        seg["label"] = seg[ed].astype(str) + " / CD=" + seg[cd].astype(str)
        fig3, ax3 = plt.subplots(); ax3.bar(seg["label"], seg[tgt])
        ax3.set_title("Acceptance Rate by Education level and CD Account"); ax3.set_xlabel("Education / CDAccount"); ax3.set_ylabel("Acceptance Rate")
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right"); st.pyplot(fig3)

        # 4) Precision & Recall vs Threshold (quick GBDT)
        st.markdown("**4) Precision & Recall vs Threshold (quick GBDT)**")
        Xs, ys, _, _ = build_xy(work); preprocess, _ = make_preprocess(Xs)
        model = Pipeline([("prep", preprocess), ("clf", GradientBoostingClassifier(random_state=42))])
        X_tr, X_te, y_tr, y_te = train_test_split(Xs, ys, test_size=0.2, random_state=42, stratify=ys)
        model.fit(X_tr, y_tr); prob = model.predict_proba(X_te)[:,1]
        thresholds = np.linspace(0.05, 0.95, 19); precs, recs = [], []
        for th in thresholds:
            pred = (prob>=th).astype(int)
            p, r, _, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
            precs.append(p); recs.append(r)
        fig4, ax4 = plt.subplots(); ax4.plot(thresholds, precs, label="Precision"); ax4.plot(thresholds, recs, label="Recall")
        ax4.set_title("Precision/Recall vs Threshold (Gradient Boosting)"); ax4.set_xlabel("Decision Threshold"); ax4.set_ylabel("Score"); ax4.legend()
        st.pyplot(fig4)

        # 5) Cumulative Gains / Lift curve
        st.markdown("**5) Cumulative Gains / Lift (by scored deciles)**")
        dfp = pd.DataFrame({"y": y_te, "p": prob}).sort_values("p", ascending=False).reset_index(drop=True)
        dfp["decile"] = pd.qcut(np.arange(len(dfp))+1, 10, labels=[f"D{i}" for i in range(1,11)])
        gains = dfp.groupby("decile")["y"].sum().cumsum() / dfp["y"].sum()
        random_line = np.linspace(0.1, 1.0, 10)
        fig5, ax5 = plt.subplots(); ax5.plot(np.arange(1,11)/10.0, gains.values, label="Model")
        ax5.plot(np.arange(1,11)/10.0, random_line, "--", label="Random")
        ax5.set_title("Cumulative Gains / Lift (Top X% by score)"); ax5.set_xlabel("Top X% targeted"); ax5.set_ylabel("Cumulative share of responders"); ax5.legend()
        st.pyplot(fig5)

# ---------------------- Modeling ----------------------
with tabs[1]:
    st.subheader("Train & evaluate models (DT / RF / GBDT)")
    if df is None:
        st.info("Upload data to run models.")
    else:
        X, y, ID, ycol = build_xy(df)
        if X is None: st.stop()
        if st.button("Run models (cv=5, stratified)"):
            with st.spinner("Training models..."):
                fitted, metrics_df, roc_cache, splits = train_all(X, y)
                st.session_state.models = fitted
                st.session_state.metrics_df = metrics_df
                st.session_state.roc_cache = roc_cache
                st.session_state.splits = splits

        if st.session_state.metrics_df is not None:
            st.success("Training complete.")
            st.dataframe(st.session_state.metrics_df)

            st.markdown("**ROC Curves (Test Set)**")
            figr, axr = plt.subplots()
            for name, (fpr, tpr, auc_val, _, _, _, _) in st.session_state.roc_cache.items():
                axr.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
            axr.plot([0,1], [0,1], "--", label="Chance")
            axr.set_xlabel("False Positive Rate"); axr.set_ylabel("True Positive Rate")
            axr.set_title("ROC – All Models"); axr.legend(loc="lower right")
            st.pyplot(figr)

            for name, (_, _, _, pipe, ytr_pred_oof, yte_pred, feat_cols) in st.session_state.roc_cache.items():
                st.markdown(f"**{name} — Confusion Matrices**")
                y_train = st.session_state.splits[2]; y_test = st.session_state.splits[3]
                cm_train = confusion_matrix(y_train, ytr_pred_oof, labels=[0,1])
                cm_test = confusion_matrix(y_test, yte_pred, labels=[0,1])
                st.pyplot(plot_confusion(cm_train, f"{name} — Training (OOF cv=5)"))
                st.pyplot(plot_confusion(cm_test, f"{name} — Testing (Holdout)"))

                st.markdown(f"**{name} — Feature Importances**")
                clf = pipe.named_steps["clf"]
                if hasattr(clf, "feature_importances_"):
                    imp = clf.feature_importances_
                    order = np.argsort(imp)[::-1]
                    figf, axf = plt.subplots()
                    axf.bar(np.array(feat_cols)[order], imp[order])
                    plt.setp(axf.get_xticklabels(), rotation=90, ha="right")
                    axf.set_ylabel("Importance")
                    st.pyplot(figf)
                else:
                    st.info("This estimator does not expose feature_importances_.")

# ---------------------- Predict ----------------------
with tabs[2]:
    st.subheader("Upload new data → predict Personal Loan, then download")
    uploaded = st.file_uploader("Upload new CSV for scoring", type=["csv"], key="predict_upl")
    if uploaded is not None:
        newdf = pd.read_csv(uploaded)
        temp = newdf.copy()
        if pick(temp, "Personal Loan") is None and pick(temp, "PersonalLoan") is None:
            temp["PersonalLoan"] = 0
        Xnew, _, IDnew, _ = build_xy(temp)
        if Xnew is None: st.stop()

        if st.session_state.models is not None and st.session_state.metrics_df is not None:
            best = st.session_state.metrics_df["AUC"].idxmax()
            model = st.session_state.models[best]
            st.info(f"Using trained **{best}** from Modeling tab.")
        else:
            st.warning("Models not trained in this session; fitting a quick Gradient Boosting model on all rows.")
            Xall, yall, _, _ = build_xy(df)
            preprocess, _ = make_preprocess(Xall)
            model = Pipeline([("prep", preprocess), ("clf", GradientBoostingClassifier(random_state=42))])
            model.fit(Xall, yall)

        proba = model.predict_proba(Xnew)[:,1]
        pred  = (proba >= 0.5).astype(int)
        out = newdf.copy(); out["Predicted_PersonalLoan"] = pred; out["Score_Prob"] = proba
        st.dataframe(out.head(50))
        st.download_button("Download predictions CSV", data=out.to_csv(index=False).encode("utf-8"),
                           file_name="predictions_with_scores.csv", mime="text/csv")

st.caption("© Streamlit demo for Universal Bank – built for the Head of Marketing")
