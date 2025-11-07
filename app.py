# Write a clean Streamlit app.py with no writes to /mnt/data during app runtime.
app_py = r"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="HR Attrition Analytics & Prediction", layout="wide")

# ----------------------------
# Utilities
# ----------------------------
def make_ohe():
    \"\"\"OneHotEncoder that works across sklearn versions.\"\"\"
    try:
        return OneHotEncoder(handle_unknown=\"ignore\", sparse_output=False)  # new sklearn
    except TypeError:
        return OneHotEncoder(handle_unknown=\"ignore\", sparse=False)         # old sklearn

@st.cache_data
def load_data(default_path=\"EA.csv\"):
    try:
        return pd.read_csv(default_path)
    except Exception:
        return None

def ensure_target(df):
    if \"Attrition\" not in df.columns:
        st.error(\"Dataset must contain an 'Attrition' column.\")
        st.stop()

def detect_columns(df):
    cat_cols = [c for c in df.columns if df[c].dtype == \"object\" or str(df[c].dtype).startswith(\"category\")]
    num_cols = [c for c in df.columns if c not in cat_cols and c != \"Attrition\"]
    sat_cols = [c for c in df.columns if (\"Satisfaction\" in c) and (df[c].dtype != \"object\")]
    return cat_cols, num_cols, sat_cols

def apply_filters(df, jobrole_selected, sat_col, sat_range):
    if df is None or df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if jobrole_selected and \"JobRole\" in df.columns:
        mask &= df[\"JobRole\"].isin(jobrole_selected)
    if sat_col and sat_col in df.columns and sat_range:
        low, high = sat_range
        mask &= df[sat_col].between(low, high)
    return df[mask].copy()

def make_preprocess(X):
    cat_cols = [c for c in X.columns if X[c].dtype == \"object\" or str(X[c].dtype).startswith(\"category\")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocess = ColumnTransformer([
        (\"num\", SimpleImputer(strategy=\"median\"), num_cols),
        (\"cat\", Pipeline([(\"imputer\", SimpleImputer(strategy=\"most_frequent\")),
                           (\"onehot\", make_ohe())]), cat_cols)
    ], verbose_feature_names_out=False)
    return preprocess

def train_models(X, y, cv_splits=5, random_state=42):
    preprocess = make_preprocess(X)
    models = {
        \"Decision Tree\": DecisionTreeClassifier(random_state=random_state, max_depth=8, min_samples_split=5, min_samples_leaf=2),
        \"Random Forest\": RandomForestClassifier(random_state=random_state, n_estimators=150, n_jobs=-1),
        \"Gradient Boosting\": GradientBoostingClassifier(random_state=random_state, n_estimators=150, learning_rate=0.1, max_depth=3),
    }
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    fitted = {}
    rows = []
    curves = {}
    classes_order = sorted(y_train.astype(str).unique().tolist())
    pos_label = classes_order[-1]

    for name, clf in models.items():
        pipe = Pipeline([(\"preprocess\", preprocess), (\"clf\", clf)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring=\"roc_auc\", n_jobs=-1)
        pipe.fit(X_train, y_train)

        y_pred_tr = pipe.predict(X_train)
        y_pred_te = pipe.predict(X_test)
        y_proba_te = pipe.predict_proba(X_test)[:, list(pipe.named_steps[\"clf\"].classes_).index(pos_label)]

        rows.append({
            \"Algorithm\": name,
            \"Train Accuracy\": accuracy_score(y_train, y_pred_tr),
            \"Test Accuracy\": accuracy_score(y_test, y_pred_te),
            \"Precision (test)\": precision_score(y_test, y_pred_te, pos_label=pos_label, zero_division=0),
            \"Recall (test)\": recall_score(y_test, y_pred_te, pos_label=pos_label, zero_division=0),
            \"F1-score (test)\": f1_score(y_test, y_pred_te, pos_label=pos_label, zero_division=0),
            \"AUC (test)\": roc_auc_score((y_test.astype(str) == pos_label).astype(int), y_proba_te),
            \"Mean CV AUC (5-fold)\": float(cv_auc.mean())
        })

        cm_tr = confusion_matrix(y_train, y_pred_tr, labels=classes_order)
        cm_te = confusion_matrix(y_test,  y_pred_te, labels=classes_order)

        fitted[name] = {
            \"pipe\": pipe,
            \"classes\": classes_order,
            \"cm_train\": cm_tr,
            \"cm_test\": cm_te,
            \"y_test\": y_test,
            \"y_proba_test\": y_proba_te,
            \"pos_label\": pos_label
        }

        fpr, tpr, _ = roc_curve((y_test.astype(str) == pos_label).astype(int), y_proba_te)
        curves[name] = (fpr, tpr)

    metrics_df = pd.DataFrame(rows).set_index(\"Algorithm\").sort_values(\"AUC (test)\", ascending=False)
    return fitted, metrics_df, curves

def plot_cm(cm, labels, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format=\"d\", colorbar=False)
    ax.set_title(title)
    ax.set_xlabel(\"Predicted\")
    ax.set_ylabel(\"True\")
    st.pyplot(fig)

def plot_roc(curves, metrics_df):
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {\"Decision Tree\": \"tab:blue\", \"Random Forest\": \"tab:orange\", \"Gradient Boosting\": \"tab:green\"}
    for name, (fpr, tpr) in curves.items():
        auc_val = metrics_df.loc[name, \"AUC (test)\"]
        ax.plot(fpr, tpr, label=f\"{name} (AUC={auc_val:.3f})\", color=colors.get(name, None))
    ax.plot([0,1],[0,1], linestyle=\"--\", label=\"Chance\")
    ax.set_xlabel(\"False Positive Rate\")
    ax.set_ylabel(\"True Positive Rate\")
    ax.set_title(\"ROC Curves (Test Set)\")
    ax.legend(loc=\"lower right\")
    st.pyplot(fig)

def feature_importances(pipe):
    clf = pipe.named_steps[\"clf\"]
    pre = pipe.named_steps[\"preprocess\"]
    if not hasattr(clf, \"feature_importances_\"):
        return None
    names = []
    for name, transformer, cols in pre.transformers_:
        if isinstance(transformer, Pipeline):
            last = transformer.steps[-1][1]
            if hasattr(last, \"get_feature_names_out\"):
                part = last.get_feature_names_out(cols)
            else:
                part = np.array(cols, dtype=object)
        else:
            if hasattr(transformer, \"get_feature_names_out\"):
                part = transformer.get_feature_names_out(cols)
            else:
                part = np.array(cols, dtype=object)
        names.extend(part.tolist())
    return pd.DataFrame({\"feature\": names, \"importance\": clf.feature_importances_}).sort_values(\"importance\", ascending=False)

# ----------------------------
# App
# ----------------------------
st.title(\"HR Attrition Analytics & Prediction\")

with st.expander(\"Data source\", expanded=True):
    st.write(\"The app will try to load **EA.csv** from the working directory. If not found, please upload it below.\")
    df_uploaded = st.file_uploader(\"Upload EA.csv (optional)\", type=[\"csv\"])
    if df_uploaded is not None:
        df = pd.read_csv(df_uploaded)
    else:
        df = load_data()

if df is None:
    st.warning(\"Please upload EA.csv to proceed.\")
    st.stop()

ensure_target(df)

# Sidebar Filters
cat_cols_all, num_cols_all, sat_cols_all = detect_columns(df)
st.sidebar.header(\"Filters\")
jobrole_options = sorted(df[\"JobRole\"].dropna().unique().tolist()) if \"JobRole\" in df.columns else []
jobrole_selected = st.sidebar.multiselect(\"JobRole\", options=jobrole_options, default=jobrole_options[:3] if jobrole_options else [])
sat_col = st.sidebar.selectbox(\"Satisfaction column\", sat_cols_all, index=0) if sat_cols_all else None
if sat_col:
    sat_min, sat_max = float(df[sat_col].min()), float(df[sat_col].max())
    sat_range = st.sidebar.slider(\"Satisfaction range\", min_value=float(sat_min), max_value=float(sat_max),
                                  value=(float(sat_min), float(sat_max)), step=1.0)
else:
    sat_range = None
    st.sidebar.info(\"No numeric *Satisfaction* columns detected.\")

df_f = apply_filters(df, jobrole_selected, sat_col, sat_range)

tabs = st.tabs([\"📊 Insights Dashboard\", \"🤖 Modeling (DT / RF / GBRT)\", \"📥 Upload & Predict\"])

# ---------- Tab 1: Insights ----------
with tabs[0]:
    st.subheader(\"Filters Applied\")
    st.caption(f\"Rows after filter: {len(df_f)}\")
    st.dataframe(df_f.head(20))

    st.markdown(\"### 5 Actionable Insights\")

    # 1) Stacked Bar: Attrition by JobRole & OverTime
    if \"JobRole\" in df_f.columns and \"OverTime\" in df_f.columns:
        rate_df = df_f.groupby([\"JobRole\", \"OverTime\"])[\"Attrition\"].apply(lambda s: (s.astype(str)==\"Yes\").mean()).reset_index(name=\"AttritionRate\")
        chart1 = alt.Chart(rate_df).mark_bar().encode(
            x=alt.X(\"JobRole:N\", sort=\"-y\", title=\"Job Role\"),
            y=alt.Y(\"AttritionRate:Q\", title=\"Attrition Rate\"),
            color=alt.Color(\"OverTime:N\"),
            tooltip=[\"JobRole\",\"OverTime\", alt.Tooltip(\"AttritionRate:Q\", format=\".2f\")]
        ).properties(title=\"Attrition Rate by Job Role & Overtime\")
        st.altair_chart(chart1, use_container_width=True)
    else:
        st.info(\"JobRole/OverTime not found for Insight #1.\")

    # 2) Heatmap: Department vs JobLevel
    if \"Department\" in df_f.columns and \"JobLevel\" in df_f.columns:
        hm = df_f.groupby([\"Department\",\"JobLevel\"])[\"Attrition\"].apply(lambda s: (s.astype(str)==\"Yes\").mean()).reset_index(name=\"AttritionRate\")
        chart2 = alt.Chart(hm).mark_rect().encode(
            x=alt.X(\"JobLevel:O\", title=\"Job Level\"),
            y=alt.Y(\"Department:N\", title=\"Department\"),
            color=alt.Color(\"AttritionRate:Q\", scale=alt.Scale(scheme=\"redyellowgreen\"), title=\"Rate\"),
            tooltip=[\"Department\",\"JobLevel\", alt.Tooltip(\"AttritionRate:Q\", format=\".2f\")]
        ).properties(title=\"Heatmap: Attrition Rate by Department & Job Level\")
        st.altair_chart(chart2, use_container_width=True)
    else:
        st.info(\"Department/JobLevel not found for Insight #2.\")

    # 3) Boxplot: MonthlyIncome by Attrition (facet by JobRole)
    if \"MonthlyIncome\" in df_f.columns:
        if \"JobRole\" in df_f.columns:
            chart3 = alt.Chart(df_f).mark_boxplot().encode(
                x=alt.X(\"Attrition:N\", title=\"Attrition\"),
                y=alt.Y(\"MonthlyIncome:Q\", title=\"Monthly Income\"),
                color=\"Attrition:N\",
                column=alt.Column(\"JobRole:N\", header=alt.Header(labelAngle=-30, labelOrient=\"bottom\"))
            ).properties(title=\"Monthly Income by Attrition (Faceted by JobRole)\").resolve_scale(y='independent')
        else:
            chart3 = alt.Chart(df_f).mark_boxplot().encode(
                x=\"Attrition:N\", y=\"MonthlyIncome:Q\", color=\"Attrition:N\"
            ).properties(title=\"Monthly Income by Attrition\")
        st.altair_chart(chart3, use_container_width=True)
    else:
        st.info(\"MonthlyIncome not found for Insight #3.\")

    # 4) Line chart: Attrition rate vs YearsAtCompany (binned)
    if \"YearsAtCompany\" in df_f.columns:
        from pandas.api.types import is_numeric_dtype
        if is_numeric_dtype(df_f[\"YearsAtCompany\"]):
            b = pd.cut(df_f[\"YearsAtCompany\"], bins=10, include_lowest=True)
            br = df_f.groupby(b)[\"Attrition\"].apply(lambda s: (s.astype(str)==\"Yes\").mean()).reset_index(name=\"AttritionRate\")
            br = br.rename(columns={\"YearsAtCompany\": \"YearsAtCompany_bin\"})
            chart4 = alt.Chart(br).mark_line(point=True).encode(
                x=alt.X(\"YearsAtCompany_bin:N\", title=\"Years At Company (bins)\"),
                y=alt.Y(\"AttritionRate:Q\", title=\"Attrition Rate\"),
                tooltip=[alt.Tooltip(\"AttritionRate:Q\", format=\".2f\"), \"YearsAtCompany_bin:N\"]
            ).properties(title=\"Attrition Rate vs Years At Company (Binned)\")
            st.altair_chart(chart4, use_container_width=True)
        else:
            st.info(\"YearsAtCompany is not numeric; skipping Insight #4.\")
    else:
        st.info(\"YearsAtCompany not found for Insight #4.\")

    # 5) Scatter: Age vs MonthlyIncome colored by Attrition
    if \"Age\" in df_f.columns and \"MonthlyIncome\" in df_f.columns:
        chart5 = alt.Chart(df_f).mark_circle(opacity=0.6).encode(
            x=alt.X(\"Age:Q\"),
            y=alt.Y(\"MonthlyIncome:Q\"),
            color=alt.Color(\"Attrition:N\"),
            tooltip=[\"Age\",\"MonthlyIncome\",\"Attrition\",\"JobRole\"]
        ).properties(title=\"Age vs Monthly Income by Attrition\")
        st.altair_chart(chart5, use_container_width=True)
    else:
        st.info(\"Age/MonthlyIncome not found for Insight #5.\")

# ---------- Tab 2: Modeling ----------
with tabs[1]:
    st.subheader(\"Apply Models (DT, RF, GBRT) with 5-fold Stratified CV\")
    st.caption(\"Click **Run Models** to train and evaluate.\")

    if st.button(\"Run Models\"):
        X = df.drop(columns=[\"Attrition\"])
        y = df[\"Attrition\"].astype(str)

        fitted, metrics_df, curves = train_models(X, y, cv_splits=5)

        st.markdown(\"#### Metrics Table\")
        st.dataframe(metrics_df.round(4))

        st.markdown(\"#### Confusion Matrices (Train / Test)\")
        for name, obj in fitted.items():
            c1, c2 = st.columns(2)
            with c1:
                plot_cm(obj[\"cm_train\"], obj[\"classes\"], f\"{name} — Training\")
            with c2:
                plot_cm(obj[\"cm_test\"], obj[\"classes\"], f\"{name} — Testing\")

        st.markdown(\"#### ROC Curve (All Models)\")
        plot_roc(curves, metrics_df)

        st.markdown(\"#### Feature Importances (Top 20)\")
        for name, obj in fitted.items():
            fi = feature_importances(obj[\"pipe\"])
            if fi is not None and not fi.empty:
                st.write(f\"**{name}**\")
                st.bar_chart(fi.head(20).set_index(\"feature\"))
            else:
                st.info(f\"{name}: feature importances not available.\")

        best_alg = metrics_df[\"AUC (test)\"].idxmax()
        st.success(f\"**Recommended model:** {best_alg} (by highest Test AUC)\")

# ---------- Tab 3: Upload & Predict ----------
with tabs[2]:
    st.subheader(\"Upload New Dataset and Predict Attrition\")
    st.caption(\"Train the selected/best model on full current dataset, then predict on the uploaded data.\")

    model_choice = st.selectbox(\"Choose model for prediction\", [\"Random Forest\", \"Gradient Boosting\", \"Decision Tree\"], index=0)

    new_file = st.file_uploader(\"Upload new CSV for prediction\", type=[\"csv\"], key=\"pred_uploader\")
    if st.button(\"Train & Predict on Uploaded Data\"):
        if new_file is None:
            st.warning(\"Please upload a CSV file first.\")
        else:
            df_new = pd.read_csv(new_file)
            st.write(\"Uploaded sample:\", df_new.head(10))

            # Train on full current dataset
            X_full = df.drop(columns=[\"Attrition\"])
            y_full = df[\"Attrition\"].astype(str)

            preprocess = make_preprocess(X_full)

            if model_choice == \"Random Forest\":
                clf = RandomForestClassifier(random_state=42, n_estimators=150, n_jobs=-1)
            elif model_choice == \"Gradient Boosting\":
                clf = GradientBoostingClassifier(random_state=42, n_estimators=150, learning_rate=0.1, max_depth=3)
            else:
                clf = DecisionTreeClassifier(random_state=42, max_depth=8, min_samples_split=5, min_samples_leaf=2)

            pipe = Pipeline([(\"preprocess\", preprocess), (\"clf\", clf)])
            pipe.fit(X_full, y_full)

            # Predict on new data
            X_new = df_new.copy()
            preds = pipe.predict(X_new)
            classes = list(pipe.named_steps[\"clf\"].classes_)
            pos_label = sorted(classes)[-1]
            proba = pipe.predict_proba(X_new)[:, classes.index(pos_label)]

            df_out = df_new.copy()
            df_out[\"Predicted_Attrition\"] = preds
            df_out[\"Attrition_Probability_Positive\"] = proba

            st.success(\"Prediction complete.\")
            st.dataframe(df_out.head(20))

            # Download
            csv_bytes = df_out.to_csv(index=False).encode(\"utf-8\")
            st.download_button(\"Download predictions CSV\", data=csv_bytes, file_name=\"predictions_with_attrition.csv\", mime=\"text/csv\")
"""
with open("/mnt/data/app.py", "w", encoding="utf-8") as f:
    f.write(app_py)
"/mnt/data/app.py"
