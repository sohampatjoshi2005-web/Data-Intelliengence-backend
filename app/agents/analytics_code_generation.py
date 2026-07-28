from __future__ import annotations

import re

from app.agents.base import BaseAgent
from app.graph.state import AgentState
from app.services.analytics_helpers import DEFAULT_FIGSIZE, extract_first_code_block
from app.services.analytics_llm import AnalyticsModelConfig, chat_complete
from app.services.dspy_runtime import dspy_nl_to_sql, dspy_ready


def _unsafe_file_io_in_code(code: str) -> bool:
    patterns = [
        r"\bread_csv\s*\(",
        r"\bread_excel\s*\(",
        r"\bread_json\s*\(",
        r"\bread_table\s*\(",
        r"\bopen\s*\(",
    ]
    return any(re.search(p, code) for p in patterns)


def _extract_sql(text: str) -> str:
    """Best-effort extraction of a single SQL statement."""
    if not text:
        return ""
    cleaned = text.strip()
    # Remove markdown fences if present.
    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
    cleaned = re.sub(r"\n```$", "", cleaned)
    # Find first SELECT/WITH statement.
    match = re.search(r"(?is)\b(select|with)\b.*", cleaned)
    if not match:
        return ""
    sql = match.group(0).strip()
    # Stop at the first code block or extraneous text after a semicolon.
    if ";" in sql:
        sql = sql.split(";", 1)[0] + ";"
    return sql


def _is_structured_report_query(query: str) -> bool:
    q = query.lower()
    markers = [
        "class distribution",
        "summary stats",
        "correlation",
        "anova",
        "outlier",
        "confusion matrix",
        "precision",
        "recall",
        "f1",
        "recommendation",
    ]
    hit_count = sum(1 for m in markers if m in q)
    return hit_count >= 4


def _is_predictive_outcome_query(query: str) -> bool:
    q = query.lower()
    prediction_markers = [
        "forecast",
        "predict",
        "prediction",
        "probability",
        "expected outcome",
        "potential outcome",
        "potential outcomes",
        "likely outcome",
    ]
    target_markers = ["species", "class", "variety", "label", "target"]
    return any(marker in q for marker in prediction_markers) and any(marker in q for marker in target_markers)


def _is_distribution_means_query(query: str) -> bool:
    q = query.lower()
    return (
        ("class distribution" in q or "distribution" in q)
        and "mean" in q
        and ("variety" in q or "class" in q or "species" in q)
    )


def _distribution_means_fallback_code() -> str:
    return (
        "import pandas as pd\n\n"
        "df_work = df.copy()\n"
        "num_cols = df_work.select_dtypes(include=['number']).columns.tolist()\n"
        "cat_cols = [c for c in df_work.columns if c not in num_cols]\n\n"
        "target_col = None\n"
        "for c in df_work.columns:\n"
        "    if str(c).lower() in {'target','label','class','species','variety'}:\n"
        "        target_col = c\n"
        "        break\n"
        "if target_col is None and cat_cols:\n"
        "    target_col = cat_cols[0]\n"
        "if target_col is None:\n"
        "    target_col = df_work.columns[-1]\n\n"
        "df_work = df_work.dropna(subset=[target_col])\n"
        "class_counts = df_work[target_col].value_counts(dropna=False)\n"
        "class_distribution = [\n"
        "    {'class': str(idx), 'count': int(val), 'pct': round(float(val) / max(len(df_work), 1) * 100.0, 2)}\n"
        "    for idx, val in class_counts.items()\n"
        "]\n\n"
        "preferred = [c for c in ['petal.length', 'petal.width', 'petal_length', 'petal_width'] if c in df_work.columns]\n"
        "mean_cols = preferred if len(preferred) >= 1 else num_cols[:2]\n"
        "mean_by_group = []\n"
        "if mean_cols:\n"
        "    means = df_work.groupby(target_col)[mean_cols].mean().reset_index()\n"
        "    for _, row in means.iterrows():\n"
        "        rec = {'group': str(row[target_col])}\n"
        "        for c in mean_cols:\n"
        "            rec[f'{c}_mean'] = float(row[c])\n"
        "        mean_by_group.append(rec)\n\n"
        "result = {\n"
        "    'target_column': str(target_col),\n"
        "    'row_count': int(len(df_work)),\n"
        "    'class_distribution': class_distribution,\n"
        "    'mean_by_group': mean_by_group,\n"
        "}\n"
    )


def _structured_report_fallback_code() -> str:
    return (
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.metrics import classification_report, confusion_matrix, f1_score\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "try:\n"
        "    from scipy.stats import f_oneway\n"
        "except Exception:\n"
        "    f_oneway = None\n\n"
        "df_work = df.copy()\n"
        "num_cols = df_work.select_dtypes(include=['number']).columns.tolist()\n"
        "non_num_cols = [c for c in df_work.columns if c not in num_cols]\n\n"
        "target_col = None\n"
        "for c in df_work.columns:\n"
        "    if str(c).lower() in {'target','label','class','species','variety'}:\n"
        "        target_col = c\n"
        "        break\n"
        "if target_col is None and non_num_cols:\n"
        "    target_col = non_num_cols[0]\n"
        "if target_col is None:\n"
        "    target_col = df_work.columns[-1]\n\n"
        "df_work = df_work.dropna(subset=[target_col])\n"
        "class_counts = df_work[target_col].value_counts(dropna=False)\n"
        "class_distribution = [\n"
        "    {'class': str(idx), 'count': int(val), 'pct': round(float(val) / max(len(df_work), 1) * 100.0, 2)}\n"
        "    for idx, val in class_counts.items()\n"
        "]\n\n"
        "if num_cols:\n"
        "    per_class_stats_df = df_work.groupby(target_col)[num_cols].agg(['mean', 'std']).round(4)\n"
        "    per_class_stats = []\n"
        "    for cls in per_class_stats_df.index:\n"
        "        row = {'class': str(cls)}\n"
        "        for feature in num_cols:\n"
        "            row[feature] = {\n"
        "                'mean': float(per_class_stats_df.loc[cls, (feature, 'mean')]),\n"
        "                'std': float(per_class_stats_df.loc[cls, (feature, 'std')]),\n"
        "            }\n"
        "        per_class_stats.append(row)\n"
        "else:\n"
        "    per_class_stats = []\n\n"
        "correlations = []\n"
        "if len(num_cols) >= 2:\n"
        "    corr = df_work[num_cols].corr()\n"
        "    pairs = []\n"
        "    for i, c1 in enumerate(num_cols):\n"
        "        for c2 in num_cols[i + 1:]:\n"
        "            val = corr.loc[c1, c2]\n"
        "            if pd.notna(val):\n"
        "                pairs.append({'pair': f'{c1}__{c2}', 'corr': float(val), 'abs_corr': float(abs(val))})\n"
        "    correlations = sorted(pairs, key=lambda x: x['abs_corr'], reverse=True)[:5]\n\n"
        "anova = []\n"
        "if f_oneway is not None and num_cols:\n"
        "    for feature in num_cols:\n"
        "        groups = [g[feature].dropna().values for _, g in df_work.groupby(target_col) if len(g[feature].dropna()) > 1]\n"
        "        if len(groups) >= 2:\n"
        "            stat, pval = f_oneway(*groups)\n"
        "            anova.append({'feature': feature, 'f_stat': float(stat), 'p_value': float(pval)})\n\n"
        "iqr_outliers = []\n"
        "for cls, grp in df_work.groupby(target_col):\n"
        "    rec = {'class': str(cls)}\n"
        "    for feature in num_cols:\n"
        "        s = grp[feature].dropna()\n"
        "        if len(s) < 4:\n"
        "            rec[feature] = 0\n"
        "            continue\n"
        "        q1 = s.quantile(0.25)\n"
        "        q3 = s.quantile(0.75)\n"
        "        iqr = q3 - q1\n"
        "        lb = q1 - 1.5 * iqr\n"
        "        ub = q3 + 1.5 * iqr\n"
        "        rec[feature] = int(((s < lb) | (s > ub)).sum())\n"
        "    iqr_outliers.append(rec)\n\n"
        "model_metrics = {}\n"
        "if len(df_work[target_col].astype(str).unique()) >= 2 and num_cols:\n"
        "    X = df_work[num_cols].copy()\n"
        "    y = df_work[target_col].astype(str)\n"
        "    stratify = y if y.nunique() > 1 else None\n"
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)\n"
        "    clf = LogisticRegression(max_iter=1000)\n"
        "    clf.fit(X_train, y_train)\n"
        "    y_pred = clf.predict(X_test)\n"
        "    labels = sorted(y.unique().tolist())\n"
        "    cm = confusion_matrix(y_test, y_pred, labels=labels)\n"
        "    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)\n"
        "    per_class = []\n"
        "    for label in labels:\n"
        "        entry = report.get(label, {})\n"
        "        per_class.append({\n"
        "            'class': str(label),\n"
        "            'precision': float(entry.get('precision', 0.0)),\n"
        "            'recall': float(entry.get('recall', 0.0)),\n"
        "            'f1': float(entry.get('f1-score', 0.0)),\n"
        "            'support': int(entry.get('support', 0)),\n"
        "        })\n"
        "    model_metrics = {\n"
        "        'labels': labels,\n"
        "        'confusion_matrix': cm.tolist(),\n"
        "        'per_class': per_class,\n"
        "        'macro_f1': float(f1_score(y_test, y_pred, average='macro')),\n"
        "        'test_rows': int(len(y_test)),\n"
        "    }\n\n"
        "conclusions = []\n"
        "if class_distribution:\n"
        "    cls_txt = ', '.join([f\"{d['class']}={d['count']}\" for d in class_distribution])\n"
        "    conclusions.append(f'Class distribution: {cls_txt}.')\n"
        "if correlations:\n"
        "    top = correlations[0]\n"
        "    conclusions.append(f\"Strongest correlation is {top['pair']} ({top['corr']:.4f}).\")\n"
        "if anova:\n"
        "    sig = [a['feature'] for a in anova if a['p_value'] < 0.05]\n"
        "    conclusions.append(f'Significant ANOVA features (p<0.05): {sig}.')\n"
        "if model_metrics:\n"
        "    conclusions.append(f\"Baseline macro F1: {model_metrics['macro_f1']:.4f} on {model_metrics['test_rows']} test rows.\")\n"
        "if iqr_outliers:\n"
        "    conclusions.append('IQR outlier counts computed per class and numeric feature.')\n"
        "while len(conclusions) < 5:\n"
        "    conclusions.append('No additional statistically strong signal found beyond reported metrics.')\n"
        "conclusions = conclusions[:5]\n\n"
        "recommendations = [\n"
        "    'Validate baseline with stratified k-fold cross-validation and report macro F1 variance.',\n"
        "    'Compare linear baseline with tree-based models and calibrate probabilities.',\n"
        "    'Track per-class recall to avoid regressions in minority class performance.',\n"
        "]\n\n"
        "result = {\n"
        "    'target_column': str(target_col),\n"
        "    'row_count': int(len(df_work)),\n"
        "    'numeric_columns': [str(c) for c in num_cols],\n"
        "    'class_distribution': class_distribution,\n"
        "    'per_class_stats': per_class_stats,\n"
        "    'top_correlations': correlations,\n"
        "    'anova': anova,\n"
        "    'iqr_outliers': iqr_outliers,\n"
        "    'model_metrics': model_metrics,\n"
        "    'conclusions': conclusions,\n"
        "    'recommendations': recommendations,\n"
        "}\n"
    )


class AnalyticsCodeGenerationAgent(BaseAgent):
    name = "analytics_code_generation"

    def run(self, state: AgentState) -> AgentState:
        df = state["dataframe"]
        query = state.get("analytics_query", "")
        chat_context = state.get("analytics_chat_context", "")
        should_plot = state.get("analytics_should_plot", False)
        force_sql = bool(state.get("analytics_force_sql", False))

        cols = ", ".join(df.columns.tolist())
        if force_sql:
            sql = ""
            if dspy_ready():
                sql = dspy_nl_to_sql(query, df.columns.tolist(), table_name="data")
            if not sql:
                instruction = (
                    f"You are a SQL generator. Table name is data. Columns: {cols}. "
                    f"Write a single DuckDB SQL query to answer: '{query}'. "
                    "Return only SQL without markdown."
                )
                try:
                    cfg = AnalyticsModelConfig()
                    sql = chat_complete(
                        prompt=instruction,
                        temperature=0.0,
                        max_tokens=512,
                        trace_name="analytics_code_generation_sql",
                        provider=state.get("llm_provider", ""),
                    ).strip()
                except Exception:
                    sql = ""
            sql = _extract_sql(sql) or f"SELECT * FROM data LIMIT 100"
            state["analytics_code"] = sql
            state["analytics_should_plot"] = False
            return state
        if should_plot:
            instruction = (
                f"DataFrame df has columns: {cols}. Write Python pandas + matplotlib code to answer: '{query}'. "
                f"Assign final output to variable result. Set figsize={DEFAULT_FIGSIZE}. "
                "Code must be directly executable, avoid markdown prose, avoid print statements, and always assign `result`. "
                "Do not read files, do not call read_csv/read_excel/open, and do not create a new dataframe from disk. "
                "Use only the provided dataframe variable `df`. "
                "Return a JSON-serializable object in `result` (dict/list, not free-form prose). "
                "Return only one markdown python code block."
            )
        else:
            instruction = (
                f"DataFrame df has columns: {cols}. Write Python pandas code (no plotting) to answer: '{query}'. "
                "Code must be directly executable, avoid markdown prose, avoid print statements, and always assign `result`. "
                "Do not read files, do not call read_csv/read_excel/open, and do not create a new dataframe from disk. "
                "Use only the provided dataframe variable `df`. "
                "Return a JSON-serializable object in `result` (dict/list, not free-form prose). "
                "Return only one markdown python code block."
            )

        full_prompt = (
            "You are a senior data analyst.\n"
            f"Recent Context: {chat_context}\n"
            f"{instruction}"
        )

        code = ""
        try:
            cfg = AnalyticsModelConfig()
            content = chat_complete(
                prompt=full_prompt,
                temperature=cfg.code_generation_temperature,
                max_tokens=cfg.code_generation_max_tokens,
                trace_name="analytics_code_generation",
                provider=state.get("llm_provider", ""),
            )
            code = extract_first_code_block(content)
        except Exception:
            code = ""

        if _is_distribution_means_query(query):
            code = _distribution_means_fallback_code()
        elif _is_predictive_outcome_query(query):
            code = _structured_report_fallback_code()
        elif _is_structured_report_query(query):
            code = _structured_report_fallback_code()
        elif not code or _unsafe_file_io_in_code(code):
            if should_plot:
                code = (
                    "import matplotlib.pyplot as plt\n"
                    "num_cols = df.select_dtypes(include=['number']).columns.tolist()\n"
                    "if len(num_cols) >= 1:\n"
                    "    fig, ax = plt.subplots(figsize=(6, 4))\n"
                    "    df[num_cols[0]].dropna().plot(kind='hist', ax=ax, title=num_cols[0])\n"
                    "    result = fig\n"
                    "else:\n"
                    "    result = {'message': 'No numeric columns available for plotting.'}\n"
                )
            else:
                code = (
                    "desc = df.describe(include='all').transpose().reset_index().rename(columns={'index': 'feature'})\n"
                    "result = {'summary': desc.head(100).to_dict(orient='records')}\n"
                )

        state["analytics_code"] = code
        return state
