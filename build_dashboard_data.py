"""
Génération du fichier dashboard_data.json pour le dashboard Seattle Collision Risk.

Ce script remplace l'exécution manuelle du notebook Jupyter.
Il reprend la logique du notebook :
1. charge le jeu de données Data-Collisions.csv, localement ou depuis l'URL IBM/Cognitive Class ;
2. construit un jeu analytique par LOCATION + date, incluant les jours sans collision observée ;
3. entraîne et compare trois modèles : régression logistique, forêt aléatoire et Extra Trees ;
4. retient automatiquement le meilleur modèle selon le f1_macro ;
5. entraîne un modèle conditionnel de gravité pour les classes 1 et 2 ;
6. produit dashboard_data.json dans le même dossier que ce script.

Exécution :
    python build_dashboard_data.py

Fichiers attendus dans le même dossier :
    - Data-Collisions.csv, facultatif. Si absent, le script tente de le télécharger.

Fichier produit :
    - dashboard_data.json
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Paramètres principaux
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
LOCAL_FILE = BASE_DIR / "Data-Collisions.csv"
OUTPUT_JSON = BASE_DIR / "dashboard_data.json"

DATA_URL = (
    "https://s3.us.cloud-object-storage.appdomain.cloud/"
    "cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv"
)

RANDOM_STATE = 42
MAX_HISTORY_YEARS = 5
TOP_N_LOCATIONS_MODEL = 50

# Pour garder le JSON raisonnable, le dashboard peut exposer moins de localisations
# que le modèle n'en utilise pendant l'entraînement. Vous pouvez augmenter cette
# valeur si vous acceptez un fichier dashboard_data.json plus volumineux.
TOP_N_LOCATIONS_DASHBOARD = 25
MAX_VALUES_PER_CONTEXT = 4
TOP_N_DISPLAY = 5

REQUESTED_COLUMNS = [
    "SEVERITYCODE",
    "INCDATE",
    "INCDTTM",
    "LOCATION",
    "WEATHER",
    "ROADCOND",
    "LIGHTCOND",
    "ADDRTYPE",
    "JUNCTIONTYPE",
]

CONTEXT_COLS = ["WEATHER", "ROADCOND", "LIGHTCOND", "ADDRTYPE", "JUNCTIONTYPE"]
FEATURES_CAT = ["LOCATION", "jour_semaine", "saison"] + CONTEXT_COLS
FEATURES_NUM = ["mois", "weekend", "jour_annee", "sin_jour_annee", "cos_jour_annee"]
FEATURES = FEATURES_CAT + FEATURES_NUM

CLASS_LABELS = {
    0: "Aucune collision observée",
    1: "Collision avec dommages matériels",
    2: "Collision avec blessures",
}

JOUR_FR = {
    0: "Lundi",
    1: "Mardi",
    2: "Mercredi",
    3: "Jeudi",
    4: "Vendredi",
    5: "Samedi",
    6: "Dimanche",
}

# Valeurs importantes à garder dans le dashboard même si elles ne font pas partie
# des premières valeurs les plus fréquentes.
PREFERRED_VALUES = {
    "WEATHER": ["Clear", "Raining", "Overcast", "Snowing"],
    "ROADCOND": ["Dry", "Wet", "Ice", "Snow/Slush"],
    "LIGHTCOND": ["Daylight", "Dark - Street Lights On", "Dusk", "Dawn"],
    "ADDRTYPE": ["Intersection", "Block", "Alley", "Unknown"],
    "JUNCTIONTYPE": [
        "At Intersection (intersection related)",
        "Mid-Block (not related to intersection)",
        "Mid-Block (but intersection related)",
        "Driveway Junction",
    ],
}


def log(message: str) -> None:
    print(f"[build] {message}")


def mode_value(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return "Unknown"
    modes = values.mode()
    return str(modes.iloc[0]) if not modes.empty else "Unknown"


def make_encoder(*, sparse: bool = True) -> OneHotEncoder:
    """Compatibilité avec plusieurs versions de scikit-learn."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=10,
            sparse_output=sparse,
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse)


def season_fr(month: int) -> str:
    if month in [12, 1, 2]:
        return "Hiver"
    if month in [3, 4, 5]:
        return "Printemps"
    if month in [6, 7, 8]:
        return "Été"
    return "Automne"


def add_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["incident_date"] = pd.to_datetime(data["incident_date"], errors="coerce")
    data["incident_date"] = data["incident_date"].fillna(pd.Timestamp.today().normalize())

    data["jour_semaine_num"] = data["incident_date"].dt.dayofweek
    data["jour_semaine"] = data["jour_semaine_num"].map(JOUR_FR)
    data["mois"] = data["incident_date"].dt.month
    data["saison"] = data["mois"].apply(season_fr)
    data["weekend"] = data["jour_semaine_num"].isin([5, 6]).astype(int)
    data["jour_annee"] = data["incident_date"].dt.dayofyear
    data["sin_jour_annee"] = np.sin(2 * np.pi * data["jour_annee"] / 365.25)
    data["cos_jour_annee"] = np.cos(2 * np.pi * data["jour_annee"] / 365.25)
    return data


def load_source_data() -> tuple[pd.DataFrame, str]:
    source = LOCAL_FILE if LOCAL_FILE.exists() else DATA_URL
    log(f"Chargement des données depuis : {source}")
    try:
        df_raw = pd.read_csv(
            source,
            usecols=lambda col: col in REQUESTED_COLUMNS,
            low_memory=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Le jeu de données n'a pas pu être chargé. Placez Data-Collisions.csv "
            "dans le même dossier que ce script ou vérifiez l'accès à Internet."
        ) from exc
    return df_raw, str(source)


def prepare_source_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    date_col = "INCDTTM" if "INCDTTM" in df.columns else "INCDATE"
    df["incident_datetime"] = pd.to_datetime(df[date_col], errors="coerce")
    df["incident_date"] = df["incident_datetime"].dt.floor("D")

    for col in CONTEXT_COLS:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df.loc[df[col].eq(""), col] = "Unknown"

    if "SEVERITYCODE" not in df.columns:
        raise ValueError("La colonne SEVERITYCODE est absente du jeu de données.")

    df["severity_code"] = pd.to_numeric(df["SEVERITYCODE"], errors="coerce")
    df = df.dropna(subset=["incident_date", "LOCATION", "severity_code"]).copy()
    df = df[df["severity_code"].isin([1, 2])].copy()
    df["LOCATION"] = df["LOCATION"].astype(str).str.strip()
    df = df[df["LOCATION"].ne("")].copy()

    max_available_date = df["incident_date"].max()
    min_allowed_date = max_available_date - pd.DateOffset(years=MAX_HISTORY_YEARS)
    df = df[df["incident_date"].ge(min_allowed_date)].copy()

    log(
        f"Historique conservé : {df['incident_date'].min().date()} → "
        f"{df['incident_date'].max().date()} ({len(df):,} lignes)"
    )
    return df


def build_analytic_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    top_locations = df["LOCATION"].value_counts().head(TOP_N_LOCATIONS_MODEL).index.tolist()
    df_loc = df[df["LOCATION"].isin(top_locations)].copy()

    agg_spec = {"severity_code": ["max", "size"]}
    for col in CONTEXT_COLS:
        agg_spec[col] = mode_value

    accident_daily = df_loc.groupby(["LOCATION", "incident_date"]).agg(agg_spec)
    accident_daily.columns = [
        "max_severity" if col == ("severity_code", "max") else
        "collisions_count" if col == ("severity_code", "size") else
        col[0]
        for col in accident_daily.columns
    ]
    accident_daily = accident_daily.reset_index()
    accident_daily["target_class"] = np.where(accident_daily["max_severity"].eq(2), 2, 1)

    all_dates = pd.date_range(df["incident_date"].min(), df["incident_date"].max(), freq="D")
    grid = pd.MultiIndex.from_product(
        [top_locations, all_dates], names=["LOCATION", "incident_date"]
    ).to_frame(index=False)

    risk_df = grid.merge(
        accident_daily[["LOCATION", "incident_date", "target_class", "collisions_count"] + CONTEXT_COLS],
        on=["LOCATION", "incident_date"],
        how="left",
    )
    risk_df["target_class"] = risk_df["target_class"].fillna(0).astype(int)
    risk_df["collisions_count"] = risk_df["collisions_count"].fillna(0).astype(int)

    city_context = (
        df.groupby("incident_date", as_index=False)[CONTEXT_COLS]
        .agg(mode_value)
        .rename(columns={col: f"{col}_city" for col in CONTEXT_COLS})
    )
    risk_df = risk_df.merge(city_context, on="incident_date", how="left")

    global_context = {col: mode_value(df[col]) for col in CONTEXT_COLS}
    for col in CONTEXT_COLS:
        city_col = f"{col}_city"
        risk_df[col] = risk_df[col].fillna(risk_df[city_col]).fillna(global_context[col])
        risk_df = risk_df.drop(columns=[city_col])

    risk_df = add_temporal_features(risk_df)
    risk_df["classe"] = risk_df["target_class"].map(CLASS_LABELS)

    log(f"Jeu analytique construit : {risk_df.shape[0]:,} lignes x {risk_df.shape[1]} colonnes")
    return risk_df, top_locations, global_context


def split_train_test(risk_df: pd.DataFrame):
    dates_sorted = np.array(sorted(risk_df["incident_date"].unique()))
    cutoff_date = pd.Timestamp(dates_sorted[int(len(dates_sorted) * 0.80)])

    train_mask = risk_df["incident_date"].le(cutoff_date)
    test_mask = risk_df["incident_date"].gt(cutoff_date)

    X_train = risk_df.loc[train_mask, FEATURES]
    y_train = risk_df.loc[train_mask, "target_class"]
    X_test = risk_df.loc[test_mask, FEATURES]
    y_test = risk_df.loc[test_mask, "target_class"]

    split_info = {
        "cutoff_date": cutoff_date.date().isoformat(),
        "train_start": risk_df.loc[train_mask, "incident_date"].min().date().isoformat(),
        "train_end": risk_df.loc[train_mask, "incident_date"].max().date().isoformat(),
        "test_start": risk_df.loc[test_mask, "incident_date"].min().date().isoformat(),
        "test_end": risk_df.loc[test_mask, "incident_date"].max().date().isoformat(),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
    }
    return X_train, X_test, y_train, y_test, cutoff_date, split_info


def train_and_select_model(X_train, X_test, y_train, y_test):
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", make_encoder(sparse=True), FEATURES_CAT),
            ("num", StandardScaler(), FEATURES_NUM),
        ]
    )

    models = {
        "Régression logistique équilibrée": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Forêt aléatoire": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    fitted_models = {}

    for name, estimator in models.items():
        log(f"Entraînement du modèle : {name}")
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({
            "modele": name,
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro")),
        })
        fitted_models[name] = pipe

    results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    best_name = str(results_df.iloc[0]["modele"])
    best_model = fitted_models[best_name]
    y_pred = best_model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=[CLASS_LABELS[i] for i in [0, 1, 2]],
        zero_division=0,
        output_dict=True,
    )

    log(f"Modèle retenu : {best_name}")
    return best_model, best_name, results_df, report


def train_severity_model(risk_df: pd.DataFrame, cutoff_date: pd.Timestamp):
    accident_model_df = risk_df[risk_df["target_class"].isin([1, 2])].copy()
    sev_train_mask = accident_model_df["incident_date"].le(cutoff_date)
    sev_test_mask = accident_model_df["incident_date"].gt(cutoff_date)

    X_train = accident_model_df.loc[sev_train_mask, FEATURES]
    y_train = accident_model_df.loc[sev_train_mask, "target_class"]
    X_test = accident_model_df.loc[sev_test_mask, FEATURES]
    y_test = accident_model_df.loc[sev_test_mask, "target_class"]

    severity_preprocess = ColumnTransformer(
        transformers=[
            ("cat", make_encoder(sparse=True), FEATURES_CAT),
            ("num", StandardScaler(), FEATURES_NUM),
        ]
    )

    severity_model = Pipeline([
        ("preprocess", severity_preprocess),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    log("Entraînement du modèle conditionnel de gravité")
    severity_model.fit(X_train, y_train)
    y_pred = severity_model.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        labels=[1, 2],
        target_names=[CLASS_LABELS[i] for i in [1, 2]],
        zero_division=0,
        output_dict=True,
    )

    return severity_model, {
        "model_name": "Régression logistique conditionnelle",
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "classification_report": report,
    }


def option_values(df: pd.DataFrame, col: str) -> list[str]:
    preferred = [v for v in PREFERRED_VALUES.get(col, []) if v in set(df[col].dropna().astype(str))]
    frequent = df[col].dropna().astype(str).value_counts().head(MAX_VALUES_PER_CONTEXT).index.tolist()
    values = []
    for v in preferred + frequent:
        if v not in values:
            values.append(v)
    return values[:MAX_VALUES_PER_CONTEXT]


def scenario_key(row: dict) -> str:
    parts = [
        row["jour_semaine"],
        row["WEATHER"],
        row["ROADCOND"],
        row["LIGHTCOND"],
        row["ADDRTYPE"],
        row["JUNCTIONTYPE"],
    ]
    return "||".join(parts)


def next_date_for_day(day_name: str, reference_date: pd.Timestamp) -> pd.Timestamp:
    day_to_num = {v: k for k, v in JOUR_FR.items()}
    target = day_to_num[day_name]
    start = reference_date + pd.Timedelta(days=1)
    delta = (target - start.dayofweek) % 7
    return start + pd.Timedelta(days=int(delta))


def make_scenario_rows(locations: list[str], scenario: dict, reference_date: pd.Timestamp) -> pd.DataFrame:
    date_scenario = next_date_for_day(scenario["jour_semaine"], reference_date)
    data = pd.DataFrame({
        "LOCATION": locations,
        "incident_date": date_scenario,
    })
    for col in CONTEXT_COLS:
        data[col] = scenario[col]
    data = add_temporal_features(data)
    return data


def predict_rows(best_model, severity_model, rows: pd.DataFrame) -> pd.DataFrame:
    proba = best_model.predict_proba(rows[FEATURES])
    classes_model = list(best_model.classes_)
    output = rows.copy()

    for cls in [0, 1, 2]:
        output[f"proba_{cls}"] = proba[:, classes_model.index(cls)] if cls in classes_model else 0.0

    output["proba_accident"] = output["proba_1"] + output["proba_2"]
    output["classe_predite"] = best_model.predict(rows[FEATURES])
    output["interpretation"] = output["classe_predite"].map(CLASS_LABELS)

    sev_proba = severity_model.predict_proba(rows[FEATURES])
    sev_classes = list(severity_model.classes_)
    output["proba_blessures_si_accident"] = (
        sev_proba[:, sev_classes.index(2)] if 2 in sev_classes else 0.0
    )
    output["gravite_conditionnelle"] = severity_model.predict(rows[FEATURES])
    output["interpretation_gravite_conditionnelle"] = output["gravite_conditionnelle"].map(CLASS_LABELS)
    return output


def round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def build_dashboard_catalog(df: pd.DataFrame, top_locations: list[str], best_model, severity_model) -> dict:
    dashboard_locations = top_locations[:TOP_N_LOCATIONS_DASHBOARD]
    location_ids = {loc: str(i) for i, loc in enumerate(dashboard_locations)}

    options = {
        "jour_semaine": list(JOUR_FR.values()),
        "LOCATION": [{"id": location_ids[loc], "label": loc} for loc in dashboard_locations],
    }
    for col in CONTEXT_COLS:
        options[col] = option_values(df, col)

    reference_date = df["incident_date"].max()
    prediction_catalog = {}

    context_combinations = product(
        options["jour_semaine"],
        options["WEATHER"],
        options["ROADCOND"],
        options["LIGHTCOND"],
        options["ADDRTYPE"],
        options["JUNCTIONTYPE"],
    )

    total = 0
    for values in context_combinations:
        scenario = {
            "jour_semaine": values[0],
            "WEATHER": values[1],
            "ROADCOND": values[2],
            "LIGHTCOND": values[3],
            "ADDRTYPE": values[4],
            "JUNCTIONTYPE": values[5],
        }
        rows = make_scenario_rows(dashboard_locations, scenario, reference_date)
        pred = predict_rows(best_model, severity_model, rows)
        pred = pred.sort_values(["proba_accident", "proba_2"], ascending=False)

        locations_payload = {}
        for _, r in pred.iterrows():
            loc_id = location_ids[r["LOCATION"]]
            locations_payload[loc_id] = {
                "location": r["LOCATION"],
                "incident_date": pd.Timestamp(r["incident_date"]).date().isoformat(),
                "p0": round_float(r["proba_0"]),
                "p1": round_float(r["proba_1"]),
                "p2": round_float(r["proba_2"]),
                "p_accident": round_float(r["proba_accident"]),
                "p_blessures_si_accident": round_float(r["proba_blessures_si_accident"]),
                "classe_predite": int(r["classe_predite"]),
                "interpretation": r["interpretation"],
                "gravite_conditionnelle": int(r["gravite_conditionnelle"]),
                "interpretation_gravite_conditionnelle": r["interpretation_gravite_conditionnelle"],
            }

        top5 = [
            {
                "location_id": location_ids[r["LOCATION"]],
                "location": r["LOCATION"],
                "p_accident": round_float(r["proba_accident"]),
                "p_blessures": round_float(r["proba_2"]),
                "p_blessures_si_accident": round_float(r["proba_blessures_si_accident"]),
                "interpretation": r["interpretation"],
            }
            for _, r in pred.head(TOP_N_DISPLAY).iterrows()
        ]

        prediction_catalog[scenario_key(scenario)] = {
            "scenario": scenario,
            "top5": top5,
            "locations": locations_payload,
        }
        total += 1

    log(f"Catalogue dashboard construit : {total:,} scénarios")
    return {
        "options": options,
        "location_ids": location_ids,
        "prediction_catalog": prediction_catalog,
    }


def distribution_payload(risk_df: pd.DataFrame) -> list[dict]:
    counts = risk_df["target_class"].value_counts().sort_index()
    total = int(counts.sum())
    return [
        {
            "classe": int(cls),
            "libelle": CLASS_LABELS[int(cls)],
            "nombre": int(count),
            "proportion": round_float(count / total),
        }
        for cls, count in counts.items()
    ]


def main() -> None:
    df_raw, source_used = load_source_data()
    df = prepare_source_data(df_raw)
    risk_df, top_locations, global_context = build_analytic_dataset(df)

    X_train, X_test, y_train, y_test, cutoff_date, split_info = split_train_test(risk_df)
    best_model, best_name, results_df, report = train_and_select_model(X_train, X_test, y_train, y_test)
    severity_model, severity_summary = train_severity_model(risk_df, cutoff_date)

    dashboard = build_dashboard_catalog(df, top_locations, best_model, severity_model)

    payload = {
        "schema_version": "seattle-collision-dashboard-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": "Seattle Collision Risk — Scenario Dashboard",
            "description": "Dashboard généré par un script Python qui compare trois modèles et publie les prédictions du meilleur modèle dans un fichier JSON.",
        },
        "data": {
            "source_used": source_used,
            "max_history_years": MAX_HISTORY_YEARS,
            "top_n_locations_model": TOP_N_LOCATIONS_MODEL,
            "top_n_locations_dashboard": TOP_N_LOCATIONS_DASHBOARD,
            "rows_raw_loaded": int(df_raw.shape[0]),
            "rows_after_cleaning": int(df.shape[0]),
            "rows_analytic": int(risk_df.shape[0]),
            "date_min": df["incident_date"].min().date().isoformat(),
            "date_max": df["incident_date"].max().date().isoformat(),
            "class_distribution": distribution_payload(risk_df),
            "global_context_fallback": global_context,
        },
        "features": {
            "categorical": FEATURES_CAT,
            "numeric": FEATURES_NUM,
            "all": FEATURES,
            "target": "target_class",
            "class_labels": {str(k): v for k, v in CLASS_LABELS.items()},
        },
        "training": {
            "split": split_info,
            "models_compared": results_df.to_dict(orient="records"),
            "selection_metric": "f1_macro",
            "best_model": {
                "name": best_name,
                "metrics": results_df.iloc[0].to_dict(),
                "classification_report": report,
            },
            "severity_model": severity_summary,
            "important_note": (
                "Le dashboard ne réentraîne pas les modèles. Il lit uniquement les résultats "
                "calculés par ce script Python dans dashboard_data.json."
            ),
        },
        "dashboard": dashboard,
        "limitations": [
            "Les jours sans collision sont reconstruits à partir du calendrier.",
            "Les conditions météo des jours sans accident sont approximées par le contexte dominant.",
            "Les probabilités doivent être interprétées comme des estimations de modèle, non comme des probabilités opérationnelles certifiées.",
            "Le dashboard statique utilise un catalogue de scénarios pré-calculés par Python.",
        ],
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    size_mb = OUTPUT_JSON.stat().st_size / (1024 * 1024)
    log(f"Fichier généré : {OUTPUT_JSON.name} ({size_mb:.2f} Mo)")
    log("Terminé.")


if __name__ == "__main__":
    main()
