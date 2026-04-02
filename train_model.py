from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from heart_failure_config import (
    CATEGORY_FEATURES,
    DATA_URL,
    FEATURE_COLUMNS,
    MODEL_ARTIFACT_PATH,
    NUMERIC_FEATURES,
    PASSTHROUGH_BINARY_FEATURES,
    TARGET_COLUMN,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def build_preprocessor() -> ColumnTransformer:
    categorical_features = list(dict.fromkeys(CATEGORY_FEATURES + PASSTHROUGH_BINARY_FEATURES))

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )


clf_models = {
    "Gaussian NB": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "Ridge Classifier": RidgeClassifier(),
    "Linear SVC": LinearSVC(max_iter=5000),
    "MLP Classifier": MLPClassifier(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

clf_param_grid = {
    "Gaussian NB": {
        "model__var_smoothing": np.logspace(-12, -6, 20)
    },
    "QDA": {
        "model__reg_param": np.linspace(0.0, 1.0, 20)
    },
    "Ridge Classifier": {
        "model__alpha": np.linspace(0.01, 10, 20)
    },
    "Linear SVC": {
        "model__C": np.linspace(0.01, 10, 20)
    },
    "MLP Classifier": {
        "model__hidden_layer_sizes": [(n,) for n in range(5, 15, 5)],
        "model__alpha": np.logspace(-3, -1, 10)
    },
    "Decision Tree": {
        "model__max_depth": np.arange(2, 15),
        "model__min_samples_split": np.arange(2, 10)
    },
    "KNN": {
        "model__n_neighbors": np.arange(1, 20),
        "model__weights": ["uniform", "distance"]
    },
    "Random Forest": {
        "model__n_estimators": [50, 100],
        "model__max_depth": [None, 3, 5]
    },
    "Gradient Boosting": {
        "model__n_estimators": [50, 100],
        "model__learning_rate": [0.01, 0.1],
        "model__max_depth": [2, 3]
    },
}


def main() -> None:
    print("Загружаю датасет...")
    data = pd.read_csv(DATA_URL)
    x = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    preprocessor = build_preprocessor()
    results = []

    for name, model in clf_models.items():
        print(f"Обучение модели: {name}")
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        grid = GridSearchCV(
            pipe,
            clf_param_grid[name],
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid.fit(x_train, y_train)

        cv_scores = cross_validate(
            grid.best_estimator_,
            x_train,
            y_train,
            cv=5,
            scoring="accuracy",
            return_train_score=False,
        )

        results.append(
            {
                "Model": name,
                "Best Params": grid.best_params_,
                "Mean Accuracy": cv_scores["test_score"].mean(),
                "Std Accuracy": cv_scores["test_score"].std(),
            }
        )

    results_df = pd.DataFrame(results).sort_values("Mean Accuracy", ascending=False).reset_index(drop=True)
    print("\nИтоги CV:")
    print(results_df.to_string(index=False))

    # Нам нужно воспроизвести именно логику из последних ячеек ноутбука.
    # Там используется .iloc[1], то есть берётся ВТОРАЯ модель после сортировки по accuracy.
    selected_row = results_df.iloc[1]
    selected_name = selected_row["Model"]
    selected_params = selected_row["Best Params"]

    final_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", clf_models[selected_name]),
    ])
    final_pipe.set_params(**selected_params)
    final_pipe.fit(x_train, y_train)

    artifact = {
        "model": final_pipe,
        "metadata": {
            "selected_model": selected_name,
            "selected_params": selected_params,
            "feature_columns": FEATURE_COLUMNS,
            "dataset_url": DATA_URL,
            "note": "Модель выбрана по логике последних ячеек ноутбука: сортировка по Mean Accuracy и затем .iloc[1].",
        },
    }

    output_path = PROJECT_ROOT / MODEL_ARTIFACT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    print(f"\nГотово. Артефакт сохранён в: {output_path}")
    print(f"Финальная модель по ноутбуку: {selected_name}")
    print(f"Параметры: {selected_params}")


if __name__ == "__main__":
    main()
