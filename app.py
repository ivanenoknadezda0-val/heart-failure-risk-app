from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import altair as alt

from heart_failure_config import (
    BINARY_OPTIONS,
    CHEST_PAIN_OPTIONS,
    EXERCISE_ANGINA_OPTIONS,
    FEATURE_COLUMNS,
    FEATURE_LABELS_RU,
    FORM_DEFAULTS,
    MODEL_ARTIFACT_PATH,
    RESTING_ECG_OPTIONS,
    SEX_OPTIONS,
    ST_SLOPE_OPTIONS,
    VERDICT_LABELS,
)

st.set_page_config(
    page_title="Учебный тренажер по риску сердечной недостаточности",
    page_icon="🫀",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
AUTHOR_TEXT = "Автор проекта: Иваненок Надежда, МАОУ СОШ г. Зеленоградска, г. Зеленоградск, Россия"

CUSTOM_CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1180px;}
[data-testid="stMetricValue"] {font-size: 2rem;}
.hf-card {
    border: 1px solid rgba(49, 51, 63, 0.15);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    background: linear-gradient(180deg, rgba(240,246,255,1) 0%, rgba(255,255,255,1) 100%);
    margin-bottom: 0.8rem;
}
.hf-banner-ok {
    border-left: 6px solid #2e7d32;
    background: #eef7ef;
    padding: 1rem;
    border-radius: 12px;
}
.hf-banner-risk {
    border-left: 6px solid #c62828;
    background: #fff0f0;
    padding: 1rem;
    border-radius: 12px;
}
.small-muted {color: #5f6368; font-size: 0.95rem;}
</style>
"""

FEATURE_HELP_RU_CLINICAL = {
    "Age": "Возраст пациента. С увеличением возраста частота сердечно-сосудистых нарушений и сердечной недостаточности обычно возрастает.",
    "Sex": "Пол пациента. Пол может быть связан с различиями в структуре сердечно-сосудистого риска и клинических проявлениях.",
    "ChestPainType": "Характер болевого синдрома в грудной клетке. Тип боли помогает косвенно судить о вероятности сердечной патологии.",
    "RestingBP": "Артериальное давление в покое, мм рт. ст. Повышенные или нестабильные значения могут быть связаны с перегрузкой сердечно-сосудистой системы.",
    "Cholesterol": "Общий холестерин крови, мг/дл. Нарушения липидного обмена ассоциированы с более высоким сердечно-сосудистым риском.",
    "FastingBS": "Повышение глюкозы натощак. Может указывать на нарушения углеводного обмена, которые часто сопровождают сердечно-сосудистые заболевания.",
    "RestingECG": "ЭКГ в покое. Позволяет учитывать исходные электрокардиографические изменения, связанные с работой миокарда.",
    "MaxHR": "Максимально достигнутая частота сердечных сокращений. Отражает реакцию сердечно-сосудистой системы на нагрузку.",
    "ExerciseAngina": "Наличие стенокардии при физической нагрузке. Этот признак может указывать на ишемический компонент и ухудшение переносимости нагрузки.",
    "Oldpeak": "Степень депрессии сегмента ST относительно покоя при нагрузке. Используется как маркер возможных ишемических изменений.",
    "ST_Slope": "Характер наклона сегмента ST. Изменения этого показателя учитываются при оценке вероятности сердечной патологии.",
}

REFERENCE_VALUES = {
    "Age": FORM_DEFAULTS["Age"],
    "Sex": "M",
    "ChestPainType": "ATA",
    "RestingBP": FORM_DEFAULTS["RestingBP"],
    "Cholesterol": FORM_DEFAULTS["Cholesterol"],
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": FORM_DEFAULTS["MaxHR"],
    "ExerciseAngina": "N",
    "Oldpeak": 0.0,
    "ST_Slope": "Up",
}

FEATURE_EFFECT_TEXT = {
    "Age": {
        "up": "Более старший возраст обычно связан с большей вероятностью структурных и функциональных изменений сердечно-сосудистой системы.",
        "down": "Более молодой возраст обычно ассоциируется с меньшей накопленной сердечно-сосудистой нагрузкой."
    },
    "Sex": {
        "up": "Пол может быть связан с различиями в профиле сердечно-сосудистого риска и типичных клинических сценариях.",
        "down": "В данной комбинации признаков пол пациента не усиливает итоговую модельную оценку риска."
    },
    "ChestPainType": {
        "up": "Определенные типы боли в груди чаще встречаются у пациентов с клинически значимой сердечной патологией.",
        "down": "Менее тревожный тип болевого синдрома обычно снижает настороженность модели."
    },
    "RestingBP": {
        "up": "Повышенное артериальное давление увеличивает нагрузку на миокард и сосудистую систему.",
        "down": "Более стабильные значения давления в покое обычно выглядят для модели благоприятнее."
    },
    "Cholesterol": {
        "up": "Неблагоприятный липидный профиль связан с более высоким сердечно-сосудистым риском.",
        "down": "Менее выраженные нарушения липидного обмена обычно интерпретируются как более благоприятный признак."
    },
    "FastingBS": {
        "up": "Повышенная глюкоза натощак может отражать нарушения углеводного обмена, которые часто сопровождают сердечно-сосудистые заболевания.",
        "down": "Отсутствие повышения глюкозы натощак обычно выглядит для модели как более благоприятный метаболический фон."
    },
    "RestingECG": {
        "up": "Изменения на ЭКГ в покое могут косвенно указывать на уже имеющиеся нарушения работы сердца.",
        "down": "Более спокойная картина ЭКГ в покое уменьшает настороженность модели."
    },
    "MaxHR": {
        "up": "Недостаточно благоприятная реакция ЧСС на нагрузку может ассоциироваться с худшей переносимостью нагрузки и более высоким риском.",
        "down": "Более физиологичная реакция частоты сердечных сокращений на нагрузку обычно выглядит благоприятнее."
    },
    "ExerciseAngina": {
        "up": "Стенокардия при нагрузке может указывать на ишемический компонент и клинически значимую сердечную патологию.",
        "down": "Отсутствие стенокардии при нагрузке уменьшает вероятность того, что модель отнесет пациента к группе риска."
    },
    "Oldpeak": {
        "up": "Более выраженное снижение сегмента ST при нагрузке может быть связано с ишемическими изменениями.",
        "down": "Менее выраженные изменения ST при нагрузке обычно выглядят для модели менее тревожными."
    },
    "ST_Slope": {
        "up": "Неблагоприятный характер наклона сегмента ST может сопровождать патологические изменения со стороны миокарда.",
        "down": "Более благоприятный вариант наклона сегмента ST обычно уменьшает итоговый риск по модели."
    },
}


@st.cache_resource
def load_artifact() -> dict:
    artifact_path = PROJECT_ROOT / MODEL_ARTIFACT_PATH
    if not artifact_path.exists():
        raise FileNotFoundError(
            "Не найден файл модели. Сначала запустите train_model.py, чтобы создать артефакт в папке models/."
        )
    return joblib.load(artifact_path)


def build_sidebar() -> None:
    st.sidebar.markdown("## 🫀 Учебный ML-тренажер")
    st.sidebar.write(
        "Этот интерфейс создан для студентов-медиков. Он помогает потренироваться анализировать "
        "клинические признаки пациента и смотреть, как модель связывает их с риском сердечной недостаточности."
    )
    st.sidebar.info(
        "Результат нужно воспринимать как учебную интерпретацию работы модели, а не как клинический диагноз или готовое врачебное решение."
    )
    st.sidebar.caption(AUTHOR_TEXT)


def make_input_frame(form_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([form_data])
    return df[FEATURE_COLUMNS]


def get_model_score(model, input_df: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(input_df)[0][1])
    if hasattr(model, "decision_function"):
        raw = float(model.decision_function(input_df)[0])
        return raw
    return float(model.predict(input_df)[0])


def predict_case(model, input_df: pd.DataFrame) -> tuple[int, float | None]:
    pred = int(model.predict(input_df)[0])
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(input_df)[0][1])
    return pred, prob


def explain_verdict(pred: int) -> str:
    if pred == 1:
        return (
            "Модель отнесла этот учебный клинический случай к группе риска. "
            "Это означает, что сочетание введенных признаков похоже на те случаи, которые модель чаще связывала "
            "с наличием риска сердечной недостаточности."
        )
    return (
        "Модель не отнесла этот учебный клинический случай к группе риска. "
        "Это означает, что текущая комбинация признаков выглядит для модели менее характерной для риск-группы."
    )


def build_feature_influence_table(model, input_df: pd.DataFrame) -> pd.DataFrame:
    base_score = get_model_score(model, input_df)
    rows = []

    for feature in FEATURE_COLUMNS:
        modified = input_df.copy()
        modified.at[0, feature] = REFERENCE_VALUES.get(feature, FORM_DEFAULTS.get(feature, input_df.at[0, feature]))
        new_score = get_model_score(model, modified)
        delta = base_score - new_score

        if delta > 0:
            direction = "повышает риск"
            effect_key = "up"
        elif delta < 0:
            direction = "снижает риск"
            effect_key = "down"
        else:
            direction = "влияние минимально"
            effect_key = "down"

        rows.append(
            {
                "feature_code": feature,
                "Признак": FEATURE_LABELS_RU[feature],
                "Текущее значение": input_df.at[0, feature],
                "Влияние_raw": float(delta),
                "Влияние": round(abs(delta), 4),
                "Направление": direction,
                "Медицинская интерпретация": FEATURE_EFFECT_TEXT[feature][effect_key],
            }
        )

    result = pd.DataFrame(rows)
    result["signed_effect"] = result["Влияние_raw"]
    result["abs_effect"] = result["Влияние_raw"].abs()
    result = result.sort_values("abs_effect", ascending=False).reset_index(drop=True)
    return result


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    build_sidebar()

    try:
        artifact = load_artifact()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    pipeline = artifact["model"]
    metadata = artifact.get("metadata", {})

    hero_left, hero_right = st.columns([1.4, 1])
    with hero_left:
        st.title("Учебный тренажер: риск сердечной недостаточности")
        st.write(
            "Заполните учебную карточку пациента и посмотрите, как модель классифицирует случай: "
            "относит ли она его к группе риска сердечной недостаточности или нет. "
            "Дополнительно интерфейс показывает, какие признаки сильнее всего повлияли на вывод."
        )
    with hero_right:
        st.markdown(
            f"""
            <div class="hf-card">
                <b>Модель для разбора случая</b><br>
                <span class="small-muted">{metadata.get('selected_model', 'не указана')}</span><br><br>
                <b>Количество признаков</b><br>
                <span class="small-muted">{len(FEATURE_COLUMNS)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tab_form, tab_reference = st.tabs(["Учебный разбор случая", "Справка по признакам"])

    with tab_reference:
        st.subheader("Пояснения к признакам")
        st.write(
            "Этот раздел помогает понять, что означает каждый признак и почему он вообще может быть полезен "
            "для оценки риска сердечной недостаточности."
        )
        ref_rows = [
            {"Поле": FEATURE_LABELS_RU[col], "Описание": FEATURE_HELP_RU_CLINICAL[col]}
            for col in FEATURE_COLUMNS
        ]
        st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)

    with tab_form:
        with st.form("heart_failure_form"):
            st.subheader("Введите данные учебного клинического случая")
            c1, c2, c3 = st.columns(3)

            with c1:
                age = st.number_input(
                    "Возраст",
                    min_value=18,
                    max_value=100,
                    value=FORM_DEFAULTS["Age"],
                    step=1,
                    help=FEATURE_HELP_RU_CLINICAL["Age"],
                )
                sex_label = st.selectbox(
                    "Пол",
                    list(SEX_OPTIONS.keys()),
                    index=list(SEX_OPTIONS.keys()).index(FORM_DEFAULTS["Sex"]),
                    help=FEATURE_HELP_RU_CLINICAL["Sex"],
                )
                chest_pain_label = st.selectbox(
                    "Тип боли в груди",
                    list(CHEST_PAIN_OPTIONS.keys()),
                    index=list(CHEST_PAIN_OPTIONS.keys()).index(FORM_DEFAULTS["ChestPainType"]),
                    help=FEATURE_HELP_RU_CLINICAL["ChestPainType"],
                )
                resting_bp = st.number_input(
                    "Артериальное давление в покое, мм рт. ст.",
                    min_value=60,
                    max_value=260,
                    value=FORM_DEFAULTS["RestingBP"],
                    step=1,
                    help=FEATURE_HELP_RU_CLINICAL["RestingBP"],
                )

            with c2:
                cholesterol = st.number_input(
                    "Общий холестерин, мг/дл",
                    min_value=0,
                    max_value=700,
                    value=FORM_DEFAULTS["Cholesterol"],
                    step=1,
                    help=FEATURE_HELP_RU_CLINICAL["Cholesterol"],
                )
                fasting_bs_label = st.radio(
                    "Глюкоза натощак > 120 мг/дл",
                    list(BINARY_OPTIONS.keys()),
                    horizontal=True,
                    index=list(BINARY_OPTIONS.keys()).index(FORM_DEFAULTS["FastingBS"]),
                    help=FEATURE_HELP_RU_CLINICAL["FastingBS"],
                )
                resting_ecg_label = st.selectbox(
                    "ЭКГ в покое",
                    list(RESTING_ECG_OPTIONS.keys()),
                    index=list(RESTING_ECG_OPTIONS.keys()).index(FORM_DEFAULTS["RestingECG"]),
                    help=FEATURE_HELP_RU_CLINICAL["RestingECG"],
                )
                max_hr = st.number_input(
                    "Максимальная ЧСС",
                    min_value=50,
                    max_value=250,
                    value=FORM_DEFAULTS["MaxHR"],
                    step=1,
                    help=FEATURE_HELP_RU_CLINICAL["MaxHR"],
                )

            with c3:
                exercise_angina_label = st.radio(
                    "Стенокардия при нагрузке",
                    list(EXERCISE_ANGINA_OPTIONS.keys()),
                    horizontal=True,
                    index=list(EXERCISE_ANGINA_OPTIONS.keys()).index(FORM_DEFAULTS["ExerciseAngina"]),
                    help=FEATURE_HELP_RU_CLINICAL["ExerciseAngina"],
                )
                oldpeak = st.number_input(
                    "Снижение сегмента ST при нагрузке (Oldpeak)",
                    min_value=-5.0,
                    max_value=10.0,
                    value=float(FORM_DEFAULTS["Oldpeak"]),
                    step=0.1,
                    help=FEATURE_HELP_RU_CLINICAL["Oldpeak"],
                )
                st_slope_label = st.selectbox(
                    "Наклон сегмента ST",
                    list(ST_SLOPE_OPTIONS.keys()),
                    index=list(ST_SLOPE_OPTIONS.keys()).index(FORM_DEFAULTS["ST_Slope"]),
                    help=FEATURE_HELP_RU_CLINICAL["ST_Slope"],
                )

            submitted = st.form_submit_button("Показать разбор случая", use_container_width=True)

    if not submitted:
        return

    form_data = {
        "Age": age,
        "Sex": SEX_OPTIONS[sex_label],
        "ChestPainType": CHEST_PAIN_OPTIONS[chest_pain_label],
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": BINARY_OPTIONS[fasting_bs_label],
        "RestingECG": RESTING_ECG_OPTIONS[resting_ecg_label],
        "MaxHR": max_hr,
        "ExerciseAngina": EXERCISE_ANGINA_OPTIONS[exercise_angina_label],
        "Oldpeak": oldpeak,
        "ST_Slope": ST_SLOPE_OPTIONS[st_slope_label],
    }

    input_df = make_input_frame(form_data)
    pred, prob = predict_case(pipeline, input_df)

    st.subheader("Вывод модели по учебному случаю")
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Классификация модели", "Риск" if pred == 1 else "Нет риска")
    metric_2.metric("Возраст", f"{age}")
    metric_3.metric("Макс. ЧСС", f"{max_hr}")

    if pred == 1:
        st.markdown(
            f'<div class="hf-banner-risk"><b>{VERDICT_LABELS[pred]}</b><br><br>{explain_verdict(pred)}</div>',
            unsafe_allow_html=True,
        )
        st.info(
            "Учебная подсказка: попробуйте изменить 1–2 признака и посмотреть, как изменится вывод модели. "
            "Так удобнее понять, какие параметры оказывают наибольшее влияние."
        )
    else:
        st.markdown(
            f'<div class="hf-banner-ok"><b>{VERDICT_LABELS[pred]}</b><br><br>{explain_verdict(pred)}</div>',
            unsafe_allow_html=True,
        )
        st.info(
            "Учебная подсказка: попробуйте смоделировать менее благоприятные значения и посмотреть, "
            "в какой момент модель начнет относить случай к группе риска."
        )

    if prob is not None:
        st.progress(int(round(prob * 100)))
        st.caption(f"Оценка вероятности класса риска по модели: {prob * 100:.1f}%")

    influence_df = build_feature_influence_table(pipeline, input_df)
    top_influence = influence_df.head(7).copy()

    st.subheader("Какие признаки сильнее всего повлияли на вывод")
    st.write(
        "Этот график помогает в учебных целях увидеть, какие признаки сильнее сместили модель "
        "в сторону риска или, наоборот, в сторону более благоприятного вывода."
    )

    chart_df = top_influence.copy()
    chart_df["Цвет"] = chart_df["signed_effect"].apply(
        lambda x: "Повышает риск" if x > 0 else "Снижает риск"
    )

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X(
                "signed_effect:Q",
                title="Влияние на итоговый вывод модели",
                axis=alt.Axis(format=".3f"),
            ),
            y=alt.Y(
                "Признак:N",
                sort="-x",
                title=None,
            ),
            color=alt.Color(
                "Цвет:N",
                scale=alt.Scale(
                    domain=["Повышает риск", "Снижает риск"],
                    range=["#c62828", "#2e7d32"],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("Признак:N"),
                alt.Tooltip("Текущее значение:N"),
                alt.Tooltip("Направление:N"),
                alt.Tooltip("Влияние:Q", format=".4f"),
                alt.Tooltip("Медицинская интерпретация:N"),
            ],
        )
        .properties(height=320)
    )

    st.altair_chart(chart, use_container_width=True)

    st.markdown("### Почему модель обратила внимание именно на эти признаки")
    for _, row in top_influence.iterrows():
        icon = "🔴" if row["signed_effect"] > 0 else "🟢"
        st.markdown(
            f"""
            <div class="hf-card">
                <b>{icon} {row['Признак']}</b><br>
                <span class="small-muted">Текущее значение: {row['Текущее значение']} • {row['Направление']}</span><br><br>
                {row['Медицинская интерпретация']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Посмотреть введенные значения"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)

    st.caption(
        f"Модель: {metadata.get('selected_model', 'не указана')} | "
        f"Параметры: {metadata.get('selected_params', {})}"
    )
    st.caption(
        "Техническая справка: страница использует финальный pipeline из обучающего скрипта. "
        "Блок влияния признаков показывает приближенную локальную интерпретацию, полезную для учебного разбора случая."
    )
    st.caption(AUTHOR_TEXT)


if __name__ == "__main__":
    main()