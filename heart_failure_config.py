from __future__ import annotations

DATA_URL = "https://drive.google.com/uc?id=1hqnXghZIWWMyVfIv13mzor4n-ud1ob8-"
MODEL_ARTIFACT_PATH = "models/heart_failure_final_pipeline.joblib"

TARGET_COLUMN = "HeartDisease"

FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]

NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CATEGORY_FEATURES = ["ChestPainType", "RestingECG", "ST_Slope"]
PASSTHROUGH_BINARY_FEATURES = ["Sex", "FastingBS", "ExerciseAngina"]

FEATURE_LABELS_RU = {
    "Age": "Возраст",
    "Sex": "Пол",
    "ChestPainType": "Тип боли в груди",
    "RestingBP": "Артериальное давление в покое",
    "Cholesterol": "Холестерин",
    "FastingBS": "Глюкоза натощак > 120 мг/дл",
    "RestingECG": "ЭКГ в покое",
    "MaxHR": "Максимальная ЧСС",
    "ExerciseAngina": "Стенокардия при нагрузке",
    "Oldpeak": "Oldpeak",
    "ST_Slope": "Наклон сегмента ST",
}

FEATURE_HELP_RU = {
    "Age": "Возраст пациента в полных годах.",
    "Sex": "Пол пациента. В исходном датасете кодируется как M/F.",
    "ChestPainType": "Клинический тип боли в груди: TA — типичная стенокардия, ATA — атипичная, NAP — неангинозная боль, ASY — бессимптомное течение.",
    "RestingBP": "Систолическое артериальное давление в покое, мм рт. ст.",
    "Cholesterol": "Общий холестерин, мг/дл. В датасете встречаются и нулевые значения как особенность исходных данных.",
    "FastingBS": "Признак повышенной глюкозы натощак: 1 — больше 120 мг/дл, 0 — нет.",
    "RestingECG": "Результат ЭКГ в покое: Normal, ST, LVH.",
    "MaxHR": "Максимальная достигнутая частота сердечных сокращений.",
    "ExerciseAngina": "Наличие стенокардии при физической нагрузке: Y/N.",
    "Oldpeak": "Снижение сегмента ST относительно покоя при нагрузке; один из важных признаков ишемии.",
    "ST_Slope": "Наклон сегмента ST при нагрузке: Up, Flat, Down.",
}

SEX_OPTIONS = {"Женский": "F", "Мужской": "M"}
CHEST_PAIN_OPTIONS = {
    "TA — типичная стенокардия": "TA",
    "ATA — атипичная стенокардия": "ATA",
    "NAP — неангинозная боль": "NAP",
    "ASY — бессимптомно": "ASY",
}
RESTING_ECG_OPTIONS = {
    "Normal — норма": "Normal",
    "ST — изменения ST-T": "ST",
    "LVH — гипертрофия ЛЖ": "LVH",
}
EXERCISE_ANGINA_OPTIONS = {"Нет": "N", "Да": "Y"}
ST_SLOPE_OPTIONS = {
    "Up — восходящий": "Up",
    "Flat — горизонтальный": "Flat",
    "Down — нисходящий": "Down",
}
BINARY_OPTIONS = {"Нет": 0, "Да": 1}

FORM_DEFAULTS = {
    "Age": 55,
    "Sex": "Мужской",
    "ChestPainType": "ASY — бессимптомно",
    "RestingBP": 130,
    "Cholesterol": 220,
    "FastingBS": "Нет",
    "RestingECG": "Normal — норма",
    "MaxHR": 140,
    "ExerciseAngina": "Нет",
    "Oldpeak": 1.0,
    "ST_Slope": "Flat — горизонтальный",
}

VERDICT_LABELS = {
    0: "Сейчас модель не относит пациента к группе риска сердечной недостаточности",
    1: "Сейчас модель относит пациента к группе риска сердечной недостаточности",
}
