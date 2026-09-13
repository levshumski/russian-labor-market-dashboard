from pathlib import Path
import base64
import hashlib
import json
import re

import numpy as np
import pandas as pd
from pypdf import PdfReader
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd().resolve()
)

REGIONAL_BASE_FILE = PROJECT_DIR / "rosstat_regions_for_map.csv"
PROFESSIONAL_DEMAND_FILE = PROJECT_DIR / "Rosstat_professional_demand_2024.xlsx"
GEOJSON_FILE = PROJECT_DIR / "russia_adm1.geojson"
MAP_FILE = PROJECT_DIR / "interactive_regional_deficit_map.html"
PROFESSIONAL_REPORT_FILE = (
    PROJECT_DIR / "rosstat_professional_job_search_difficulty_by_region.csv"
)
REGION_REFERENCE_FILE = PROJECT_DIR / "region_reference_data.csv"
POPULATION_FILE = PROJECT_DIR / "OkPopul_Comp2025_Site.xlsx"
UNEMPLOYMENT_FILE = PROJECT_DIR / "Unemployment(15-72yo).xlsx"
LABOR_FORCE_FILE = PROJECT_DIR / "Chislennost_rabochey_sili_15-72yo.xlsx"
EMPLOYMENT_FILE = PROJECT_DIR / "Chislennost_rabochey_sili_i_yroven_zanatosti_15-72yo.xlsx"
CURRENT_LABOR_FORCE_FILE = (
    PROJECT_DIR / "official_sources" / "Rosstat_labor_force_15plus_2026-07-15.xlsx"
)
CURRENT_UNEMPLOYMENT_FILE = (
    PROJECT_DIR / "official_sources" / "Rosstat_unemployment_15plus_2026-07-15.xlsx"
)
WAGE_VALIDATION_FILE = (
    PROJECT_DIR / "official_sources" / "Rosstat_monthly_wages_by_region_2013_2026-06.xlsx"
)
SUBSISTENCE_MINIMUM_FILE = (
    PROJECT_DIR / "official_sources" / "Rosstat_subsistence_minimum_by_region.xlsx"
)
PROFESSIONAL_WAGE_FILE = (
    PROJECT_DIR / "official_sources" / "Rosstat_professional_wage_2025_6.xlsx"
)
DETAILED_WAGE_ARCHIVE_FILE = (
    PROJECT_DIR / "official_sources" / "Rosstat_57T_detailed_wages_2025.rar"
)
DETAILED_WAGE_BULLETIN_FILE = (
    PROJECT_DIR / "official_sources" / "Rosstat_57T_2025" / "Статбюлетень 2025.xlsx"
)
OCCUPATION_SALARY_SNAPSHOT_FILE = (
    PROJECT_DIR / "trudvsem_top_occupation_salary_snapshot_2026.csv"
)
MEDIAN_WAGE_HISTORY_FILE = (
    PROJECT_DIR / "official_sources" / "Rosstat_median_wage_2019_2025.xlsx"
)
REGIONAL_REPORT_FILE = PROJECT_DIR / "rosstat_regional_job_search_difficulty.csv"
REGIONAL_LABOR_MARKET_FILE = PROJECT_DIR / "rosstat_regional_labor_market_2024.csv"
INDEX_AUDIT_FILE = PROJECT_DIR / "index_methodology_audit.csv"
GROUP_FORECAST_FILE = PROJECT_DIR / "professional_group_forecast_2027_2032.csv"
OCCUPATION_FORECAST_FILE = PROJECT_DIR / "occupation_forecast_2027_2032.csv"
CURRENT_LABOR_CONTEXT_FILE = PROJECT_DIR / "rosstat_current_labor_context_2026.csv"
WAGE_INFLATION_FILE = PROJECT_DIR / "cbr_macro_survey_wages_inflation_2021_2029.csv"
ROSSTAT_CPI_HISTORY_FILE = (
    PROJECT_DIR / "official_sources" / "Rosstat_CPI_1992_2025.xlsx"
)
ROSSTAT_CPI_MONTHLY_FILE = (
    PROJECT_DIR / "official_sources" / "Rosstat_CPI_monthly_1991_2026-08.xlsx"
)
SEVASTOPOL_LABOR_2024_FILE = (
    PROJECT_DIR / "official_sources" / "Sevastopol_labor_force_2024.xlsx"
)
SEVASTOPOL_LABOR_2026_FILE = (
    PROJECT_DIR / "official_sources" / "Sevastopol_labor_force_2026.xlsx"
)
SEVASTOPOL_YEARBOOK_FILE = (
    PROJECT_DIR / "official_sources" / "Sevastopol_in_figures_2025.pdf"
)
ROSSTAT_LABOUR_FORCE_2026_FILE = (
    PROJECT_DIR / "official_sources" / "Rosstat_Labour_Force_2026.pdf"
)
ROSSTAT_CPI_FILE = PROJECT_DIR / "official_sources" / "Rosstat_CPI_December_2025.html"
CBR_INFLATION_FORECAST_FILE = (
    PROJECT_DIR / "official_sources" / "Bank_of_Russia_inflation_forecast_2026-07-24.html"
)
CBR_MEDIUM_TERM_FORECAST_FILE = (
    PROJECT_DIR / "official_sources" / "Bank_of_Russia_medium_term_forecast_2026-07-24.pdf"
)
CBR_MACRO_SURVEY_FILE = (
    PROJECT_DIR / "official_sources" / "Bank_of_Russia_macroeconomic_survey_2026-07.xlsx"
)
CBR_MACRO_SURVEY_PDF_FILE = (
    PROJECT_DIR / "official_sources" / "Bank_of_Russia_macroeconomic_survey_2026-07.pdf"
)
P4_METHOD_FILE = (
    PROJECT_DIR / "official_sources" / "Rosstat_P4NZ_methodology_order_519_2025.pdf"
)
MSP_METHOD_FILE = (
    PROJECT_DIR / "official_sources" / "Rosstat_MSP_employment_methodology_2026.pdf"
)
REGION_INDICATORS_FILE = (
    PROJECT_DIR / "official_sources" / "Region_Pokaz_2025.pdf"
)
SOURCE_MANIFEST_FILE = PROJECT_DIR / "official_sources" / "source_manifest.csv"
INDEX_PERIOD_PATTERN = r"октябрь.*декабрь.*2024"
CURRENT_LABOR_PERIOD_PATTERN = r"март.*май.*2026"
INDEX_PERIOD_LABEL = "октябрь–декабрь 2024"
PROFESSIONAL_REFERENCE_DATE = "31.10.2024"
LABOR_MARKET_REFERENCE_PERIOD = "2024 год"
FORECAST_YEARS = tuple(range(2027, 2033))

# Сценарные коэффициенты к числу требуемых работников 2024 года. Это не
# «предсказанные вакансии», а прозрачная стресс-корректировка последнего среза
# 1-Т(проф): краткосрочная слабость строительства и экономики сочетается с
# устойчивой кадровой потребностью в квалифицированном труде. Коэффициенты
# намеренно умеренные, поскольку официального временного ряда по профессиям нет.
PROFESSIONAL_DEMAND_SCENARIO = {
    "Промышленность, строительство и транспорт": {
        2027: 0.96, 2028: 0.94, 2029: 0.96, 2030: 0.99, 2031: 1.02, 2032: 1.05,
    },
    "Операторы машин, сборщики и водители": {
        2027: 1.02, 2028: 1.04, 2029: 1.06, 2030: 1.08, 2031: 1.10, 2032: 1.12,
    },
    "Сельское и лесное хозяйство": {
        2027: 1.00, 2028: 1.00, 2029: 1.01, 2030: 1.01, 2031: 1.02, 2032: 1.02,
    },
    "Руководители": {
        2027: 0.99, 2028: 0.98, 2029: 0.98, 2030: 0.97, 2031: 0.97, 2032: 0.96,
    },
    "Специалисты высшей квалификации": {
        2027: 1.03, 2028: 1.06, 2029: 1.09, 2030: 1.12, 2031: 1.14, 2032: 1.16,
    },
    "Неквалифицированные рабочие": {
        2027: 0.99, 2028: 0.98, 2029: 0.97, 2030: 0.96, 2031: 0.95, 2032: 0.94,
    },
    "Офисные и учётные служащие": {
        2027: 0.97, 2028: 0.94, 2029: 0.91, 2030: 0.89, 2031: 0.87, 2032: 0.85,
    },
    "Специалисты средней квалификации": {
        2027: 1.03, 2028: 1.06, 2029: 1.09, 2030: 1.12, 2031: 1.14, 2032: 1.16,
    },
    "Обслуживание, торговля и охрана": {
        2027: 1.00, 2028: 1.01, 2029: 1.02, 2030: 1.03, 2031: 1.04, 2032: 1.05,
    },
}

PROFESSIONAL_SCENARIO_NOTE = {
    "Промышленность, строительство и транспорт": (
        "в 2027–2029 годах применён циклический дисконт из-за слабого роста ВВП, "
        "дорогого кредита и снижения строительства; затем — умеренная стабилизация"
    ),
    "Операторы машин, сборщики и водители": (
        "умеренный рост: официальный кадровый прогноз указывает на высокий вес "
        "профессий со средним профессиональным образованием"
    ),
    "Сельское и лесное хозяйство": (
        "почти неизменный спрос: малая численность не превращается автоматически "
        "в быстрый рост всей отрасли"
    ),
    "Руководители": "небольшое снижение доли при слабом экономическом росте",
    "Специалисты высшей квалификации": (
        "умеренный рост за счёт здравоохранения, инженерных и других сложных функций"
    ),
    "Неквалифицированные рабочие": (
        "постепенное снижение доли под влиянием автоматизации, без сценария обвала"
    ),
    "Офисные и учётные служащие": (
        "наиболее заметное снижение из-за цифровизации стандартных операций"
    ),
    "Специалисты средней квалификации": (
        "умеренный рост в соответствии с официальным приоритетом кадров СПО"
    ),
    "Обслуживание, торговля и охрана": "почти стабильный спрос с небольшим ростом",
}


def normalize_region(value):
    """Приводит варианты названий субъектов из разных таблиц Росстата к одному ключу."""
    name = re.sub(r"\s+", " ", str(value)).strip()
    aliases = {
        "г.Москва": "Москва",
        "г. Москва": "Москва",
        "г.Санкт-Петербург": "Санкт-Петербург",
        "г. Санкт-Петербург": "Санкт-Петербург",
        "г.Севастополь": "Севастополь",
        "г. Севастополь": "Севастополь",
        "Республика Северная Осетия - Алания": "Республика Северная Осетия-Алания",
        "Республика Северная Осетия – Алания": "Республика Северная Осетия-Алания",
        "Республика Северная Осетия- Алания": "Республика Северная Осетия-Алания",
        "Еврейская авт.область": "Еврейская автономная область",
        "Чукотский авт.округ": "Чукотский автономный округ",
        "Ямало-Ненецкий авт.округ": "Ямало-Ненецкий автономный округ",
        "Ханты-Мансийский авт.округ-Югра": "Ханты-Мансийский автономный округ — Югра",
        "Ханты-Мансийский авт.округ": "Ханты-Мансийский автономный округ — Югра",
        "Ханты-Мансийский автономный округ-Югра": "Ханты-Мансийский автономный округ — Югра",
        "Ханты-Мансийский автономный округ – Югра": "Ханты-Мансийский автономный округ — Югра",
        "в том числе: Ханты-Мансийский авт.округ": "Ханты-Мансийский автономный округ — Югра",
        "Ненецкий авт.округ": "Ненецкий автономный округ",
        "Hенецкий авт.округ": "Ненецкий автономный округ",
        "в том числе: Ненецкий авт.округ": "Ненецкий автономный округ",
        "в том числе Ненецкий автономный округ": "Ненецкий автономный округ",
        "Ямало-Hенецкий авт.округ": "Ямало-Ненецкий автономный округ",
        "Hижегородская область": "Нижегородская область",
        "Кемеровская область-Кузбасс": "Кемеровская область",
        "Архангельская область без авт. округа": "Архангельская область",
        "Архангельская область без автономного округа": "Архангельская область",
        "Тюменская область без авт. округов": "Тюменская область",
        "Тюменская область без автономных округов": "Тюменская область",
    }
    return aliases.get(name, name)


def percentile_signal(series, higher_means_tightness=True):
    """Переводит показатель в относительную шкалу 0–1 среди сравниваемых строк."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid_count = int(numeric.notna().sum())
    if valid_count <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    ranks = numeric.rank(method="average", ascending=higher_means_tightness)
    return ((ranks - 1) / (valid_count - 1)).fillna(0.5)


def robust_unit_signal(series, log_transform=False, lower_quantile=0.05, upper_quantile=0.95):
    """Нормирует показатель в 0–1, не удаляя выбросы и не давая им растянуть шкалу."""
    numeric = pd.to_numeric(series, errors="coerce").clip(lower=0)
    transformed = np.log1p(numeric) if log_transform else numeric
    valid = transformed.dropna()
    if valid.empty:
        return pd.Series(0.5, index=series.index, dtype=float)
    lower = float(valid.quantile(lower_quantile))
    upper = float(valid.quantile(upper_quantile))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return pd.Series(0.5, index=series.index, dtype=float)
    return ((transformed - lower) / (upper - lower)).clip(0, 1).fillna(0.5)


def robust_logit_signal(series, lower_quantile=0.05, upper_quantile=0.95):
    """Устойчивая шкала для долей: логит сохраняет различия, квантили сдерживают выбросы."""
    numeric = pd.to_numeric(series, errors="coerce").clip(1e-6, 1 - 1e-6)
    transformed = np.log(numeric / (1 - numeric))
    valid = transformed.dropna()
    if valid.empty:
        return pd.Series(0.5, index=series.index, dtype=float)
    lower = float(valid.quantile(lower_quantile))
    upper = float(valid.quantile(upper_quantile))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return pd.Series(0.5, index=series.index, dtype=float)
    return ((transformed - lower) / (upper - lower)).clip(0, 1).fillna(0.5)


def read_indicator_period(file_path, sheet_name, value_name, period_pattern):
    """Читает один явно заданный период, не подменяя его последним доступным."""
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    normalized_periods = raw.iloc[4].map(
        lambda value: re.sub(r"\s+", " ", str(value)).strip()
    )
    matches = [
        int(column)
        for column, period in normalized_periods.items()
        if re.search(period_pattern, period, flags=re.IGNORECASE)
    ]
    if not matches:
        raise ValueError(
            f"Не найден согласованный период {period_pattern!r}: "
            f"{Path(file_path).name}, лист {sheet_name}"
        )
    value_column = matches[-1]
    result = raw.iloc[5:, [0, value_column]].copy()
    result.columns = ["subject", value_name]
    result["subject"] = result["subject"].map(normalize_region)
    result[value_name] = pd.to_numeric(result[value_name], errors="coerce")
    result = result.dropna(subset=[value_name]).drop_duplicates("subject", keep="last")
    return result, normalized_periods.iloc[value_column]


def read_sevastopol_labor(file_path, row_from_end=1):
    """Читает числовую строку территориальной таблицы Крымстата/Росстата."""
    raw = pd.read_excel(file_path, header=None)
    numeric = raw.iloc[:, 1:7].apply(pd.to_numeric, errors="coerce")
    rows = numeric.loc[numeric.iloc[:, :3].notna().all(axis=1)]
    row = rows.iloc[-row_from_end]
    return {
        "labor_force": float(row.iloc[0]),
        "employed": float(row.iloc[1]),
        "unemployed": float(row.iloc[2]),
        "participation": float(row.iloc[3]),
        "employment": float(row.iloc[4]),
        "unemployment": float(row.iloc[5]),
    }


def estimate_beta_prior(frame, group_column, vacancies_column, jobs_column):
    """Оценивает Beta-приор методом моментов отдельно для каждой группы.

    Межрегиональная дисперсия очищается от приближённой биномиальной
    дисперсии наблюдения. Возвращаемая kappa — сила сглаживания, оцененная
    из данных, а не выбранная вручную.
    """
    records = []
    for group_name, group in frame.groupby(group_column, dropna=False):
        vacancies = pd.to_numeric(group[vacancies_column], errors="coerce")
        jobs = pd.to_numeric(group[jobs_column], errors="coerce")
        valid = vacancies.notna() & jobs.notna() & (jobs > 0)
        vacancies = vacancies.loc[valid].clip(lower=0)
        jobs = jobs.loc[valid].clip(lower=1)
        rates = (vacancies / jobs).clip(1e-6, 1 - 1e-6)
        if rates.empty:
            mean_rate, between_variance, kappa = 0.5, 0.0, 1_000_000.0
        else:
            mean_rate = float(vacancies.sum() / jobs.sum())
            observed_variance = float(rates.var(ddof=1)) if len(rates) > 1 else 0.0
            sampling_variance = float(
                (mean_rate * (1 - mean_rate) / jobs).mean()
            )
            between_variance = max(observed_variance - sampling_variance, 0.0)
            if between_variance <= 1e-12:
                kappa = 1_000_000.0
            else:
                kappa = mean_rate * (1 - mean_rate) / between_variance - 1
                kappa = float(np.clip(kappa, 2.0, 1_000_000.0))
        records.append({
            group_column: group_name,
            "prior_mean_rate": mean_rate,
            "between_region_variance": between_variance,
            "kappa_mom": kappa,
        })
    return pd.DataFrame(records).set_index(group_column)


def read_wage_validation():
    """Читает официальную зарплату полного круга организаций для внешней проверки."""
    raw = pd.read_excel(WAGE_VALIDATION_FILE, sheet_name="с 2019", header=None)
    year_headers = raw.iloc[1, 1:].ffill().map(
        lambda value: int(re.search(r"20\d{2}", str(value)).group())
    )
    month_headers = raw.iloc[2, 1:].map(
        lambda value: re.sub(r"\s+", " ", str(value)).strip().lower()
    )

    def year_columns(year):
        return [index + 1 for index, value in enumerate(year_headers) if value == year]

    def month_column(year, month):
        matches = [
            index + 1
            for index, (header_year, header_month) in enumerate(
                zip(year_headers, month_headers)
            )
            if header_year == year and header_month == month
        ]
        if not matches:
            raise ValueError(f"В зарплатном файле нет периода {month} {year}")
        return matches[-1]

    columns_2023 = year_columns(2023)
    columns_2024 = year_columns(2024)
    june_2025 = month_column(2025, "июнь")
    june_2026 = month_column(2026, "июнь")
    result = pd.DataFrame({"subject": raw.iloc[3:, 0].map(normalize_region)})
    result["average_wage_2023"] = raw.iloc[3:, columns_2023].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1)
    result["average_wage_2024"] = raw.iloc[3:, columns_2024].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1)
    result["wage_growth_2024_pct"] = (
        result["average_wage_2024"] / result["average_wage_2023"] - 1
    ) * 100
    result["wage_june_2025"] = pd.to_numeric(raw.iloc[3:, june_2025], errors="coerce")
    result["wage_june_2026"] = pd.to_numeric(raw.iloc[3:, june_2026], errors="coerce")
    result["wage_growth_june_2026_yoy_pct"] = (
        result["wage_june_2026"] / result["wage_june_2025"] - 1
    ) * 100
    return result.drop_duplicates("subject", keep="last")


def read_monthly_regional_wages():
    """Возвращает помесячную зарплату по субъектам с 2019 по июнь 2026 года."""
    raw = pd.read_excel(WAGE_VALIDATION_FILE, sheet_name="с 2019", header=None)
    year_headers = raw.iloc[1, 1:].ffill().map(
        lambda value: int(re.search(r"20\d{2}", str(value)).group())
    )
    month_names = {
        "январь": 1, "февраль": 2, "март": 3, "апрель": 4,
        "май": 5, "июнь": 6, "июль": 7, "август": 8,
        "сентябрь": 9, "октябрь": 10, "ноябрь": 11, "декабрь": 12,
    }
    month_headers = raw.iloc[2, 1:].map(
        lambda value: re.sub(r"\s+", " ", str(value)).strip().lower()
    )
    rows = []
    for offset, (year, month_name) in enumerate(zip(year_headers, month_headers)):
        month = month_names.get(month_name)
        if month is None:
            continue
        frame = pd.DataFrame({
            "subject": raw.iloc[3:, 0].map(normalize_region),
            "year": year,
            "month": month,
            "average_wage": pd.to_numeric(raw.iloc[3:, offset + 1], errors="coerce"),
        }).dropna()
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def ridge_wage_forecast(subjects, forecast_years=FORECAST_YEARS):
    """Месячный Ridge-прогноз зарплаты по регионам и backtest последних 12 месяцев."""
    wages = read_monthly_regional_wages()
    wages = wages.loc[wages["subject"].isin(subjects)].copy()
    categories = sorted(set(subjects))

    def design(frame):
        time = (
            (frame["year"].to_numpy(dtype=float) - 2019.0) * 12
            + frame["month"].to_numpy(dtype=float) - 1
        ).reshape(-1, 1)
        month_angle = 2 * np.pi * (frame["month"].to_numpy(dtype=float) - 1) / 12
        region = pd.Categorical(frame["subject"], categories=categories)
        one_hot = pd.get_dummies(region, dtype=float).to_numpy()
        # Общий тренд, сезонность, уровень региона и регуляризованный тренд региона.
        return np.column_stack([
            time, np.sin(month_angle), np.cos(month_angle),
            one_hot[:, 1:], one_hot[:, 1:] * time,
        ])

    validation_start = pd.Timestamp(2025, 6, 1)
    wages["date"] = pd.to_datetime(dict(year=wages.year, month=wages.month, day=1))
    train = wages.loc[wages["date"] < validation_start].copy()
    validation = wages.loc[wages["date"] >= validation_start].copy()
    scaler_validation = StandardScaler()
    x_train = scaler_validation.fit_transform(design(train))
    validation_model = RidgeCV(alphas=np.logspace(-3, 4, 40)).fit(
        x_train, np.log(train["average_wage"].to_numpy())
    )
    predicted_validation = np.exp(
        validation_model.predict(scaler_validation.transform(design(validation)))
    )
    validation_mape = float(
        mean_absolute_percentage_error(validation["average_wage"], predicted_validation)
    )

    scaler = StandardScaler()
    x_full = scaler.fit_transform(design(wages))
    model = RidgeCV(alphas=np.logspace(-3, 4, 40)).fit(
        x_full, np.log(wages["average_wage"].to_numpy())
    )
    model_years = (2025, 2026, *forecast_years)
    future = pd.MultiIndex.from_product(
        [categories, model_years, range(1, 13)],
        names=["subject", "year", "month"],
    ).to_frame(index=False)
    future["predicted_average_wage"] = np.exp(
        model.predict(scaler.transform(design(future)))
    )
    annual_future = future.groupby(["subject", "year"], as_index=False)[
        "predicted_average_wage"
    ].mean()
    baseline = annual_future.loc[
        annual_future["year"].eq(2025), ["subject", "predicted_average_wage"]
    ].rename(columns={"predicted_average_wage": "predicted_average_wage_2025"})
    result = annual_future.merge(baseline, on="subject", how="left")
    result["wage_growth_factor_from_2025"] = (
        result["predicted_average_wage"] / result["predicted_average_wage_2025"]
    )
    result["ridge_alpha"] = float(model.alpha_)
    result["validation_mape_pct"] = validation_mape * 100
    month_count = int(wages[["year", "month"]].drop_duplicates().shape[0])
    return result, validation_mape, float(model.alpha_), month_count


def read_national_median_wage_history():
    """Фактическая годовая медианная зарплата России за 2019–2025 годы."""
    raw = pd.read_excel(MEDIAN_WAGE_HISTORY_FILE, sheet_name="Лист1", header=None)
    years = raw.iloc[3, 1:].map(lambda value: int(re.search(r"20\d{2}", str(value)).group()))
    values = pd.to_numeric(raw.iloc[4, 1:], errors="coerce")
    return pd.DataFrame({"year": years, "median_wage": values}).dropna()


def national_median_wage_scenario(forecast_years=FORECAST_YEARS):
    """Фактическая медиана и сценарий по официальному макроопросу ЦБ."""
    history = read_national_median_wage_history()
    median_2025 = float(history.loc[history["year"].eq(2025), "median_wage"].iloc[0])
    predicted = official_nominal_wage_scenario(forecast_years)
    predicted["median_wage"] = median_2025 * predicted["wage_growth_factor_from_2025"]
    actual = history.loc[history["year"].between(2020, 2025)].copy()
    actual["status"] = "Фактические данные Росстата"
    predicted["status"] = np.where(
        predicted["year"].eq(2026),
        "Сценарная оценка по прогнозу роста зарплаты из опроса ЦБ",
        np.where(
            predicted["year"].le(2029),
            "Сценарий по медиане макроэкономического опроса ЦБ",
            "Техническое продолжение сценария после горизонта опроса ЦБ",
        ),
    )
    return pd.concat([
        actual[["year", "median_wage", "status"]],
        predicted[["year", "median_wage", "status"]],
    ], ignore_index=True).sort_values("year")


def read_national_inflation_history():
    """Читает официальный ИПЦ России: декабрь к декабрю предыдущего года."""
    raw = pd.read_excel(ROSSTAT_CPI_HISTORY_FILE, sheet_name="Лист1", header=None)
    years = raw.iloc[3, 1:].map(
        lambda value: int(re.search(r"(?:19|20)\d{2}", str(value)).group())
    )
    price_indices = pd.to_numeric(raw.iloc[4, 1:], errors="coerce")
    history = pd.DataFrame({
        "year": years,
        "inflation_pct": price_indices - 100.0,
    }).dropna()
    return history.loc[history["year"].between(2000, 2025)].reset_index(drop=True)


def cbr_macro_survey_wages_inflation():
    """Читает июльский макроопрос ЦБ: зарплаты и среднегодовой ИПЦ.

    Для корректного сравнения с ростом номинальной зарплаты используется
    среднегодовой ИПЦ, а не показатель декабря к декабрю. Прогнозные медианы и
    границы 10–90-го процентилей читаются из официального Excel ЦБ. Фактические
    значения 2021–2025 взяты из сводной таблицы того же выпуска.
    """
    actual = pd.DataFrame({
        "year": range(2021, 2026),
        "salary_growth_pct": [11.5, 14.1, 14.6, 19.0, 14.3],
        "inflation_pct": [6.7, 13.8, 5.9, 8.4, 8.7],
        "real_wage_growth_pct": [4.5, 0.3, 8.2, 9.7, 5.2],
    })

    def read_latest_forecast(sheet_name):
        raw = pd.read_excel(CBR_MACRO_SURVEY_FILE, sheet_name=sheet_name, header=None)
        latest_columns = [
            column for column in range(raw.shape[1])
            if pd.to_datetime(raw.iloc[5, column], errors="coerce") == pd.Timestamp("2026-07-01")
        ]
        if len(latest_columns) != 1:
            raise ValueError(f"Не найден единственный столбец июля 2026 на листе {sheet_name}")
        column = latest_columns[0]
        records = []
        for offset, year in enumerate(range(2026, 2030)):
            records.append({
                "year": year,
                "median": float(raw.iloc[12 + offset, column]),
                "p10": float(raw.iloc[66 + offset, column]),
                "p90": float(raw.iloc[39 + offset, column]),
            })
        return pd.DataFrame(records).set_index("year")

    inflation = read_latest_forecast("2")
    wages = read_latest_forecast("7")
    forecast = pd.DataFrame({
        "year": inflation.index,
        "salary_growth_pct": wages["median"].round(1).to_numpy(),
        "inflation_pct": inflation["median"].round(1).to_numpy(),
        "salary_lower_pct": wages["p10"].round(2).to_numpy(),
        "salary_upper_pct": wages["p90"].round(2).to_numpy(),
        "inflation_lower_pct": inflation["p10"].round(2).to_numpy(),
        "inflation_upper_pct": inflation["p90"].round(2).to_numpy(),
        "real_wage_growth_pct": [4.1, 2.5, 2.6, 2.3],
    })
    actual["salary_lower_pct"] = np.nan
    actual["salary_upper_pct"] = np.nan
    actual["inflation_lower_pct"] = np.nan
    actual["inflation_upper_pct"] = np.nan
    frame = pd.concat([actual, forecast], ignore_index=True)
    frame["status"] = np.where(
        frame["year"].le(2025),
        "Факт из сводной таблицы макроэкономического опроса Банка России",
        "Медиана прогнозов участников июльского опроса Банка России",
    )
    metadata = {
        "publication_date": "15.07.2026",
        "survey_dates": "10–14.07.2026",
        "respondents": 31,
        "table_last_year": 2029,
        "inflation_measure": "ИПЦ в среднем за год",
        "wage_measure": "номинальная заработная плата, в среднем за год",
    }
    return frame, metadata


def official_nominal_wage_scenario(forecast_years=FORECAST_YEARS):
    """Накопленный рост оплаты относительно 2025 года.

    2026–2029: медианы июльского макроэкономического опроса Банка России.
    2030–2032: осторожное техническое продолжение с замедлением номинального
    роста до 6,0%, 5,5% и 5,0%; эти годы не являются прогнозом ЦБ.
    """
    macro, _ = cbr_macro_survey_wages_inflation()
    annual_growth = {
        int(row.year): float(row.salary_growth_pct) / 100
        for row in macro.loc[macro["year"].between(2026, 2029)].itertuples(index=False)
    }
    annual_growth.update({2030: 0.060, 2031: 0.055, 2032: 0.050})
    factor = 1.0
    records = []
    for year in range(2026, max(forecast_years) + 1):
        factor *= 1 + annual_growth[year]
        records.append({
            "year": year,
            "annual_nominal_wage_growth_pct": annual_growth[year] * 100,
            "wage_growth_factor_from_2025": factor,
            "wage_scenario_source": (
                "Медиана макроэкономического опроса Банка России, июль 2026"
                if year <= 2029 else
                "Техническое продолжение после горизонта опроса ЦБ"
            ),
        })
    return pd.DataFrame(records)


def read_regional_median_wage(subjects):
    """Читает медианную зарплату за апрель 2025 года из таблицы 4.7 Росстата."""
    reader = PdfReader(str(REGION_INDICATORS_FILE))
    value = r"(?:\d{1,3}(?: \d{3})?|…|\.\.\.)"
    row_pattern = re.compile(rf"^(.*?)\s+({value}(?:\s+{value}){{8}})\s*$")
    subject_set = set(subjects)
    records = []
    for pdf_page in (195, 196):
        buffer = ""
        text = reader.pages[pdf_page - 1].extract_text() or ""
        for line in text.splitlines():
            normalized = re.sub(r"\s+", " ", line).strip()
            if not normalized:
                continue
            buffer = f"{buffer} {normalized}".strip()
            match = row_pattern.match(buffer)
            if match:
                name = re.sub(r"^в том числе:\s*", "", match.group(1)).strip()
                name = normalize_region(name)
                values = re.findall(value, match.group(2))
                if name in subject_set and values[-1] not in {"…", "..."}:
                    records.append({
                        "subject": name,
                        "median_wage_2025": float(values[-1].replace(" ", "")),
                    })
                buffer = ""
            elif len(buffer) > 450:
                buffer = normalized
    result = pd.DataFrame(records).drop_duplicates("subject", keep="last")
    missing = sorted(subject_set - set(result["subject"]))
    if missing:
        raise ValueError(f"Не прочитана медианная зарплата для субъектов: {missing}")
    return result


def read_subsistence_minimum(subjects):
    """Читает прожиточный минимум трудоспособного населения за 2025 год."""
    raw = pd.read_excel(SUBSISTENCE_MINIMUM_FILE, sheet_name="2013-2026", header=None)
    result = raw.iloc[5:, [0, 54]].copy()  # BC — трудоспособное население, 2025.
    result.columns = ["subject", "working_age_subsistence_2025"]
    result["subject"] = result["subject"].map(normalize_region)
    cleaned_values = (
        result["working_age_subsistence_2025"].astype(str)
        .str.replace(r"[^0-9,.-]", "", regex=True)
        .replace("", np.nan)
        .str.replace(",", ".", regex=False)
    )
    result["working_age_subsistence_2025"] = pd.to_numeric(
        cleaned_values, errors="coerce"
    )
    result = result.loc[
        result["subject"].isin(subjects),
        ["subject", "working_age_subsistence_2025"],
    ].dropna().drop_duplicates("subject", keep="last")
    if "Севастополь" in set(subjects):
        result = pd.concat([
            result.loc[~result["subject"].eq("Севастополь")],
            pd.DataFrame([{
                "subject": "Севастополь",
                "working_age_subsistence_2025": 19716.0,
            }]),
        ], ignore_index=True)
    missing = sorted(set(subjects) - set(result["subject"]))
    if missing:
        raise ValueError(f"Не прочитан прожиточный минимум для субъектов: {missing}")
    return result


def read_professional_wages():
    """Читает среднюю зарплату по девяти укрупнённым группам за октябрь 2025."""
    raw = pd.read_excel(
        PROFESSIONAL_WAGE_FILE,
        sheet_name="2015,2017, 2019, 2021, 2023",
        header=None,
    )
    rows = raw.iloc[4:, [0, 6]].copy()
    rows.columns = ["professional_group_raw", "national_average_wage_2025"]
    rows["national_average_wage_2025"] = pd.to_numeric(
        rows["national_average_wage_2025"], errors="coerce"
    )
    rows = rows.dropna(subset=["national_average_wage_2025"])
    national_all = float(
        rows.loc[rows["professional_group_raw"].astype(str).str.strip().eq("Всего"),
                 "national_average_wage_2025"].iloc[0]
    )
    wage_labels = {
        "руководители": "Руководители",
        "специалисты высшего уровня квалификации": "Специалисты высшей квалификации",
        "специалисты среднего уровня квалификации": "Специалисты средней квалификации",
        "служащие, занятые подготовкой и оформлением документации, учетом и обслуживанием": "Офисные и учётные служащие",
        "работники сферы обслуживания и торговли, охраны граждан и собственности": "Обслуживание, торговля и охрана",
        "квалифицированные работники сельского и лесного хозяйства, рыбоводства и рыболовства": "Сельское и лесное хозяйство",
        "квалифицированные рабочие промышленности, строительства, транспорта и рабочие родственных занятий": "Промышленность, строительство и транспорт",
        "операторы производственных установок и машин, сборщики и водители": "Операторы машин, сборщики и водители",
        "неквалифицированные рабочие": "Неквалифицированные рабочие",
    }
    rows["professional_group"] = rows["professional_group_raw"].map(
        lambda value: wage_labels.get(str(value).strip().lower())
    )
    rows = rows.dropna(subset=["professional_group"])
    rows["national_professional_wage_factor"] = (
        rows["national_average_wage_2025"] / national_all
    )
    return rows[[
        "professional_group", "national_average_wage_2025",
        "national_professional_wage_factor",
    ]].drop_duplicates("professional_group")


def read_regional_labor_market_indicators(subjects):
    """Извлекает из официального сборника Росстата таблицы 3.15 и 3.24.

    Таблица 3.15 даёт обычную безработицу и более широкий показатель LU3:
    безработные плюс потенциальная рабочая сила. Таблица 3.24 нужна для
    проверки здравого смысла индекса: официальный коэффициент напряжённости,
    среднее время поиска и доля длительной безработицы.
    """
    reader = PdfReader(str(REGION_INDICATORS_FILE))
    value_pattern = re.compile(
        r"^(.*?)\s+([0-9]+,[0-9])\s+([0-9]+,[0-9])\s+([0-9]+,[0-9])"
        r"\s+([0-9]+,[0-9])\s+([0-9]+,[0-9])\s+([0-9]+,[0-9])\s*$"
    )
    broad_rows = []
    for pdf_page in (153, 154):
        text = reader.pages[pdf_page - 1].extract_text() or ""
        for line in text.splitlines():
            match = value_pattern.match(re.sub(r"\s+", " ", line).strip())
            if not match:
                continue
            values = [float(value.replace(",", ".")) for value in match.groups()[1:]]
            broad_rows.append({
                "subject": normalize_region(match.group(1)),
                "unemployment_rate_2024_annual": values[2],
                "broad_underutilization_rate_2024": values[5],
            })
    broad = pd.DataFrame(broad_rows).drop_duplicates("subject", keep="last")

    tension_text = " ".join(
        reader.pages[pdf_page - 1].extract_text() or ""
        for pdf_page in (171, 172)
    )
    tension_text = re.sub(r"\s+", " ", tension_text)
    pdf_name_variants = {
        "Москва": "г. Москва",
        "Санкт-Петербург": "г. Санкт-Петербург",
        "Севастополь": "г. Севастополь",
        "Республика Северная Осетия-Алания": "Республика Северная Осетия –Алания",
        "Ханты-Мансийский автономный округ — Югра": (
            "Ханты-Мансийский автономный округ – Югра"
        ),
        "Архангельская область": "Архангельская область5)",
        "Тюменская область": "Тюменская область5)",
    }
    token = r"(?:\d+(?:,\d+)?|…)"
    metric_pattern = r"\s+" + r"\s+".join(f"({token})" for _ in range(10))
    tension_rows = []
    for subject in subjects:
        pdf_name = pdf_name_variants.get(subject, subject)
        match = re.search(re.escape(pdf_name) + metric_pattern, tension_text)
        if not match:
            continue
        values = [
            np.nan if value == "…" else float(value.replace(",", "."))
            for value in match.groups()
        ]
        tension_rows.append({
            "subject": subject,
            "official_employment_rate_2024": values[0],
            "official_unemployment_rate_2024": values[1],
            "official_tension_coefficient_2024": values[2],
            "average_job_search_months_2024": values[3],
            "long_term_unemployment_share_2024": values[4],
        })
    tension = pd.DataFrame(tension_rows)
    result = broad.merge(tension, on="subject", how="inner")
    result = result.loc[result["subject"].isin(subjects)].drop_duplicates("subject")
    missing = sorted(set(subjects) - set(result["subject"]))
    if missing:
        raise ValueError(
            "Не удалось извлечь официальные показатели рынка труда для: "
            + ", ".join(missing)
        )
    result.to_csv(REGIONAL_LABOR_MARKET_FILE, index=False, encoding="utf-8-sig")
    return result


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def score_label(score):
    if score >= 8.5:
        return "Очень трудно найти приемлемо оплачиваемую работу: выбор мал относительно числа соискателей"
    if score >= 7:
        return "Трудно найти приемлемо оплачиваемую работу: конкуренция и зарплатные ограничения выше среднего"
    if score >= 5:
        return "Скорее трудно найти приемлемо оплачиваемую работу"
    if score >= 3:
        return "Умеренная сложность поиска приемлемо оплачиваемой работы"
    return "Сравнительно легче найти приемлемо оплачиваемую работу"


def read_professional_sheet(sheet_name, value_name):
    """Читает региональную таблицу 1-Т(проф) с девятью группами занятий."""
    raw = pd.read_excel(PROFESSIONAL_DEMAND_FILE, sheet_name=sheet_name, header=None)
    header_candidates = raw.index[
        raw.iloc[:, 2].astype(str).str.contains("руководители", case=False, na=False)
    ]
    if len(header_candidates) == 0:
        raise ValueError(f"Не найдена строка профессиональных групп на листе {sheet_name}")
    header_row = int(header_candidates[0])
    group_names = [re.sub(r"\s+", " ", str(value)).strip() for value in raw.iloc[header_row, 2:11]]
    frame = raw.iloc[header_row + 1:, :11].copy()
    frame.columns = ["subject", f"total_{value_name}", *group_names]
    frame["subject"] = frame["subject"].map(normalize_region)
    frame = frame.dropna(subset=["subject"])
    for column in frame.columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[f"total_{value_name}"])
    long = frame.melt(
        id_vars=["subject", f"total_{value_name}"],
        value_vars=group_names,
        var_name="professional_group_raw",
        value_name=value_name,
    )
    return long, group_names


GROUP_LABELS = {
    "руководители": "Руководители",
    "специалисты высшего уровня квалификации": "Специалисты высшей квалификации",
    "специалисты среднего уровня квалификации": "Специалисты средней квалификации",
    "служащие, занятые подготовкой и оформлением документации, учетом и обслуживанием": "Офисные и учётные служащие",
    "работники сферы обслуживания и торговли, охраны граждан и собственности": "Обслуживание, торговля и охрана",
    "квалифицированные работники, сельского и лесного хозяйств, рыбоводства и рыболовства": "Сельское и лесное хозяйство",
    "квалифицированные рабочие промышленности, строительства, транспорта и рабочие родственных занятий": "Промышленность, строительство и транспорт",
    "операторы производственных установок и машин, сборщики и водители": "Операторы машин, сборщики и водители",
    "неквалифицированные рабочие": "Неквалифицированные рабочие",
}

# В листах 20–22 Росстат публикует детальные профессии по России в целом.
# Они дополняют девять региональных групп из листов 33, 35 и 37 и позволяют
# находить профессию, даже если она не входит в показанный по умолчанию топ.
DETAIL_PARENT_MARKERS = {
    "Специалисты высшего уровня квалификации": "Специалисты высшей квалификации",
    "Специалисты среднего уровня квалификации": "Специалисты средней квалификации",
    "Служащие, занятые подготовкой и оформление документации, учетом и обслуживанием": "Офисные и учётные служащие",
    "Работники сферы обслуживания и торговли, охраны граждан и собственности": "Обслуживание, торговля и охрана",
    "Квалифицированные работники сельского и лесного хозяйства, рыбоводства и рыболовства": "Сельское и лесное хозяйство",
    "Квалифицированные рабочие промышленности, строительства, транспорта и рабочие родственных занятий": "Промышленность, строительство и транспорт",
    "Операторы производственных установок и машин, сборщики и водители": "Операторы машин, сборщики и водители",
}


def read_detailed_occupation_catalog():
    """Читает детальные профессии РФ и связывает их с региональной группой."""
    records = []
    for sheet_name in ("20", "21", "22"):
        raw = pd.read_excel(PROFESSIONAL_DEMAND_FILE, sheet_name=sheet_name, header=None)
        current_parent = None
        for _, row in raw.iterrows():
            name = re.sub(r"\s+", " ", str(row.iloc[0])).strip()
            if name in DETAIL_PARENT_MARKERS:
                current_parent = DETAIL_PARENT_MARKERS[name]
                continue
            staff_count = pd.to_numeric(row.iloc[1], errors="coerce")
            required_workers = pd.to_numeric(row.iloc[2], errors="coerce")
            vacancy_rate_pct = pd.to_numeric(row.iloc[3], errors="coerce")
            if current_parent and pd.notna(vacancy_rate_pct) and pd.notna(required_workers):
                records.append({
                    "name": name,
                    "parent_group": current_parent,
                    "staff_count": float(staff_count) if pd.notna(staff_count) else None,
                    "required_workers": float(required_workers),
                    "vacancy_rate_pct": float(vacancy_rate_pct),
                })
    catalog = pd.DataFrame(records).drop_duplicates(["name", "parent_group"])
    catalog["total_jobs"] = catalog["staff_count"].fillna(0) + catalog["required_workers"]
    catalog["raw_vacancy_rate"] = (catalog["vacancy_rate_pct"] / 100).clip(0, 1)

    # Для детальных профессий Росстат публикует данные только по России в целом.
    # Сила сглаживания теперь оценивается методом моментов, а не задаётся
    # медианным размером группы. Это уменьшает влияние случайно больших процентов
    # у профессий с несколькими рабочими местами, не удаляя такие профессии.
    parent_stats = estimate_beta_prior(
        catalog,
        group_column="parent_group",
        vacancies_column="required_workers",
        jobs_column="total_jobs",
    ).rename(columns={
        "prior_mean_rate": "parent_rate",
        "kappa_mom": "prior_strength",
    })
    catalog = catalog.join(
        parent_stats[["parent_rate", "prior_strength", "between_region_variance"]],
        on="parent_group",
    )
    catalog["credibility_weight"] = (
        catalog["total_jobs"] / (catalog["total_jobs"] + catalog["prior_strength"])
    ).clip(0, 1)
    catalog["adjusted_vacancy_rate"] = (
        catalog["credibility_weight"] * catalog["raw_vacancy_rate"]
        + (1 - catalog["credibility_weight"]) * catalog["parent_rate"]
    )
    # Для соискателя высокий процент свободных мест означает больше возможностей,
    # а не большую сложность. Поэтому первый сигнал разворачивается. Второй
    # сравнивает долю уже занятых работников профессии с долей вакансий: если
    # работников относительно больше, конкуренция за редкие вакансии выше.
    catalog["staff_share"] = (
        catalog["staff_count"].fillna(0) / catalog["staff_count"].fillna(0).sum()
    )
    catalog["vacancy_share"] = (
        catalog["required_workers"] / catalog["required_workers"].sum()
    )
    catalog["competition_log_ratio"] = np.log(
        (catalog["staff_share"] + 1e-9) / (catalog["vacancy_share"] + 1e-9)
    )
    catalog["vacancy_scarcity_signal"] = percentile_signal(
        catalog["adjusted_vacancy_rate"], higher_means_tightness=False
    )
    catalog["competition_signal"] = percentile_signal(
        catalog["competition_log_ratio"], higher_means_tightness=True
    )
    catalog["national_job_search_difficulty_signal"] = (
        0.65 * catalog["vacancy_scarcity_signal"]
        + 0.35 * catalog["competition_signal"]
    ).clip(0, 1)
    catalog["national_score"] = (
        1 + 9 * catalog["national_job_search_difficulty_signal"]
    ).round(2)
    return catalog.sort_values(
        ["national_score", "required_workers"], ascending=[False, False]
    ).reset_index(drop=True)


if not REGIONAL_BASE_FILE.exists():
    raise FileNotFoundError(
        "Не найден rosstat_regions_for_map.csv. Сначала выполните блок загрузки "
        "региональных Excel Росстата в ноутбуке."
    )
if not PROFESSIONAL_DEMAND_FILE.exists():
    raise FileNotFoundError(f"Не найден официальный файл: {PROFESSIONAL_DEMAND_FILE.name}")
if not GEOJSON_FILE.exists():
    raise FileNotFoundError(f"Не найден файл геометрии: {GEOJSON_FILE.name}")
if not REGION_REFERENCE_FILE.exists():
    raise FileNotFoundError(f"Не найден справочник субъектов: {REGION_REFERENCE_FILE.name}")
for required_source in [
    UNEMPLOYMENT_FILE, LABOR_FORCE_FILE, EMPLOYMENT_FILE,
    CURRENT_LABOR_FORCE_FILE, CURRENT_UNEMPLOYMENT_FILE,
    WAGE_VALIDATION_FILE, MEDIAN_WAGE_HISTORY_FILE, ROSSTAT_CPI_HISTORY_FILE,
    ROSSTAT_CPI_MONTHLY_FILE, CBR_MEDIUM_TERM_FORECAST_FILE,
    P4_METHOD_FILE, REGION_INDICATORS_FILE, SEVASTOPOL_LABOR_2024_FILE,
    SEVASTOPOL_LABOR_2026_FILE, SEVASTOPOL_YEARBOOK_FILE,
    ROSSTAT_LABOUR_FORCE_2026_FILE,
]:
    if not required_source.exists():
        raise FileNotFoundError(f"Не найден официальный источник: {required_source}")


regional = pd.read_csv(REGIONAL_BASE_FILE)
regional = regional.rename(columns={
    "Код_ISO_субъекта": "shapeISO",
    "Субъект_РФ": "subject",
    "Период": "period",
    "Уровень_безработицы_pct": "unemployment_rate",
    "Уровень_занятости_pct": "employment_rate",
    "Рабочая_сила_тыс_человек": "labor_force_thousand",
    "Участие_в_рабочей_силе_pct": "labor_force_participation_rate",
    "Профильное_или_высшее_образование_pct": "qualified_labor_force_share",
    "Индекс_напряженности_рынка_труда_1_10": "supply_score_2026",
})
regional["subject"] = regional["subject"].map(normalize_region)

region_reference = pd.read_csv(
    REGION_REFERENCE_FILE,
    dtype={"Код_региона": "string"},
)
region_reference["subject"] = region_reference["Субъект_РФ"].map(normalize_region)
region_reference["region_code"] = region_reference["Код_региона"].str.zfill(2)
region_reference["area_thousand_km2"] = pd.to_numeric(
    region_reference["Площадь_тыс_км2"], errors="coerce"
)

# Последний единый официальный срез Росстата для всех субъектов РФ,
# опубликованный на центральной странице демографической статистики.
population = pd.read_excel(
    POPULATION_FILE,
    sheet_name="Всего",
    header=None,
    skiprows=8,
    usecols=[0, 6],
    names=["subject", "population_people"],
)
population = population.dropna(subset=["subject", "population_people"])
population["subject"] = population["subject"].map(normalize_region)
population["population_people"] = pd.to_numeric(
    population["population_people"], errors="coerce"
)
population = population.loc[
    population["subject"].isin(region_reference["subject"]),
    ["subject", "population_people"],
].drop_duplicates("subject")
if "Севастополь" in set(region_reference["subject"]):
    population = pd.concat([
        population.loc[~population["subject"].eq("Севастополь")],
        pd.DataFrame([{"subject": "Севастополь", "population_people": 558302}]),
    ], ignore_index=True)
region_reference = region_reference.merge(
    population,
    on="subject",
    how="left",
    validate="one_to_one",
)
regional = regional.merge(
    region_reference[["subject", "region_code", "area_thousand_km2", "population_people"]],
    on="subject",
    how="left",
    validate="one_to_one",
)
missing_reference = regional.loc[
    regional[["region_code", "area_thousand_km2", "population_people"]].isna().any(axis=1), "subject"
].tolist()
if missing_reference:
    raise ValueError(f"Нет кода, площади или населения для субъектов: {missing_reference}")

staff, staff_groups = read_professional_sheet("33", "staff_count")
demand, demand_groups = read_professional_sheet("35", "required_workers")
rates, rate_groups = read_professional_sheet("37", "vacancy_rate_pct")
detailed_occupations = read_detailed_occupation_catalog()

if not (staff_groups == demand_groups == rate_groups):
    raise ValueError("Названия профессиональных групп не совпадают между листами 33, 35 и 37")

professional = (
    staff[["subject", "professional_group_raw", "staff_count", "total_staff_count"]]
    .merge(
        demand[["subject", "professional_group_raw", "required_workers", "total_required_workers"]],
        on=["subject", "professional_group_raw"],
        how="inner",
    )
    .merge(
        rates[["subject", "professional_group_raw", "vacancy_rate_pct", "total_vacancy_rate_pct"]],
        on=["subject", "professional_group_raw"],
        how="inner",
    )
)
professional["professional_group"] = professional["professional_group_raw"].map(
    lambda value: GROUP_LABELS.get(str(value).strip().lower(), str(value).strip())
)

unemployment_2024, unemployment_period_2024 = read_indicator_period(
    UNEMPLOYMENT_FILE, "4", "unemployment_rate_2024", INDEX_PERIOD_PATTERN
)
employment_2024, employment_period_2024 = read_indicator_period(
    EMPLOYMENT_FILE, "4", "employment_rate_2024", INDEX_PERIOD_PATTERN
)
labor_force_2024, labor_force_period_2024 = read_indicator_period(
    LABOR_FORCE_FILE, "3", "labor_force_thousand_2024", INDEX_PERIOD_PATTERN
)
aligned_2024 = unemployment_2024.merge(
    employment_2024, on="subject", how="inner"
).merge(labor_force_2024, on="subject", how="inner")
sev_2024 = read_sevastopol_labor(SEVASTOPOL_LABOR_2024_FILE, row_from_end=2)
aligned_2024 = pd.concat([aligned_2024.loc[~aligned_2024["subject"].eq("Севастополь")], pd.DataFrame([{
    "subject": "Севастополь",
    "unemployment_rate_2024": sev_2024["unemployment"],
    "employment_rate_2024": sev_2024["employment"],
    "labor_force_thousand_2024": sev_2024["labor_force"],
}])], ignore_index=True)
regional = regional.merge(aligned_2024, on="subject", how="left", validate="one_to_one")

# Самый свежий сопоставимый трёхмесячный срез Росстата используется как
# информационный контекст, но не подменяет базовый 2024 год в составном индексе:
# остальные его компоненты за март–май 2026 ещё не опубликованы в том же разрезе.
labor_force_current, labor_force_current_period = read_indicator_period(
    CURRENT_LABOR_FORCE_FILE, "3", "labor_force_thousand_current", CURRENT_LABOR_PERIOD_PATTERN
)
unemployment_current, unemployment_current_period = read_indicator_period(
    CURRENT_UNEMPLOYMENT_FILE, "4", "unemployment_rate_current", CURRENT_LABOR_PERIOD_PATTERN
)
current_labor_context = labor_force_current.merge(
    unemployment_current, on="subject", how="inner", validate="one_to_one"
)
sev_current = read_sevastopol_labor(SEVASTOPOL_LABOR_2026_FILE)
current_labor_context = pd.concat([current_labor_context.loc[~current_labor_context["subject"].eq("Севастополь")], pd.DataFrame([{
    "subject": "Севастополь",
    "labor_force_thousand_current": sev_current["labor_force"],
    "unemployment_rate_current": sev_current["unemployment"],
}])], ignore_index=True)
current_labor_context["estimated_unemployed_current"] = (
    current_labor_context["labor_force_thousand_current"] * 1000
    * current_labor_context["unemployment_rate_current"] / 100
)
current_labor_context["period"] = labor_force_current_period
current_labor_context.loc[
    current_labor_context["subject"].eq("Севастополь"), "period"
] = "апрель–июнь 2026"
current_labor_context.to_csv(
    CURRENT_LABOR_CONTEXT_FILE, index=False, encoding="utf-8-sig"
)
regional = regional.merge(
    current_labor_context.drop(columns="period"),
    on="subject", how="left", validate="one_to_one",
)

professional = professional.merge(
    regional[["shapeISO", "subject", "labor_force_thousand_2024"]],
    on="subject",
    how="inner",
)

# Периоды согласованы: профессиональный спрос и все основные показатели
# трудоустройства относятся к 2024 году. Более свежие данные 2026 года
# остаются только справочным контекстом и не смешиваются с индексом.
regional_demand = (
    professional[[
        "shapeISO", "subject", "total_required_workers", "total_staff_count",
        "total_vacancy_rate_pct",
    ]]
    .drop_duplicates(["shapeISO", "subject"])
)
regional = regional.merge(regional_demand, on=["shapeISO", "subject"], how="left")
official_labor_market = read_regional_labor_market_indicators(
    regional["subject"].dropna().unique().tolist()
)
regional = regional.merge(
    official_labor_market, on="subject", how="left", validate="one_to_one"
)
regional_salary = read_regional_median_wage(regional["subject"].dropna().tolist())
regional_salary = regional_salary.merge(
    read_subsistence_minimum(regional["subject"].dropna().tolist()),
    on="subject",
    how="inner",
    validate="one_to_one",
)
regional = regional.merge(
    regional_salary, on="subject", how="left", validate="one_to_one"
)
regional["median_wage_to_subsistence_ratio_2025"] = (
    regional["median_wage_2025"] / regional["working_age_subsistence_2025"]
)
# Основной зарплатный показатель — абсолютный запас медианной зарплаты после
# вычитания регионального прожиточного минимума. В отличие от одного отношения,
# он различает, например, 45/15 тыс. руб. и 120/40 тыс. руб., хотя в обоих
# случаях зарплата формально равна трём прожиточным минимумам.
regional["median_wage_buffer_2025"] = (
    regional["median_wage_2025"] - regional["working_age_subsistence_2025"]
)
regional["salary_attractiveness_signal"] = percentile_signal(
    regional["median_wage_buffer_2025"],
    higher_means_tightness=True,
)
regional["Salary_Attractiveness_Score"] = (
    1 + 9 * regional["salary_attractiveness_signal"]
).round(2)
regional["low_salary_signal"] = percentile_signal(
    regional["median_wage_buffer_2025"],
    higher_means_tightness=False,
)
regional["estimated_unemployed_people_2024"] = (
    regional["unemployment_rate_2024"] / 100
    * regional["labor_force_thousand_2024"] * 1000
)
regional["vacancies_to_unemployed_ratio"] = (
    regional["total_required_workers"]
    / regional["estimated_unemployed_people_2024"].replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan)
regional["vacancies_per_10k_labor_force"] = (
    regional["total_required_workers"]
    / (regional["labor_force_thousand_2024"] * 1000) * 10000
).replace([np.inf, -np.inf], np.nan)

# Структурное несовпадение: доли девяти групп среди занятых сравниваются с
# долями тех же групп среди вакансий. Половина суммы абсолютных расхождений —
# total variation distance. Маленькая ниша не может «раздуть» регион: её вклад
# пропорционален её доле, а не единичному проценту внутри самой ниши.
professional["staff_share_region"] = (
    professional["staff_count"]
    / professional.groupby("subject")["staff_count"].transform("sum")
)
professional["vacancy_share_region"] = (
    professional["required_workers"]
    / professional.groupby("subject")["required_workers"].transform("sum")
)
professional["composition_abs_gap"] = (
    professional["staff_share_region"] - professional["vacancy_share_region"]
).abs()
regional_mismatch = (
    professional.groupby("subject", as_index=False)["composition_abs_gap"].sum()
    .rename(columns={"composition_abs_gap": "occupational_mismatch_tvd"})
)
regional_mismatch["occupational_mismatch_tvd"] *= 0.5
regional = regional.merge(
    regional_mismatch, on="subject", how="left", validate="one_to_one"
)

# Новый индекс отвечает на вопрос соискателя: насколько трудно найти не просто
# любую, а приемлемо оплачиваемую работу. 85% балла по-прежнему описывают
# доступность рабочих мест, 15% — низкую медианную зарплату относительно
# прожиточного минимума трудоспособного населения того же региона.
# Все компоненты переводятся в процентиль среди субъектов, поэтому абсолютный
# размер Москвы или единичный выброс не дают автоматической прибавки к баллу.
regional["broad_underutilization_signal"] = percentile_signal(
    regional["broad_underutilization_rate_2024"], higher_means_tightness=True
)
regional["vacancy_scarcity_signal"] = percentile_signal(
    regional["total_vacancy_rate_pct"], higher_means_tightness=False
)
regional["occupational_mismatch_signal"] = percentile_signal(
    regional["occupational_mismatch_tvd"], higher_means_tightness=True
)
regional["job_search_months_signal"] = percentile_signal(
    regional["average_job_search_months_2024"], higher_means_tightness=True
)
regional["long_term_unemployment_signal"] = percentile_signal(
    regional["long_term_unemployment_share_2024"], higher_means_tightness=True
)
regional["search_persistence_signal"] = regional[[
    "job_search_months_signal", "long_term_unemployment_signal",
]].mean(axis=1)

# Не подменяем отсутствующие профессиональные таблицы нейтральным значением.
# Для Севастополя официальный сопоставимый срез 1-Т(проф) опубликован лишь
# частично, поэтому общий индекс строится по доступным официальным компонентам.
regional.loc[regional["total_vacancy_rate_pct"].isna(), "vacancy_scarcity_signal"] = np.nan
regional.loc[regional["occupational_mismatch_tvd"].isna(), "occupational_mismatch_signal"] = np.nan

signal_columns = [
    "broad_underutilization_signal", "vacancy_scarcity_signal",
    "occupational_mismatch_signal", "search_persistence_signal",
    "low_salary_signal",
]
base_component_weights = pd.Series(
    [0.34, 0.30, 0.13, 0.08, 0.15], index=signal_columns
)
available_weight = regional[signal_columns].notna().mul(
    base_component_weights, axis=1
).sum(axis=1)
valid_index_inputs = available_weight.ge(0.50)
regional_scenarios = {
    "base": (0.34, 0.30, 0.13, 0.08, 0.15),
    "reserve_heavy": (0.42, 0.26, 0.09, 0.08, 0.15),
    "vacancy_heavy": (0.30, 0.38, 0.09, 0.08, 0.15),
    "mismatch_heavy": (0.30, 0.26, 0.21, 0.08, 0.15),
    "duration_heavy": (0.30, 0.26, 0.13, 0.16, 0.15),
    "salary_light": (0.36, 0.32, 0.14, 0.08, 0.10),
    "salary_heavy": (0.32, 0.28, 0.12, 0.08, 0.20),
}
regional_scenario_scores = []
regional_scenario_ranks = []
for scenario_name, weights in regional_scenarios.items():
    score_column = f"regional_score_{scenario_name}"
    rank_column = f"regional_rank_{scenario_name}"
    scenario_weights = pd.Series(weights, index=signal_columns)
    available = regional[signal_columns].notna().mul(scenario_weights, axis=1)
    weighted_signal = regional[signal_columns].mul(
        scenario_weights, axis=1
    ).sum(axis=1, min_count=1) / available.sum(axis=1).replace(0, np.nan)
    regional[score_column] = 1 + 9 * weighted_signal.clip(0, 1)
    regional[rank_column] = regional[score_column].rank(method="min", ascending=False)
    regional_scenario_scores.append(score_column)
    regional_scenario_ranks.append(rank_column)
regional["Regional_Job_Search_Difficulty_Score"] = regional[
    "regional_score_base"
].round(2)
regional["Regional_Tightness_Score_Min"] = regional[regional_scenario_scores].min(axis=1).round(2)
regional["Regional_Tightness_Score_Max"] = regional[regional_scenario_scores].max(axis=1).round(2)
regional["Regional_Rank_Min"] = regional[regional_scenario_ranks].min(axis=1).astype("Int64")
regional["Regional_Rank_Max"] = regional[regional_scenario_ranks].max(axis=1).astype("Int64")
regional["interpretation"] = regional[
    "Regional_Job_Search_Difficulty_Score"
].map(score_label)
regional["denominator_stability"] = np.where(
    regional[[
        "average_job_search_months_2024", "long_term_unemployment_share_2024",
    ]].isna().any(axis=1),
    "Для части показателей длительности поиска нет опубликованного значения; использован нейтральный процентиль.",
    "Все пять компонентов опубликованы; стандартные ошибки региона в сборнике не приведены.",
)
regional.loc[regional["subject"].eq("Севастополь"), "denominator_stability"] = (
    "Индекс рассчитан по трём доступным официальным компонентам из пяти; "
    "профессиональный срез 1-Т(проф) для города неполон, поэтому его два "
    "компонента исключены, а веса доступных показателей пропорционально пересчитаны."
)
regional["coverage_note"] = (
    "Вакансии относятся к организациям без субъектов малого предпринимательства; "
    "показатели населения — к выборочному обследованию рабочей силы."
)

# Жёсткие проверки экономических тождеств и границ индекса.
recomputed_vacancy_rate = (
    regional["total_required_workers"]
    / (regional["total_staff_count"] + regional["total_required_workers"])
    * 100
)
rate_difference = (
    recomputed_vacancy_rate - regional["total_vacancy_rate_pct"]
).abs().dropna()
if not rate_difference.empty and float(rate_difference.max()) > 0.02:
    raise ValueError("Доля вакансий не согласуется с V / (E + V)")
valid_scores = regional.loc[
    valid_index_inputs, "Regional_Job_Search_Difficulty_Score"
]
if not valid_scores.between(1, 10).all():
    raise ValueError("Региональный индекс вышел за границы 1–10")
component_correlation = regional.loc[valid_index_inputs, signal_columns].corr(
    method="spearman"
)
component_spearman = float(
    component_correlation.where(~np.eye(len(signal_columns), dtype=bool)).abs().max().max()
)
# Версия 2.0 сохраняется только как контроль направления. Она измеряла
# сложность найма для работодателя, поэтому не должна подменять новую задачу.
legacy_vacancy_pressure = percentile_signal(
    regional["total_vacancy_rate_pct"], higher_means_tightness=True
)
legacy_reserve_scarcity = percentile_signal(
    regional["unemployment_rate_2024"], higher_means_tightness=False
)
regional["legacy_employer_hiring_score"] = 1 + 9 * (
    0.5 * legacy_vacancy_pressure + 0.5 * legacy_reserve_scarcity
).clip(0, 1)
wage_validation = read_wage_validation()
regional = regional.merge(wage_validation, on="subject", how="left", validate="one_to_one")
validation_rows = regional[[
    "Regional_Job_Search_Difficulty_Score", "official_tension_coefficient_2024",
]].dropna()
validation_spearman = float(
    validation_rows["Regional_Job_Search_Difficulty_Score"].rank().corr(
        validation_rows["official_tension_coefficient_2024"].rank()
    )
) if len(validation_rows) >= 3 else np.nan
legacy_validation_rows = regional[[
    "legacy_employer_hiring_score", "official_tension_coefficient_2024",
]].dropna()
legacy_validation_spearman = float(
    legacy_validation_rows["legacy_employer_hiring_score"].rank().corr(
        legacy_validation_rows["official_tension_coefficient_2024"].rank()
    )
) if len(legacy_validation_rows) >= 3 else np.nan
regional["score_without_search_persistence"] = 1 + 9 * (
    (0.34 / 0.92) * regional["broad_underutilization_signal"]
    + (0.30 / 0.92) * regional["vacancy_scarcity_signal"]
    + (0.13 / 0.92) * regional["occupational_mismatch_signal"]
    + (0.15 / 0.92) * regional["low_salary_signal"]
).clip(0, 1)
search_duration_validation_spearman = float(
    regional[[
        "score_without_search_persistence", "average_job_search_months_2024",
    ]].dropna().corr(method="spearman").iloc[0, 1]
)
wage_validation_spearman = float(
    regional[[
        "Regional_Job_Search_Difficulty_Score", "wage_growth_2024_pct",
    ]].dropna().corr(method="spearman").iloc[0, 1]
)
if pd.isna(validation_spearman):
    validation_summary = "Сопоставление с официальной напряжённостью невозможно."
elif validation_spearman >= 0.60:
    validation_summary = (
        f"Сильное согласование с официальным коэффициентом напряжённости: "
        f"ρ={validation_spearman:.2f}."
    )
elif validation_spearman >= 0.40:
    validation_summary = (
        f"Умеренное согласование с официальным коэффициентом напряжённости: "
        f"ρ={validation_spearman:.2f}."
    )
else:
    validation_summary = (
        f"Согласование с официальным коэффициентом напряжённости слабое: "
        f"ρ={validation_spearman:.2f}; индекс требует осторожной интерпретации."
    )

# Профессиональный рейтинг использует тот же период 2024 года.
professional = professional.merge(
    regional[[
        "shapeISO", "Regional_Job_Search_Difficulty_Score",
        "median_wage_2025", "working_age_subsistence_2025",
        "Salary_Attractiveness_Score",
    ]],
    on="shapeISO",
    how="left",
    validate="many_to_one",
)
professional["total_group_jobs"] = (
    professional["staff_count"].fillna(0) + professional["required_workers"].fillna(0)
)
professional["raw_vacancy_rate"] = (
    pd.to_numeric(professional["vacancy_rate_pct"], errors="coerce") / 100
).clip(0, 1)

# Нулевой спрос в микроскопической региональной нише — не доказательство того,
# что соискателю «максимально трудно». OECD при сравнении отраслей исключает
# рынки с занятостью ниже 5% медианы региона. Здесь используется мягче:
# без ранга остаётся только сочетание нулевого спроса и практически отсутствующей
# базы занятых; малый рынок с наблюдаемым спросом сохраняется и сглаживается.
professional["regional_median_group_staff"] = professional.groupby("subject")[
    "staff_count"
].transform("median")
professional["market_presence_to_region_median"] = (
    professional["staff_count"]
    / professional["regional_median_group_staff"].replace(0, np.nan)
)
professional["tiny_local_market"] = (
    (professional["market_presence_to_region_median"] < 0.05)
    | (professional["staff_share_region"] < 0.005)
)
professional["structural_zero_market"] = (
    professional["tiny_local_market"]
    & (professional["required_workers"].fillna(0) < 0.5)
)
professional["small_market_with_demand"] = (
    professional["tiny_local_market"]
    & ~professional["structural_zero_market"]
)

professional = professional.merge(
    read_professional_wages(),
    on="professional_group",
    how="left",
    validate="many_to_one",
)
if professional["national_professional_wage_factor"].isna().any():
    missing_groups = sorted(
        professional.loc[
            professional["national_professional_wage_factor"].isna(),
            "professional_group",
        ].unique()
    )
    raise ValueError(f"Нет зарплатного коэффициента для групп: {missing_groups}")
professional["estimated_professional_wage_2025"] = (
    professional["median_wage_2025"]
    * professional["national_professional_wage_factor"]
)
professional["estimated_wage_to_subsistence_ratio"] = (
    professional["estimated_professional_wage_2025"]
    / professional["working_age_subsistence_2025"]
)
professional["estimated_professional_wage_buffer_2025"] = (
    professional["estimated_professional_wage_2025"]
    - professional["working_age_subsistence_2025"]
)
professional["salary_attractiveness_signal"] = percentile_signal(
    professional["estimated_professional_wage_buffer_2025"],
    higher_means_tightness=True,
)
professional["Professional_Salary_Attractiveness_Score"] = (
    1 + 9 * professional["salary_attractiveness_signal"]
).round(2)
# Вакансия с оплатой ниже региональной медианы учитывается частично; вакансия
# с медианной или более высокой оплатой остаётся одной вакансией, а не
# «размножается» из-за высокой зарплаты.
professional["salary_quality_weight"] = (
    professional["estimated_professional_wage_2025"]
    / professional["median_wage_2025"]
).clip(0.25, 1.0)
professional["salary_adjusted_required_workers"] = (
    professional["required_workers"] * professional["salary_quality_weight"]
)

# Сила сглаживания kappa оценивается методом моментов из самих данных.
group_priors = estimate_beta_prior(
    professional.loc[~professional["structural_zero_market"]],
    group_column="professional_group",
    vacancies_column="required_workers",
    jobs_column="total_group_jobs",
).rename(columns={
    "prior_mean_rate": "national_group_rate",
    "kappa_mom": "prior_strength",
})
professional = professional.join(
    group_priors[["national_group_rate", "prior_strength", "between_region_variance"]],
    on="professional_group",
)
professional["credibility_weight"] = (
    professional["total_group_jobs"]
    / (professional["total_group_jobs"] + professional["prior_strength"])
).clip(0, 1)
# Дополнительная защита малых локальных рынков от случайного нуля/скачка.
professional["market_relevance_weight"] = (
    professional["staff_share_region"] / 0.01
).clip(0, 1)
professional["credibility_weight"] *= professional["market_relevance_weight"]
professional["adjusted_vacancy_rate"] = (
    professional["credibility_weight"] * professional["raw_vacancy_rate"]
    + (1 - professional["credibility_weight"]) * professional["national_group_rate"]
)

# Сколько работников требуется на 10 тысяч участников рабочей силы остаётся
# справочным показателем. В индекс он не входит, чтобы крупный регион или
# большая профессиональная группа не получали автоматическую прибавку.
professional["vacancies_per_10k_labor_force"] = (
    professional["required_workers"]
    / (professional["labor_force_thousand_2024"] * 1000)
    * 10000
).replace([np.inf, -np.inf], np.nan)

national_staff_share = (
    professional.groupby("professional_group")["staff_count"].sum()
    / professional["staff_count"].sum()
)
professional["salary_adjusted_vacancy_rate"] = (
    professional["adjusted_vacancy_rate"] * professional["salary_quality_weight"]
)
national_vacancy_share = (
    professional.groupby("professional_group")["required_workers"].sum()
    / professional["required_workers"].sum()
)
professional["national_staff_share"] = professional["professional_group"].map(
    national_staff_share
)
professional["national_vacancy_share"] = professional["professional_group"].map(
    national_vacancy_share
)
professional["adjusted_staff_share"] = (
    professional["credibility_weight"] * professional["staff_share_region"]
    + (1 - professional["credibility_weight"]) * professional["national_staff_share"]
)
professional["adjusted_vacancy_share"] = (
    professional["credibility_weight"] * professional["vacancy_share_region"]
    + (1 - professional["credibility_weight"]) * professional["national_vacancy_share"]
)
professional["competition_log_ratio"] = np.log(
    (professional["adjusted_staff_share"] + 1e-9)
    / (professional["adjusted_vacancy_share"] + 1e-9)
)
professional["region_context_signal"] = (
    professional["Regional_Job_Search_Difficulty_Score"] - 1
) / 9
professional_scenario_scores = []
professional_scenario_ranks = []
for kappa_multiplier in (0.5, 1.0, 2.0):
    scenario_kappa = professional["prior_strength"] * kappa_multiplier
    scenario_weight = (
        professional["total_group_jobs"]
        / (professional["total_group_jobs"] + scenario_kappa)
    ).clip(0, 1) * professional["market_relevance_weight"]
    scenario_rate = (
        scenario_weight * professional["raw_vacancy_rate"]
        + (1 - scenario_weight) * professional["national_group_rate"]
    ) * professional["salary_quality_weight"]
    scenario_staff_share = (
        scenario_weight * professional["staff_share_region"]
        + (1 - scenario_weight) * professional["national_staff_share"]
    )
    scenario_vacancy_share = (
        scenario_weight * professional["vacancy_share_region"]
        + (1 - scenario_weight) * professional["national_vacancy_share"]
    )
    scenario_competition = np.log(
        (scenario_staff_share + 1e-9) / (scenario_vacancy_share + 1e-9)
    )
    scenario_vacancy_scarcity = percentile_signal(
        scenario_rate, higher_means_tightness=False
    )
    scenario_competition_signal = percentile_signal(
        scenario_competition, higher_means_tightness=True
    )
    for vacancy_weight, competition_weight, region_weight in (
        (0.45, 0.35, 0.20),
        (0.55, 0.30, 0.15),
        (0.65, 0.25, 0.10),
    ):
        suffix = (
            f"k{str(kappa_multiplier).replace('.', '_')}"
            f"_v{int(vacancy_weight * 100)}"
        )
        score_column = f"professional_score_{suffix}"
        rank_column = f"professional_rank_{suffix}"
        professional[score_column] = 1 + 9 * (
            vacancy_weight * scenario_vacancy_scarcity
            + competition_weight * scenario_competition_signal
            + region_weight * professional["region_context_signal"]
        ).clip(0, 1)
        professional.loc[
            professional["structural_zero_market"], score_column
        ] = np.nan
        professional[rank_column] = professional.groupby("shapeISO")[score_column].rank(
            method="min", ascending=False
        )
        professional_scenario_scores.append(score_column)
        professional_scenario_ranks.append(rank_column)
professional["Professional_Job_Search_Difficulty_Score"] = professional[
    "professional_score_k1_0_v55"
].round(2)
professional["Professional_Job_Search_Difficulty_Score_Min"] = professional[
    professional_scenario_scores
].min(axis=1).round(2)
professional["Professional_Job_Search_Difficulty_Score_Max"] = professional[
    professional_scenario_scores
].max(axis=1).round(2)
professional["Professional_Rank_Min"] = professional[
    professional_scenario_ranks
].min(axis=1).astype("Int64")
professional["Professional_Rank_Max"] = professional[
    professional_scenario_ranks
].max(axis=1).astype("Int64")

# Диагностика нетипичных долей: устойчивый z-score по логиту внутри каждой
# группы. Наблюдения не удаляются — флаг лишь помогает увидеть выброс.
clipped_rate = professional["raw_vacancy_rate"].clip(1e-6, 1 - 1e-6)
professional["rate_logit"] = np.log(clipped_rate / (1 - clipped_rate))
grouped_logit = professional.groupby("professional_group")["rate_logit"]
logit_median = grouped_logit.transform("median")
logit_mad = grouped_logit.transform(lambda values: (values - values.median()).abs().median())
professional["robust_rate_z"] = (
    0.6745 * (professional["rate_logit"] - logit_median) / logit_mad.replace(0, np.nan)
)
professional["base_status"] = np.select(
    [
        professional["structural_zero_market"],
        professional["small_market_with_demand"],
        professional["credibility_weight"] < 0.35,
        professional["credibility_weight"] < 0.65,
    ],
    [
        "Локальный рынок почти отсутствует: нулевой спрос не превращён в максимальный балл",
        "Малый локальный рынок, но спрос наблюдается — строка сохранена со сглаживанием",
        "Малая база — сильное сглаживание",
        "Средняя база — умеренное сглаживание",
    ],
    default="Устойчивая база",
)
professional["outlier_note"] = np.select(
    [professional["robust_rate_z"] > 3.5, professional["robust_rate_z"] < -3.5],
    [
        "; нетипично высокая доля — строка сохранена",
        "; нетипично низкая доля — строка сохранена",
    ],
    default="",
)
professional["observation_status"] = (
    professional["base_status"] + professional["outlier_note"]
)
professional["rank_in_region"] = professional.groupby("shapeISO")[
    "Professional_Job_Search_Difficulty_Score"
].rank(method="first", ascending=False).astype("Int64")
professional = professional.sort_values(
    ["shapeISO", "rank_in_region", "required_workers"],
    ascending=[True, True, False], na_position="last",
)

# Прогнозные сценарии 2027–2032. Текущий индекс карты не меняется. Будущий
# сценарий отделён от него и опирается на: срез спроса Росстата 31.10.2024,
# официальный макроопрос ЦБ по зарплате, слабую макродинамику/строительство 2026
# и официальный приоритет работников со средним профессиональным образованием.
# Ridge оставлен только как диагностический backtest исторического ряда; его
# механическая дальняя экстраполяция больше не определяет прогнозную зарплату.
wage_forecast, wage_forecast_mape, wage_ridge_alpha, wage_forecast_months = ridge_wage_forecast(
    regional["subject"].dropna().tolist()
)
national_median_series = national_median_wage_scenario()
wage_inflation_series, inflation_model_meta = cbr_macro_survey_wages_inflation()
wage_inflation_series.to_csv(WAGE_INFLATION_FILE, index=False, encoding="utf-8-sig")


def weighted_average(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.average(values, weights=np.maximum(weights, 1.0)))


group_current = (
    professional.groupby("professional_group", as_index=False)
    .apply(
        lambda frame: pd.Series({
            "staff_count_2024": frame["staff_count"].sum(),
            "required_workers_2024": frame["required_workers"].sum(),
            "baseline_vacancy_rate_2024": (
                frame["required_workers"].sum()
                / (frame["staff_count"].sum() + frame["required_workers"].sum())
            ),
            "current_salary_2025": weighted_average(
                frame["estimated_professional_wage_2025"], frame["staff_count"]
            ),
        }),
        include_groups=False,
    )
    .reset_index(drop=True)
)
group_current["demand_signal"] = percentile_signal(
    group_current["baseline_vacancy_rate_2024"], higher_means_tightness=True
)
group_current["current_salary_signal"] = percentile_signal(
    group_current["current_salary_2025"], higher_means_tightness=True
)
group_current["Demand_Score_1_10"] = 1 + 9 * group_current["demand_signal"]
group_current["Current_Salary_Attractiveness_1_10"] = (
    1 + 9 * group_current["current_salary_signal"]
)

wage_scenario = official_nominal_wage_scenario()
group_forecast = group_current.merge(
    wage_scenario.loc[wage_scenario["year"].isin(FORECAST_YEARS)], how="cross"
)
group_forecast["predicted_salary"] = (
    group_forecast["current_salary_2025"]
    * group_forecast["wage_growth_factor_from_2025"]
)
group_forecast["demand_scenario_factor"] = group_forecast.apply(
    lambda row: PROFESSIONAL_DEMAND_SCENARIO[row["professional_group"]][int(row["year"])],
    axis=1,
)
group_forecast["projected_required_workers"] = (
    group_forecast["required_workers_2024"]
    * group_forecast["demand_scenario_factor"]
)
group_forecast["projected_vacancy_rate"] = (
    group_forecast["projected_required_workers"]
    / (group_forecast["staff_count_2024"] + group_forecast["projected_required_workers"])
)
group_forecast["demand_signal"] = group_forecast.groupby("year")[
    "projected_vacancy_rate"
].transform(lambda series: percentile_signal(series, higher_means_tightness=True))
group_forecast["Demand_Score_1_10"] = 1 + 9 * group_forecast["demand_signal"]
group_forecast["scenario_note"] = group_forecast["professional_group"].map(
    PROFESSIONAL_SCENARIO_NOTE
)
group_forecast["salary_signal"] = group_forecast.groupby("year")[
    "predicted_salary"
].transform(lambda series: percentile_signal(series, higher_means_tightness=True))
group_forecast["Salary_Attractiveness_1_10"] = 1 + 9 * group_forecast["salary_signal"]
group_forecast["Salary_Index_Change"] = (
    group_forecast["Salary_Attractiveness_1_10"]
    - group_forecast["Current_Salary_Attractiveness_1_10"]
)
group_forecast["Salary_Growth_Pct"] = (
    group_forecast["predicted_salary"] / group_forecast["current_salary_2025"] - 1
) * 100
group_forecast["Opportunity_Score_1_10"] = 1 + 9 * (
    0.65 * group_forecast["demand_signal"] + 0.35 * group_forecast["salary_signal"]
)
group_forecast["Forecast_Rank"] = group_forecast.groupby("year")[
    "Opportunity_Score_1_10"
].rank(method="first", ascending=False).astype(int)
group_forecast = group_forecast.sort_values(["year", "Forecast_Rank"])

occupation_forecast = detailed_occupations.merge(
    group_forecast[[
        "year", "professional_group", "current_salary_2025", "predicted_salary",
        "Current_Salary_Attractiveness_1_10", "Salary_Attractiveness_1_10",
        "Salary_Index_Change", "Salary_Growth_Pct", "demand_scenario_factor",
        "scenario_note",
    ]].rename(columns={"professional_group": "parent_group"}),
    on="parent_group", how="left", validate="many_to_many",
)

# Для профессий-лидеров используем не одну зарплату крупной группы, а отдельную
# медиану актуальных зарплатных предложений официального портала «Работа России».
# Медиана считается по середине опубликованной вилки. При недостатке наблюдений
# скрипт сбора использует ближайшую строку полного обследования Росстата 57-Т.
if OCCUPATION_SALARY_SNAPSHOT_FILE.exists():
    occupation_salary = pd.read_csv(OCCUPATION_SALARY_SNAPSHOT_FILE).rename(columns={
        "Профессия": "name",
        "Медианная_предлагаемая_зарплата_2026_руб": "occupation_salary_base",
        "Источник_зарплаты": "occupation_salary_source",
        "Дата_среза": "occupation_salary_date",
        "Наблюдений_с_зарплатой": "occupation_salary_observations",
    })
    occupation_forecast = occupation_forecast.merge(
        occupation_salary[[
            "name", "occupation_salary_base", "occupation_salary_source",
            "occupation_salary_date", "occupation_salary_observations",
        ]],
        on="name", how="left", validate="many_to_one",
    )
else:
    occupation_forecast["occupation_salary_base"] = np.nan
    occupation_forecast["occupation_salary_source"] = np.nan
    occupation_forecast["occupation_salary_date"] = np.nan
    occupation_forecast["occupation_salary_observations"] = np.nan

occupation_forecast["salary_base_year"] = np.where(
    occupation_forecast["occupation_salary_base"].notna(), 2026, 2025
)
occupation_forecast["salary_source"] = occupation_forecast[
    "occupation_salary_source"
].fillna("Средняя зарплата крупной профессиональной группы Росстата за октябрь 2025")
wage_factor_by_year = official_nominal_wage_scenario().set_index("year")[
    "wage_growth_factor_from_2025"
]
factor_2026 = float(wage_factor_by_year.loc[2026])
individual_mask = occupation_forecast["occupation_salary_base"].notna()
occupation_forecast.loc[individual_mask, "current_salary_2025"] = occupation_forecast.loc[
    individual_mask, "occupation_salary_base"
]
occupation_forecast.loc[individual_mask, "predicted_salary"] = (
    occupation_forecast.loc[individual_mask, "occupation_salary_base"]
    * occupation_forecast.loc[individual_mask, "year"].map(wage_factor_by_year)
    / factor_2026
)
occupation_forecast.loc[individual_mask, "Salary_Growth_Pct"] = (
    occupation_forecast.loc[individual_mask, "predicted_salary"]
    / occupation_forecast.loc[individual_mask, "occupation_salary_base"] - 1
) * 100
occupation_forecast["projected_required_workers"] = (
    occupation_forecast["required_workers"]
    * occupation_forecast["demand_scenario_factor"]
)
occupation_forecast["projected_vacancy_rate"] = (
    occupation_forecast["projected_required_workers"]
    / (occupation_forecast["staff_count"] + occupation_forecast["projected_required_workers"])
)
occupation_forecast["demand_signal"] = occupation_forecast.groupby("year")[
    "projected_vacancy_rate"
].transform(lambda series: percentile_signal(series, higher_means_tightness=True))
occupation_forecast["Demand_Score_1_10"] = 1 + 9 * occupation_forecast["demand_signal"]
occupation_forecast["Opportunity_Score_1_10"] = 1 + 9 * (
    0.65 * occupation_forecast["demand_signal"]
    + 0.35 * ((occupation_forecast["Salary_Attractiveness_1_10"] - 1) / 9)
)
occupation_forecast["Forecast_Rank"] = occupation_forecast.groupby("year")[
    "Opportunity_Score_1_10"
].rank(method="first", ascending=False).astype(int)
occupation_forecast = occupation_forecast.sort_values(["year", "Forecast_Rank"])

group_forecast_export = group_forecast.rename(columns={
    "year": "Год_прогноза",
    "professional_group": "Профессиональная_группа",
    "staff_count_2024": "Занято_работников_2024",
    "required_workers_2024": "Требуется_работников_2024",
    "projected_required_workers": "Сценарная_потребность_работников",
    "projected_vacancy_rate": "Сценарная_доля_вакансий",
    "demand_scenario_factor": "Коэффициент_сценария_спроса",
    "scenario_note": "Основание_сценария",
    "baseline_vacancy_rate_2024": "Базовая_доля_вакансий_2024",
    "current_salary_2025": "Актуальная_зарплата_2025_руб",
    "predicted_salary": "Прогнозная_зарплата_руб",
    "Demand_Score_1_10": "Балл_спроса_1_10",
    "Current_Salary_Attractiveness_1_10": "Актуальный_индекс_оплаты_1_10",
    "Salary_Attractiveness_1_10": "Прогнозный_индекс_оплаты_1_10",
    "Salary_Index_Change": "Изменение_индекса_оплаты",
    "Salary_Growth_Pct": "Рост_зарплаты_к_2025_pct",
    "Opportunity_Score_1_10": "Итоговый_ориентир_1_10",
    "Forecast_Rank": "Место_в_прогнозе",
})
group_forecast_export["Базовая_доля_вакансий_2024_pct"] = (
    group_forecast_export.pop("Базовая_доля_вакансий_2024") * 100
).round(2)
group_forecast_export["Сценарная_доля_вакансий_pct"] = (
    group_forecast_export.pop("Сценарная_доля_вакансий") * 100
).round(2)
group_forecast_export["Ошибка_backtest_2025_MAPE_pct"] = round(
    wage_forecast_mape * 100, 2
)
group_forecast_export = group_forecast_export.drop(
    columns=["Изменение_индекса_оплаты"], errors="ignore"
)
group_forecast_export["Ограничение"] = (
    "Спрос — сценарная корректировка среза Росстата 31.10.2024, а не прогноз "
    "точного числа вакансий; зарплата 2026–2029 следует медиане июльского "
    "макроопроса ЦБ, 2030–2032 — техническое продолжение"
)
group_forecast_export.to_csv(GROUP_FORECAST_FILE, index=False, encoding="utf-8-sig")

occupation_forecast_export = occupation_forecast[[
    "year", "Forecast_Rank", "name", "parent_group", "staff_count",
    "required_workers", "vacancy_rate_pct", "adjusted_vacancy_rate",
    "current_salary_2025", "predicted_salary", "Demand_Score_1_10",
    "Current_Salary_Attractiveness_1_10", "Salary_Attractiveness_1_10",
    "Salary_Index_Change", "Salary_Growth_Pct", "Opportunity_Score_1_10",
    "projected_required_workers", "projected_vacancy_rate",
    "demand_scenario_factor", "scenario_note",
    "salary_base_year", "salary_source", "occupation_salary_date",
    "occupation_salary_observations",
]].rename(columns={
    "year": "Год_прогноза",
    "Forecast_Rank": "Место_в_прогнозе",
    "name": "Профессия",
    "parent_group": "Профессиональная_группа",
    "staff_count": "Занято_работников_2024",
    "required_workers": "Требуется_работников_2024",
    "projected_required_workers": "Сценарная_потребность_работников",
    "projected_vacancy_rate": "Сценарная_доля_вакансий",
    "demand_scenario_factor": "Коэффициент_сценария_спроса",
    "scenario_note": "Основание_сценария",
    "salary_base_year": "Год_базовой_зарплаты",
    "salary_source": "Источник_базовой_зарплаты",
    "occupation_salary_date": "Дата_среза_зарплаты",
    "occupation_salary_observations": "Наблюдений_с_зарплатой",
    "vacancy_rate_pct": "Исходная_доля_вакансий_2024_pct",
    "adjusted_vacancy_rate": "Сглаженная_доля_вакансий_2024",
    "current_salary_2025": "Базовая_зарплата_профессии_или_группы_руб",
    "predicted_salary": "Прогнозная_зарплата_профессии_или_группы_руб",
    "Demand_Score_1_10": "Балл_спроса_1_10",
    "Current_Salary_Attractiveness_1_10": "Актуальный_индекс_оплаты_1_10",
    "Salary_Attractiveness_1_10": "Прогнозный_индекс_оплаты_1_10",
    "Salary_Index_Change": "Изменение_индекса_оплаты",
    "Salary_Growth_Pct": "Рост_зарплаты_к_2025_pct",
    "Opportunity_Score_1_10": "Итоговый_ориентир_1_10",
})
occupation_forecast_export = occupation_forecast_export.drop(
    columns=["Изменение_индекса_оплаты"], errors="ignore"
)
occupation_forecast_export["Сглаженная_доля_вакансий_2024_pct"] = (
    occupation_forecast_export.pop("Сглаженная_доля_вакансий_2024") * 100
).round(2)
occupation_forecast_export["Сценарная_доля_вакансий_pct"] = (
    occupation_forecast_export.pop("Сценарная_доля_вакансий") * 100
).round(2)
occupation_forecast_export["Ограничение"] = (
    "Для профессий со специальным зарплатным срезом используется медиана "
    "предложений «Работы России»; иначе зарплата наследуется от группы. "
    "Результат не является обещанием числа вакансий"
)
occupation_forecast_export.to_csv(
    OCCUPATION_FORECAST_FILE, index=False, encoding="utf-8-sig"
)

professional_export = professional[[
    "shapeISO", "subject", "rank_in_region", "professional_group",
    "Professional_Job_Search_Difficulty_Score",
    "Professional_Job_Search_Difficulty_Score_Min",
    "Professional_Job_Search_Difficulty_Score_Max",
    "Professional_Rank_Min", "Professional_Rank_Max",
    "vacancy_rate_pct", "required_workers",
    "staff_count", "adjusted_vacancy_rate", "vacancies_per_10k_labor_force",
    "salary_adjusted_required_workers", "salary_adjusted_vacancy_rate",
    "estimated_professional_wage_2025", "estimated_professional_wage_buffer_2025",
    "national_average_wage_2025",
    "Professional_Salary_Attractiveness_Score", "salary_quality_weight",
    "median_wage_2025", "working_age_subsistence_2025",
    "staff_share_region", "vacancy_share_region", "competition_log_ratio",
    "credibility_weight", "prior_strength", "between_region_variance",
    "robust_rate_z", "observation_status", "structural_zero_market",
    "market_presence_to_region_median",
    "Regional_Job_Search_Difficulty_Score",
]].rename(columns={
    "shapeISO": "Код_ISO_субъекта",
    "subject": "Субъект_РФ",
    "rank_in_region": "Место_в_регионе",
    "professional_group": "Профессиональная_группа",
    "Professional_Job_Search_Difficulty_Score": "Сложность_поиска_работы_в_группе_1_10",
    "vacancy_rate_pct": "Потребность_в_общем_числе_рабочих_мест_pct",
    "Professional_Job_Search_Difficulty_Score_Min": "Индекс_при_других_настройках_min",
    "Professional_Job_Search_Difficulty_Score_Max": "Индекс_при_других_настройках_max",
    "Professional_Rank_Min": "Лучшее_место_при_других_настройках",
    "Professional_Rank_Max": "Худшее_место_при_других_настройках",
    "required_workers": "Требуется_работников_человек",
    "staff_count": "Списочная_численность_работников_человек",
    "adjusted_vacancy_rate": "Скорректированная_доля_вакансий",
    "salary_adjusted_required_workers": "Эквивалент_вакансий_с_учетом_оплаты",
    "salary_adjusted_vacancy_rate": "Доля_вакансий_с_учетом_оплаты",
    "estimated_professional_wage_2025": "Оценочная_зарплата_группы_2025_руб",
    "estimated_professional_wage_buffer_2025": "Оценочный_зарплатный_запас_группы_2025_руб",
    "national_average_wage_2025": "Средняя_зарплата_группы_по_РФ_2025_руб",
    "Professional_Salary_Attractiveness_Score": "Привлекательность_зарплаты_1_10",
    "salary_quality_weight": "Вес_качества_оплаты_0_1",
    "median_wage_2025": "Медианная_зарплата_региона_2025_руб",
    "working_age_subsistence_2025": "Прожиточный_минимум_трудоспособных_2025_руб",
    "vacancies_per_10k_labor_force": "Требуется_на_10_тыс_рабочей_силы",
    "staff_share_region": "Доля_группы_среди_занятых",
    "vacancy_share_region": "Доля_группы_среди_вакансий",
    "competition_log_ratio": "Лог_отношение_доли_занятых_к_доле_вакансий",
    "credibility_weight": "Вес_региональных_данных",
    "robust_rate_z": "Робастный_z_доли_вакансий",
    "prior_strength": "Kappa_метод_моментов",
    "between_region_variance": "Оценка_межрегиональной_дисперсии",
    "observation_status": "Статус_наблюдения",
    "structural_zero_market": "Структурный_ноль_локального_рынка",
    "market_presence_to_region_median": "Размер_группы_к_медианной_группе_региона",
    "Regional_Job_Search_Difficulty_Score": "Сложность_трудоустройства_в_регионе_1_10",
})
professional_export["Скорректированная_доля_вакансий_pct"] = (
    professional_export.pop("Скорректированная_доля_вакансий") * 100
).round(2)
professional_export["Доля_вакансий_с_учетом_оплаты_pct"] = (
    professional_export.pop("Доля_вакансий_с_учетом_оплаты") * 100
).round(2)
professional_export["Вес_региональных_данных_pct"] = (
    professional_export.pop("Вес_региональных_данных") * 100
).round(1)
professional_export["Требуется_на_10_тыс_рабочей_силы"] = professional_export[
    "Требуется_на_10_тыс_рабочей_силы"
].round(2)
professional_export["Робастный_z_доли_вакансий"] = professional_export[
    "Робастный_z_доли_вакансий"
].round(2)
for share_column in ["Доля_группы_среди_занятых", "Доля_группы_среди_вакансий"]:
    professional_export[share_column + "_pct"] = (
        professional_export.pop(share_column) * 100
    ).round(2)

regional_export = regional[[
    "shapeISO", "subject", "Regional_Job_Search_Difficulty_Score",
    "Regional_Tightness_Score_Min", "Regional_Tightness_Score_Max",
    "Regional_Rank_Min", "Regional_Rank_Max",
    "total_required_workers", "estimated_unemployed_people_2024",
    "vacancies_to_unemployed_ratio", "vacancies_per_10k_labor_force",
    "total_vacancy_rate_pct", "unemployment_rate_2024", "employment_rate_2024",
    "unemployment_rate_2024_annual", "broad_underutilization_rate_2024",
    "occupational_mismatch_tvd", "average_job_search_months_2024",
    "long_term_unemployment_share_2024", "official_tension_coefficient_2024",
    "broad_underutilization_signal", "vacancy_scarcity_signal",
    "occupational_mismatch_signal", "search_persistence_signal",
    "low_salary_signal", "median_wage_2025",
    "working_age_subsistence_2025", "median_wage_to_subsistence_ratio_2025",
    "median_wage_buffer_2025",
    "Salary_Attractiveness_Score",
    "labor_force_thousand_2024", "wage_growth_2024_pct",
    "wage_growth_june_2026_yoy_pct", "denominator_stability", "coverage_note",
]].rename(columns={
    "shapeISO": "Код_ISO_субъекта",
    "subject": "Субъект_РФ",
    "Regional_Job_Search_Difficulty_Score": "Индекс_сложности_трудоустройства_1_10",
    "Regional_Tightness_Score_Min": "Индекс_при_других_весах_min",
    "Regional_Tightness_Score_Max": "Индекс_при_других_весах_max",
    "Regional_Rank_Min": "Лучшее_место_при_других_весах",
    "Regional_Rank_Max": "Худшее_место_при_других_весах",
    "total_required_workers": "Требуется_работников_человек",
    "estimated_unemployed_people_2024": "Оценка_безработных_человек",
    "vacancies_to_unemployed_ratio": "Вакансий_на_одного_безработного_V_U",
    "vacancies_per_10k_labor_force": "Вакансий_на_10_тыс_рабочей_силы",
    "total_vacancy_rate_pct": "Доля_вакантных_мест_в_обследованных_организациях_pct",
    "unemployment_rate_2024": "Уровень_безработицы_окт_дек_2024_pct",
    "unemployment_rate_2024_annual": "Уровень_безработицы_2024_pct",
    "broad_underutilization_rate_2024": "Безработица_плюс_потенциальная_рабочая_сила_2024_pct",
    "occupational_mismatch_tvd": "Структурное_несовпадение_профгрупп_0_1",
    "average_job_search_months_2024": "Среднее_время_поиска_работы_месяцев",
    "long_term_unemployment_share_2024": "Ищут_работу_12_месяцев_и_более_pct",
    "official_tension_coefficient_2024": "Официальный_коэффициент_напряженности_2024",
    "broad_underutilization_signal": "Сигнал_недоиспользования_труда_0_1",
    "vacancy_scarcity_signal": "Сигнал_нехватки_вакансий_0_1",
    "occupational_mismatch_signal": "Сигнал_структурного_несовпадения_0_1",
    "search_persistence_signal": "Сигнал_длительности_поиска_0_1",
    "low_salary_signal": "Сигнал_низкой_оплаты_0_1",
    "median_wage_2025": "Медианная_зарплата_апрель_2025_руб",
    "working_age_subsistence_2025": "Прожиточный_минимум_трудоспособных_2025_руб",
    "median_wage_to_subsistence_ratio_2025": "Медианная_зарплата_к_прожиточному_минимуму",
    "median_wage_buffer_2025": "Зарплатный_запас_после_прожиточного_минимума_руб",
    "Salary_Attractiveness_Score": "Привлекательность_зарплаты_1_10",
    "employment_rate_2024": "Уровень_занятости_окт_дек_2024_pct",
    "labor_force_thousand_2024": "Рабочая_сила_окт_дек_2024_тыс",
    "wage_growth_2024_pct": "Рост_средней_зарплаты_2024_к_2023_pct",
    "wage_growth_june_2026_yoy_pct": "Рост_зарплаты_июнь_2026_г_г_pct",
    "denominator_stability": "Осторожность_при_интерпретации_безработицы",
    "coverage_note": "Ограничение_охвата",
})
regional_export["Внешняя_проверка_Спирмен_rho"] = round(validation_spearman, 3) if pd.notna(validation_spearman) else np.nan
regional_export["Максимальная_корреляция_компонентов_Спирмен_rho"] = round(component_spearman, 3)
regional_export["Версия_формулы"] = (
    "4.1: 34% широкое недоиспользование труда + 30% нехватка вакансий + "
    "13% профессиональное несовпадение + 8% длительность поиска + "
    "15% малый абсолютный запас медианной зарплаты после прожиточного минимума"
)
regional_export["Интерпретация_внешней_проверки"] = validation_summary
regional_export.to_csv(REGIONAL_REPORT_FILE, index=False, encoding="utf-8-sig")

regional_rank_span = (
    regional["Regional_Rank_Max"] - regional["Regional_Rank_Min"]
).dropna().astype(float)
professional_rank_span = (
    professional["Professional_Rank_Max"] - professional["Professional_Rank_Min"]
).dropna().astype(float)
population_score_spearman = float(
    regional[["Regional_Job_Search_Difficulty_Score", "population_people"]]
    .corr(method="spearman").iloc[0, 1]
)
labor_force_score_spearman = float(
    regional[["Regional_Job_Search_Difficulty_Score", "labor_force_thousand_2024"]]
    .corr(method="spearman").iloc[0, 1]
)
professional_size_score_spearman = float(
    professional[["Professional_Job_Search_Difficulty_Score", "staff_count"]]
    .corr(method="spearman").iloc[0, 1]
)
salary_score_spearman = float(
    regional[["Salary_Attractiveness_Score", "median_wage_buffer_2025"]]
    .corr(method="spearman").iloc[0, 1]
)
structural_zero_count = int(professional["structural_zero_market"].sum())
zero_markets_with_score = int(
    professional.loc[
        professional["structural_zero_market"],
        "Professional_Job_Search_Difficulty_Score",
    ].notna().sum()
)
salary_adjusted_vacancies_above_raw = int(
    (
        professional["salary_adjusted_required_workers"]
        > professional["required_workers"] + 1e-9
    ).sum()
)
small_markets_with_demand_without_score = int(
    professional.loc[
        professional["small_market_with_demand"],
        "Professional_Job_Search_Difficulty_Score",
    ].isna().sum()
)
moscow_score = float(
    regional.loc[
        regional["subject"].eq("Москва"),
        "Regional_Job_Search_Difficulty_Score",
    ].iloc[0]
)
sakha_score = float(
    regional.loc[
        regional["subject"].eq("Республика Саха (Якутия)"),
        "Regional_Job_Search_Difficulty_Score",
    ].iloc[0]
)
audit_rows = [
    ("Версия формулы", "4.1", "Сложность поиска приемлемо оплачиваемой работы"),
    ("Периоды", "пройдено", f"Вакансии {PROFESSIONAL_REFERENCE_DATE}; рынок труда — {LABOR_MARKET_REFERENCE_PERIOD}; зарплата и прожиточный минимум — 2025"),
    ("Субъекты с рассчитанным индексом", int(valid_scores.notna().sum()), "из 84 строк регионального справочника"),
    ("Максимальная ошибка пересчёта доли вакансий, п.п.", round(float(rate_difference.max()), 4), "допуск 0,02 п.п."),
    ("Индексы вне шкалы 1–10", int((~valid_scores.between(1, 10)).sum()), "должно быть 0"),
    ("Максимальная абсолютная корреляция компонентов, Спирмен", round(component_spearman, 3), "контроль двойного счёта; чем дальше от 1, тем лучше"),
    ("Согласование с официальным коэффициентом напряжённости", round(validation_spearman, 3), "положительная конвергентная проверка Росстат/Роструд"),
    ("Старая работодательская формула против напряжённости", round(legacy_validation_spearman, 3), "контроль смены экономического направления"),
    ("Формула без длительности против среднего времени поиска", round(search_duration_validation_spearman, 3), "честная leave-one-component-out проверка"),
    ("Связь с ростом зарплаты 2024", round(wage_validation_spearman, 3), "справочная, не критерий качества индекса соискателя"),
    ("Связь индекса с численностью населения", round(population_score_spearman, 3), "контроль автоматического преимущества крупных регионов"),
    ("Связь индекса с размером рабочей силы", round(labor_force_score_spearman, 3), "контроль масштабного смещения"),
    ("Связь балла профгруппы с числом занятых", round(professional_size_score_spearman, 3), "крупные группы не должны получать балл только за размер"),
    ("Проверка зарплатной привлекательности", round(salary_score_spearman, 3), "должна монотонно расти вместе с абсолютным зарплатным запасом после регионального минимума"),
    ("Структурные нули локального рынка", structural_zero_count, "нулевой спрос в практически отсутствующей группе не получает ложный максимальный балл"),
    ("Структурные нули с присвоенным баллом", zero_markets_with_score, "должно быть 0"),
    ("Оплаченные эквиваленты выше исходного числа вакансий", salary_adjusted_vacancies_above_raw, "должно быть 0: высокая зарплата не размножает вакансии"),
    ("Малые рынки со спросом без балла", small_markets_with_demand_without_score, "должно быть 0: реально наблюдаемый спрос сохраняется"),
    ("Москва: сложность трудоустройства", round(moscow_score, 2), "ожидаемо ниже среднего без ручной поправки"),
    ("Республика Саха (Якутия): сложность трудоустройства", round(sakha_score, 2), "широкое недоиспользование труда и нехватка вакансий повышают балл"),
    ("Медианный диапазон места региона в сценариях весов", round(float(regional_rank_span.median()), 1), "чем меньше, тем устойчивее место"),
    ("90-й процентиль диапазона места региона", round(float(regional_rank_span.quantile(0.9)), 1), "показывает чувствительные регионы"),
    ("Максимальный диапазон места региона", round(float(regional_rank_span.max()), 1), "показывается пользователю рядом с баллом"),
    ("Медианный диапазон места профгруппы", round(float(professional_rank_span.median()), 1), "сценарии весов и κ ×0,5/×2"),
    ("Максимальный диапазон места профгруппы", round(float(professional_rank_span.max()), 1), "сценарии весов и κ ×0,5/×2"),
    ("Месяцев в диагностическом Ridge-backtest", wage_forecast_months, "историческая проверка; дальний прогноз зарплаты Ridge больше не используется"),
    ("Диагностический Ridge MAPE, %", round(wage_forecast_mape * 100, 2), "ошибка на последних 12 известных месяцах; не источник сценария 2027–2032"),
    ("Диагностический Ridge alpha", round(wage_ridge_alpha, 6), "сохранён для аудита прежней модели"),
    ("Участников макроэкономического опроса ЦБ", inflation_model_meta["respondents"], "июль 2026; итог — медиана прогнозов экономистов"),
    ("Прогноз среднегодового ИПЦ на 2026–2029, %", "6,0; 4,9; 4,1; 4,0", "июльский макроэкономический опрос Банка России"),
    ("Прогноз роста номинальной зарплаты на 2026–2029, %", "10,2; 8,0; 7,0; 6,5", "не медианная зарплата, а рост номинальной зарплаты в среднем за год"),
    ("Временных точек профессионального спроса", 1, "поэтому спрос 2027–2032 — стресс-сценарий вокруг структуры 31.10.2024, а не временная экстраполяция"),
    ("Вес сценарного спроса в будущем ориентире, %", 65, "зарплатная привлекательность получает 35%; текущий индекс карты не меняется"),
]
pd.DataFrame(audit_rows, columns=["Тест", "Результат", "Интерпретация"]).to_csv(
    INDEX_AUDIT_FILE, index=False, encoding="utf-8-sig"
)

source_rows = [
    {
        "Файл": GEOJSON_FILE.name,
        "Официальный_URL": "https://github.com/prokopenkoad42-creator/karta-rossii",
        "Период": "единый топологический слой; 85 контуров базового состава",
        "Назначение": "Границы субъектов для карты, включая Крым и Севастополь; не статистический источник",
    },
    {
        "Файл": PROFESSIONAL_DEMAND_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/compendium/document/13266",
        "Период": PROFESSIONAL_REFERENCE_DATE,
        "Назначение": "Вакансии и занятые по профессиональным группам, форма 1-Т(проф)",
    },
    {
        "Файл": REGION_INDICATORS_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/storage/mediabank/Region_Pokaz_2025.pdf",
        "Период": LABOR_MARKET_REFERENCE_PERIOD,
        "Назначение": (
            "Региональная безработица, потенциальная рабочая сила, "
            "длительность поиска и официальный коэффициент напряжённости"
        ),
    },
    {
        "Файл": SUBSISTENCE_MINIMUM_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/storage/mediabank/nb_pm_4-1.xlsx",
        "Период": "2025",
        "Назначение": "Региональный прожиточный минимум трудоспособного населения для поправки зарплаты",
    },
    {
        "Файл": PROFESSIONAL_WAGE_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/storage/mediabank/sr-zpl6_2025.xlsx",
        "Период": "октябрь 2025",
        "Назначение": "Средняя зарплата по девяти профессиональным группам для относительного зарплатного профиля",
    },
    {
        "Файл": DETAILED_WAGE_ARCHIVE_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/storage/mediabank/sved_57-t_2025.rar",
        "Период": "октябрь 2025",
        "Назначение": "Полный бюллетень 57-Т с зарплатами по составным профессиональным группам ОКЗ",
    },
    {
        "Файл": UNEMPLOYMENT_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/labour_force",
        "Период": INDEX_PERIOD_LABEL,
        "Назначение": "Безработица по обследованию рабочей силы",
    },
    {
        "Файл": LABOR_FORCE_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/labour_force",
        "Период": INDEX_PERIOD_LABEL,
        "Назначение": "Численность рабочей силы",
    },
    {
        "Файл": EMPLOYMENT_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/labour_force",
        "Период": INDEX_PERIOD_LABEL,
        "Назначение": "Уровень занятости для поясняющего контекста",
    },
    {
        "Файл": P4_METHOD_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/storage/mediabank/met_519-22092025.pdf",
        "Период": "Методология, приказ Росстата № 519 от 22.09.2025",
        "Назначение": "Проверка определения вакансий и ограничения охвата малого бизнеса",
    },
    {
        "Файл": MSP_METHOD_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/storage/mediabank/XIAbF4QQ/MET010018.pdf",
        "Период": "Методология 2026",
        "Назначение": "Контроль определения занятых в секторе МСП; коэффициент не подставляется в индекс",
    },
    {
        "Файл": WAGE_VALIDATION_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/storage/mediabank/tab2-zpl_06-2026.xlsx",
        "Период": "2013 — июнь 2026; обновлено 02.09.2026",
        "Назначение": "Независимая проверка индекса по динамике зарплаты",
    },
    {
        "Файл": CURRENT_LABOR_FORCE_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/storage/mediabank/trud-1_15-s.xlsx",
        "Период": "март–май 2026; обновлено 15.07.2026",
        "Назначение": "Актуальный контекст численности рабочей силы 15 лет и старше",
    },
    {
        "Файл": CURRENT_UNEMPLOYMENT_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/storage/mediabank/trud_3_15-s.xlsx",
        "Период": "март–май 2026; обновлено 15.07.2026",
        "Назначение": "Актуальный контекст безработицы населения 15 лет и старше",
    },
    {
        "Файл": MEDIAN_WAGE_HISTORY_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/storage/mediabank/tab9-zpl_2025.xlsx",
        "Период": "2019–2025",
        "Назначение": "Фактическая годовая медианная начисленная зарплата по России",
    },
    {
        "Файл": ROSSTAT_CPI_HISTORY_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/storage/mediabank/ipc_s_1992-2025.xlsx",
        "Период": "1992–2025; в модели используются 2000–2025",
        "Назначение": "Официальный фактический годовой ИПЦ для исторической части графика",
    },
    {
        "Файл": ROSSTAT_CPI_MONTHLY_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/storage/mediabank/ipc_mes_08-2026.xlsx",
        "Период": "январь–август 2026; обновлено 11.09.2026",
        "Назначение": "Контроль текущей инфляции 2026 года: ИПЦ к декабрю 2025 года",
    },
    {
        "Файл": ROSSTAT_CPI_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/storage/mediabank/3_16-01-2026.html",
        "Период": "декабрь 2025; последний полный официальный год — 2025",
        "Назначение": "Контроль последнего фактического ИПЦ: декабрь к декабрю",
    },
    {
        "Файл": CBR_INFLATION_FORECAST_FILE.name,
        "Официальный_URL": "https://www.cbr.ru/press/pr/?file=24072026_133000key.htm",
        "Период": "прогноз от 24.07.2026",
        "Назначение": "Прогноз инфляции 6–7% в 2026 году и возврат к цели 4% с 2027 года",
    },
    {
        "Файл": CBR_MEDIUM_TERM_FORECAST_FILE.name,
        "Официальный_URL": "https://www.cbr.ru/Content/Document/File/194258/forecast_260724.pdf",
        "Период": "среднесрочный прогноз от 24.07.2026; горизонт 2026–2029",
        "Назначение": "Базовый прогноз самого Банка России; справочный источник, не ряд совмещённого графика",
    },
    {
        "Файл": CBR_MACRO_SURVEY_FILE.name,
        "Официальный_URL": "https://www.cbr.ru/Content/Document/File/144490/full.xlsx",
        "Период": "опрос 10–14.07.2026; опубликовано 15.07.2026; горизонт 2026–2029",
        "Назначение": "Медианы и диапазоны прогнозов инфляции и роста номинальной зарплаты",
    },
    {
        "Файл": CBR_MACRO_SURVEY_PDF_FILE.name,
        "Официальный_URL": "https://www.cbr.ru/Content/Document/File/193892/Survey_pdf.pdf",
        "Период": "июль 2026; факты 2021–2025 и прогноз 2026–2029",
        "Назначение": "Официальная презентация с таблицами и готовыми графиками макроопроса",
    },
    {
        "Файл": SEVASTOPOL_LABOR_2024_FILE.name,
        "Официальный_URL": "https://82.rosstat.gov.ru/folder/27489",
        "Период": "октябрь–декабрь 2024",
        "Назначение": "Сопоставимые показатели рабочей силы Севастополя для индекса",
    },
    {
        "Файл": SEVASTOPOL_LABOR_2026_FILE.name,
        "Официальный_URL": "https://82.rosstat.gov.ru/folder/27489",
        "Период": "апрель–июнь 2026; обновлено 17.08.2026",
        "Назначение": "Последний оперативный контекст рабочей силы Севастополя",
    },
    {
        "Файл": SEVASTOPOL_YEARBOOK_FILE.name,
        "Официальный_URL": "https://82.rosstat.gov.ru/storage/mediabank/1-Севастополь в цифрах 2025.pdf",
        "Период": "2024–2025; опубликовано в 2026 году",
        "Назначение": "Население, прожиточный минимум и контроль показателей Севастополя",
    },
    {
        "Файл": ROSSTAT_LABOUR_FORCE_2026_FILE.name,
        "Официальный_URL": "https://rosstat.gov.ru/folder/210/document/13211",
        "Период": "годовые данные по 2025 год; опубликовано 22.07.2026",
        "Назначение": "Последний годовой сборник о рабочей силе, занятости и безработице",
    },
]
for source in source_rows:
    source_path = PROJECT_DIR / source["Файл"]
    if not source_path.exists():
        source_path = PROJECT_DIR / "official_sources" / source["Файл"]
    source["SHA256"] = file_sha256(source_path)
source_manifest = pd.DataFrame(source_rows)
source_manifest["Дата_формирования_реестра"] = "2026-09-12"
source_manifest.to_csv(SOURCE_MANIFEST_FILE, index=False, encoding="utf-8-sig")
professional_export.to_csv(PROFESSIONAL_REPORT_FILE, index=False, encoding="utf-8-sig")

with GEOJSON_FILE.open("r", encoding="utf-8") as file:
    geojson = json.load(file)

ranking_by_iso = {
    shape_iso: group.sort_values("rank_in_region").to_dict(orient="records")
    for shape_iso, group in professional.groupby("shapeISO")
}

regional_for_map = regional.dropna(subset=[
    "Regional_Job_Search_Difficulty_Score", "estimated_unemployed_people_2024",
    "median_wage_2025", "working_age_subsistence_2025",
]).copy()
map_data = {}
for row in regional_for_map.itertuples(index=False):
    ranking = []
    for item in ranking_by_iso.get(row.shapeISO, []):
        observed_score = item["Professional_Job_Search_Difficulty_Score"]
        ranking.append({
            "name": item["professional_group"],
            "rank": int(item["rank_in_region"]) if pd.notna(item["rank_in_region"]) else None,
            "score": round(float(observed_score), 2) if pd.notna(observed_score) else None,
            "score_min": round(float(item["Professional_Job_Search_Difficulty_Score_Min"]), 2) if pd.notna(item["Professional_Job_Search_Difficulty_Score_Min"]) else None,
            "score_max": round(float(item["Professional_Job_Search_Difficulty_Score_Max"]), 2) if pd.notna(item["Professional_Job_Search_Difficulty_Score_Max"]) else None,
            "rank_min": int(item["Professional_Rank_Min"]) if pd.notna(item["Professional_Rank_Min"]) else None,
            "rank_max": int(item["Professional_Rank_Max"]) if pd.notna(item["Professional_Rank_Max"]) else None,
            "kappa_mom": round(float(item["prior_strength"]), 2),
            "between_region_variance": round(float(item["between_region_variance"]), 8),
            "vacancy_rate_pct": round(float(item["vacancy_rate_pct"]), 2),
            "adjusted_vacancy_rate_pct": round(float(item["adjusted_vacancy_rate"]) * 100, 2),
            "salary_adjusted_vacancy_rate_pct": round(float(item["salary_adjusted_vacancy_rate"]) * 100, 2),
            "salary_adjusted_required_workers": round(float(item["salary_adjusted_required_workers"]), 1),
            "estimated_professional_wage": round(float(item["estimated_professional_wage_2025"])),
            "estimated_professional_wage_buffer": round(float(item["estimated_professional_wage_buffer_2025"])),
            "salary_attractiveness_score": round(float(item["Professional_Salary_Attractiveness_Score"]), 2),
            "salary_quality_weight_pct": round(float(item["salary_quality_weight"]) * 100, 1),
            "vacancies_per_10k": round(float(item["vacancies_per_10k_labor_force"]), 2),
            "staff_share_pct": round(float(item["staff_share_region"]) * 100, 2),
            "vacancy_share_pct": round(float(item["vacancy_share_region"]) * 100, 2),
            "credibility_weight_pct": round(float(item["credibility_weight"]) * 100, 1),
            "observation_status": item["observation_status"],
            "structural_zero_market": bool(item["structural_zero_market"]),
            "required_workers": int(round(float(item["required_workers"]))) if pd.notna(item["required_workers"]) else None,
            "staff_count": int(round(float(item["staff_count"]))) if pd.notna(item["staff_count"]) else None,
        })
    map_data[row.shapeISO] = {
        "region": row.subject,
        "region_code": str(row.region_code).zfill(2),
        "score_min": round(float(row.Regional_Tightness_Score_Min), 2),
        "score_max": round(float(row.Regional_Tightness_Score_Max), 2),
        "rank_min": int(row.Regional_Rank_Min),
        "rank_max": int(row.Regional_Rank_Max),
        "index_period": LABOR_MARKET_REFERENCE_PERIOD,
        "unemployment_rate_index": round(float(row.unemployment_rate_2024), 1),
        "unemployment_rate_annual": round(float(row.unemployment_rate_2024_annual), 1),
        "broad_underutilization_rate": round(float(row.broad_underutilization_rate_2024), 1),
        "average_job_search_months": round(float(row.average_job_search_months_2024), 1) if pd.notna(row.average_job_search_months_2024) else None,
        "long_term_unemployment_share": round(float(row.long_term_unemployment_share_2024), 1) if pd.notna(row.long_term_unemployment_share_2024) else None,
        "occupational_mismatch_pct": round(float(row.occupational_mismatch_tvd) * 100, 1) if pd.notna(row.occupational_mismatch_tvd) else None,
        "official_tension_coefficient": round(float(row.official_tension_coefficient_2024), 1),
        "employment_rate_index": round(float(row.employment_rate_2024), 1),
        "labor_force_thousand_index": round(float(row.labor_force_thousand_2024), 1),
        "current_labor_period": row.period,
        "labor_force_thousand_current": round(float(row.labor_force_thousand_current), 1) if pd.notna(row.labor_force_thousand_current) else None,
        "unemployment_rate_current": round(float(row.unemployment_rate_current), 1) if pd.notna(row.unemployment_rate_current) else None,
        "estimated_unemployed_current": int(round(float(row.estimated_unemployed_current))) if pd.notna(row.estimated_unemployed_current) else None,
        "estimated_unemployed_people": int(round(float(row.estimated_unemployed_people_2024))),
        "required_workers_total": int(round(float(row.total_required_workers))) if pd.notna(row.total_required_workers) else None,
        "vacancies_to_unemployed_ratio": round(float(row.vacancies_to_unemployed_ratio), 3) if pd.notna(row.vacancies_to_unemployed_ratio) else None,
        "vacancies_per_10k_labor_force": round(float(row.vacancies_per_10k_labor_force), 1) if pd.notna(row.vacancies_per_10k_labor_force) else None,
        "wage_growth_2024_pct": round(float(row.wage_growth_2024_pct), 1) if pd.notna(row.wage_growth_2024_pct) else None,
        "wage_growth_june_2026_yoy_pct": round(float(row.wage_growth_june_2026_yoy_pct), 1) if pd.notna(row.wage_growth_june_2026_yoy_pct) else None,
        "median_wage_2025": int(round(float(row.median_wage_2025))),
        "working_age_subsistence_2025": int(round(float(row.working_age_subsistence_2025))),
        "median_wage_to_subsistence_ratio": round(float(row.median_wage_to_subsistence_ratio_2025), 2),
        "median_wage_buffer": int(round(float(row.median_wage_buffer_2025))),
        "salary_attractiveness_score": round(float(row.Salary_Attractiveness_Score), 2),
        "denominator_stability": row.denominator_stability,
        "coverage_note": row.coverage_note,
        "validation_summary": validation_summary,
        "validation_spearman": round(validation_spearman, 3) if pd.notna(validation_spearman) else None,
        "method_note": "34% — люди без работы и потенциальная рабочая сила; 30% — нехватка вакансий; 13% — несовпадение структуры профессий; 8% — длительность поиска; 15% — малый зарплатный запас после вычитания прожиточного минимума.",
        "area_thousand_km2": round(float(row.area_thousand_km2), 1),
        "population_people": int(round(float(row.population_people))),
        "score": round(float(row.Regional_Job_Search_Difficulty_Score), 2),
        "interpretation": row.interpretation,
        "unemployment_rate": round(float(row.unemployment_rate), 1),
        "employment_rate": round(float(row.employment_rate), 1),
        "labor_force_thousand": round(float(row.labor_force_thousand), 1),
        "participation_rate": round(float(row.labor_force_participation_rate), 1),
        "qualified_share": round(float(row.qualified_labor_force_share), 1) if pd.notna(row.qualified_labor_force_share) else None,
        "overall_vacancy_rate_pct": (
            round(float(row.total_vacancy_rate_pct), 2)
            if pd.notna(row.total_vacancy_rate_pct) else None
        ),
        "ranking": ranking,
    }


# Субъекты без полного сопоставимого набора остаются интерактивными и серыми.
no_data_regions = {}
reference_lookup = region_reference.set_index("subject")
for feature in geojson["features"]:
    shape_iso = feature["properties"].get("shapeISO")
    if shape_iso in map_data:
        continue
    subject = feature["properties"].get("shapeName", "Субъект без названия")
    if subject in reference_lookup.index:
        reference_row = reference_lookup.loc[subject]
        region_code = str(reference_row["region_code"]).zfill(2)
        area_value = reference_row["area_thousand_km2"]
        population_value = reference_row.get("population_people")
        area = float(area_value) if pd.notna(area_value) else None
        population_people = int(round(float(population_value))) if pd.notna(population_value) else None
    else:
        region_code = "91" if subject == "Республика Крым" else "—"
        area = 26.1 if subject == "Республика Крым" else None
        population_people = None
    no_data_regions[shape_iso] = {
        "region": subject,
        "region_code": region_code,
        "area_thousand_km2": area,
        "population_people": population_people,
        "score": None,
        "has_data": False,
        "message": (
            "Росстат пока не опубликовал полный сопоставимый набор показателей, "
            "необходимый для расчёта индекса этого субъекта. Регион сохранён на "
            "карте и в списке, но искусственный балл ему не присваивается."
        ),
    }

occupation_catalog = []
for item in detailed_occupations.itertuples(index=False):
    occupation_catalog.append({
        "name": item.name,
        "parent_group": item.parent_group,
        "national_score": round(float(item.national_score), 2),
        "vacancy_rate_pct": round(float(item.vacancy_rate_pct), 2),
        "adjusted_vacancy_rate_pct": round(float(item.adjusted_vacancy_rate) * 100, 2),
        "credibility_weight_pct": round(float(item.credibility_weight) * 100, 1),
        "required_workers": int(round(float(item.required_workers))),
        "staff_count": (
            int(round(float(item.staff_count))) if pd.notna(item.staff_count) else None
        ),
    })

forecast_by_year = {str(year): [] for year in FORECAST_YEARS}
for item in occupation_forecast.itertuples(index=False):
    forecast_by_year[str(int(item.year))].append({
        "rank": int(item.Forecast_Rank),
        "name": item.name,
        "parent_group": item.parent_group,
        "demand_score": round(float(item.Demand_Score_1_10), 2),
        "current_salary_score": round(float(item.Current_Salary_Attractiveness_1_10), 2),
        "salary_score": round(float(item.Salary_Attractiveness_1_10), 2),
        "salary_growth_pct": round(float(item.Salary_Growth_Pct), 1),
        "opportunity_score": round(float(item.Opportunity_Score_1_10), 2),
        "current_salary_2025": int(round(float(item.current_salary_2025))),
        "predicted_salary": int(round(float(item.predicted_salary))),
        "salary_base_year": int(item.salary_base_year),
        "salary_source": item.salary_source,
        "salary_observations": (
            int(item.occupation_salary_observations)
            if pd.notna(item.occupation_salary_observations) else None
        ),
        "required_workers_2024": int(round(float(item.required_workers))),
        "adjusted_vacancy_rate_pct": round(float(item.adjusted_vacancy_rate) * 100, 2),
        "projected_required_workers": int(round(float(item.projected_required_workers))),
        "projected_vacancy_rate_pct": round(float(item.projected_vacancy_rate) * 100, 2),
        "scenario_factor": round(float(item.demand_scenario_factor), 3),
        "scenario_note": item.scenario_note,
    })

group_forecast_by_year = {str(year): [] for year in FORECAST_YEARS}
for item in group_forecast.itertuples(index=False):
    group_forecast_by_year[str(int(item.year))].append({
        "rank": int(item.Forecast_Rank),
        "name": item.professional_group,
        "demand_score": round(float(item.Demand_Score_1_10), 2),
        "current_salary_score": round(float(item.Current_Salary_Attractiveness_1_10), 2),
        "salary_score": round(float(item.Salary_Attractiveness_1_10), 2),
        "salary_growth_pct": round(float(item.Salary_Growth_Pct), 1),
        "opportunity_score": round(float(item.Opportunity_Score_1_10), 2),
        "current_salary_2025": int(round(float(item.current_salary_2025))),
        "predicted_salary": int(round(float(item.predicted_salary))),
        "required_workers_2024": int(round(float(item.required_workers_2024))),
        "vacancy_rate_pct": round(float(item.baseline_vacancy_rate_2024) * 100, 2),
        "projected_required_workers": int(round(float(item.projected_required_workers))),
        "projected_vacancy_rate_pct": round(float(item.projected_vacancy_rate) * 100, 2),
        "scenario_factor": round(float(item.demand_scenario_factor), 3),
        "scenario_note": item.scenario_note,
    })

city_job_explanations = {}
for city in ("Москва", "Санкт-Петербург"):
    city_rows = professional.loc[
        professional["subject"].eq(city)
        & professional["Professional_Job_Search_Difficulty_Score"].notna()
    ].nlargest(5, "Professional_Job_Search_Difficulty_Score")
    details = []
    for item in city_rows.itertuples(index=False):
        example_names = (
            detailed_occupations.loc[
                detailed_occupations["parent_group"].eq(item.professional_group)
            ]
            .nlargest(3, "national_score")["name"]
            .tolist()
        )
        details.append({
            "name": item.professional_group,
            "score": round(float(item.Professional_Job_Search_Difficulty_Score), 2),
            "required_workers": int(round(float(item.required_workers))),
            "staff_count": int(round(float(item.staff_count))),
            "adjusted_vacancy_rate_pct": round(float(item.adjusted_vacancy_rate) * 100, 2),
            "salary_adjusted_vacancy_rate_pct": round(float(item.salary_adjusted_vacancy_rate) * 100, 2),
            "estimated_wage": int(round(float(item.estimated_professional_wage_2025))),
            "examples": example_names,
        })
    regional_city = regional.loc[regional["subject"].eq(city)].iloc[0]
    city_job_explanations[city] = {
        "regional_score": round(float(regional_city.Regional_Job_Search_Difficulty_Score), 2),
        "broad_underutilization_rate": round(float(regional_city.broad_underutilization_rate_2024), 1),
        "mismatch_pct": round(float(regional_city.occupational_mismatch_tvd) * 100, 1),
        "salary_buffer": int(round(float(regional_city.median_wage_buffer_2025))),
        "rows": details,
    }


html_template = r'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Карта сложности поиска приемлемо оплачиваемой работы</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<style>
:root{--bg:#f4f7fb;--panel:#fff;--ink:#14213d;--muted:#667085;--line:#d7dee9;--blue:#1769d2}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Segoe UI,Arial,sans-serif}
.shell{max-width:1450px;margin:0 auto;padding:18px}.title{font-size:24px;font-weight:750;margin:0 0 5px}.subtitle{color:var(--muted);font-size:14px;margin-bottom:13px}
.method{display:grid;grid-template-columns:350px minmax(0,1fr);gap:18px;align-items:center;height:132px;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:13px 20px;margin-bottom:14px;overflow:hidden}.method>div{min-width:0}
.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:700}.method-region{display:-webkit-box;min-height:44px;max-height:44px;overflow:hidden;-webkit-box-orient:vertical;-webkit-line-clamp:2;font-size:16px;font-weight:750;line-height:1.35;margin-top:4px}.method-title{display:-webkit-box;overflow:hidden;-webkit-box-orient:vertical;-webkit-line-clamp:2;font-size:13px;font-weight:750;line-height:1.35}.method-text{display:-webkit-box;overflow:hidden;-webkit-box-orient:vertical;-webkit-line-clamp:2;font-size:12px;line-height:1.45;margin-top:5px;color:var(--muted)}
.layout{display:grid;grid-template-columns:minmax(0,2.35fr) minmax(330px,.9fr);gap:14px;align-items:start}.map-card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;box-shadow:0 8px 28px rgba(27,42,78,.08)}
.map-card{height:620px;overflow:hidden;position:relative}.map-card svg{width:100%;height:620px;display:block}.panel{padding:18px;height:620px;min-height:620px;overflow-y:auto;scrollbar-gutter:stable}.region-name{font-size:22px;font-weight:750;margin:5px 0 8px}
.overall{display:flex;align-items:center;justify-content:space-between;padding:12px;border-radius:12px;background:#f7f8fb;margin-bottom:12px}.overall strong{font-size:25px}.official-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-bottom:13px}.official-grid>div{padding:8px 10px;background:#f7f8fb;border-radius:10px}.official-grid span{display:block;font-size:10px;color:var(--muted)}.official-grid strong{display:block;font-size:14px;margin-top:2px}
.rank-head{display:grid;grid-template-columns:1fr;gap:9px;margin:13px 0 7px}.rank-title{font-size:14px;font-weight:750;margin:0}.rank-search-label{display:block;width:min(210px,100%);justify-self:end;font-size:9px;color:var(--muted)}.rank-search{width:100%;height:32px;margin-top:3px;padding:6px 9px;border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--ink);font:inherit;font-size:11px;outline:none}.rank-search:focus{border-color:#62a5ed;box-shadow:0 0 0 3px rgba(23,105,210,.12)}.rank-subtitle{font-size:11px;color:var(--muted);line-height:1.35;margin-bottom:7px}.rank-row{display:grid;grid-template-columns:22px minmax(0,1fr) 46px;gap:8px;align-items:center;padding:8px 3px;border-bottom:1px solid #edf0f5}.rank-num{color:var(--muted);font-weight:700}.rank-name{appearance:none;border:0;background:none;padding:0;color:var(--ink);font:inherit;font-size:12px;line-height:1.25;text-align:left;cursor:pointer}.rank-name:hover,.rank-name:focus-visible{color:var(--blue);text-decoration:underline;text-underline-offset:3px}.rank-meta{font-size:10px;color:var(--muted);margin-top:3px}.badge{padding:5px 3px;border-radius:8px;color:white;text-align:center;font-weight:800;font-variant-numeric:tabular-nums}.rank-empty{padding:12px 4px;color:var(--muted);font-size:11px;border-bottom:1px solid #edf0f5}
.group-detail{margin:0 3px 8px 30px;padding:9px 10px;border-left:3px solid #7ebaf4;background:#f5f9ff;font-size:11px;line-height:1.45}.group-detail strong{font-size:12px}.more-wrap{text-align:right;margin-top:8px}.more{appearance:none;border:0;background:none;color:var(--blue);font:inherit;font-size:11px;cursor:pointer;text-decoration:underline;text-underline-offset:3px}
.legend{margin-top:15px}.gradient-scale{height:28px;position:relative}.gradient{position:absolute;left:0;right:0;top:8px;height:12px;border-radius:99px;background:linear-gradient(90deg,#238b45 0%,#86c440 28%,#ffd54f 52%,#ef6c00 74%,#7f001f 100%)}.gradient-slider{position:absolute;inset:0;width:100%;height:28px;margin:0;appearance:none;-webkit-appearance:none;background:transparent;cursor:pointer;z-index:3}.gradient-slider::-webkit-slider-runnable-track{height:12px;background:transparent}.gradient-slider::-moz-range-track{height:12px;background:transparent}.gradient-slider::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;margin-top:-3px;border:3px solid #fff;border-radius:50%;background:#1769d2;box-shadow:0 1px 5px rgba(0,0,0,.4)}.gradient-slider::-moz-range-thumb{width:14px;height:14px;border:3px solid #fff;border-radius:50%;background:#1769d2;box-shadow:0 1px 5px rgba(0,0,0,.4)}.legend-labels{display:flex;justify-content:space-between;font-size:10px;color:var(--muted);margin-top:2px}.scale-readout{display:flex;justify-content:flex-end;align-items:center;margin-top:8px}.index-preview{min-width:96px;padding:6px 9px;border-radius:8px;text-align:center;font-weight:800;font-size:11px}.note{font-size:11px;line-height:1.45;color:var(--muted);margin-top:12px;padding-top:11px;border-top:1px solid var(--line)}
.region{cursor:pointer;vector-effect:non-scaling-stroke;transition:fill .12s}.region:hover{filter:brightness(1.04)}.region-outline{fill:none;stroke:var(--blue);stroke-width:2.8px;vector-effect:non-scaling-stroke;pointer-events:none}.region-outline.pinned{stroke:#0b3f92;stroke-width:3.4px}
.analytics{margin-top:14px;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:0 8px 28px rgba(27,42,78,.06)}.analytics-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:14px}.analytics-title{font-size:20px;font-weight:750;margin:0 0 4px}.analytics-note{font-size:11px;color:var(--muted);line-height:1.45}.chart-title{font-size:15px;font-weight:750;margin:0 0 3px}.chart-note{font-size:10px;color:var(--muted);line-height:1.4;margin-bottom:8px}.salary-chart-wrap{min-height:460px}.chart-controls{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 10px}.chart-control{appearance:none;border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--ink);padding:6px 9px;font:inherit;font-size:10px;cursor:pointer}.chart-control:hover,.chart-control:focus-visible{border-color:#62a5ed;color:#0b5fba;outline:none}.chart-control.is-active{border-color:#1769d2;background:#e8f3ff;color:#0b4f9c;font-weight:750}.chart-svg{width:100%;display:block;overflow:visible}.chart-frame{fill:#fff;stroke:#d7dee9}.chart-grid line{stroke:#e8edf4;stroke-width:1}.chart-axis text{fill:#667085;font-size:10px}.chart-axis path,.chart-axis line{stroke:#cbd4e1}.chart-axis-title{fill:#344054;font-size:11px;font-weight:650}.salary-bar{fill:#4f91dc;cursor:pointer}.salary-bar:hover{fill:#1769d2}.salary-label{fill:#344054;font-size:10px}.salary-value{fill:#14213d;font-size:10px;font-weight:700}.selected-bar{fill:#ef7d32}.reference-line{stroke:#8b95a5;stroke-width:1.4;stroke-dasharray:5 4}.reference-label{fill:#667085;font-size:10px}.analytics-grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,.75fr);gap:18px;margin-top:20px}.analysis-plot{min-width:0}.scatter-point{fill:#4f91dc;fill-opacity:.65;stroke:#fff;stroke-width:1;cursor:pointer}.scatter-point:hover{fill-opacity:1;stroke:#1769d2;stroke-width:2}.scatter-point.is-selected{fill:#ef7d32;fill-opacity:1;stroke:#0b3f92;stroke-width:2.4}.selected-label{fill:#14213d;font-size:10px;font-weight:750;paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round}.quadrant-line{stroke:#9aa6b5;stroke-width:1;stroke-dasharray:4 4}.hist-bar{fill:#8eb8e8}.hist-bar.is-selected{fill:#ef7d32}.hist-value{fill:#344054;font-size:10px;text-anchor:middle}.chart-tooltip{position:fixed;z-index:50;pointer-events:none;display:none;max-width:280px;padding:9px 11px;border:1px solid #ccd6e3;border-radius:9px;background:#fff;color:#14213d;box-shadow:0 8px 24px rgba(27,42,78,.18);font-size:11px;line-height:1.45}.chart-tooltip strong{display:block;margin-bottom:3px}
.region-directory{margin-top:14px;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:0 8px 28px rgba(27,42,78,.06)}.directory-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:12px}.directory-title{font-size:18px;font-weight:750;margin:0 0 4px}.directory-note{font-size:11px;color:var(--muted)}.directory-actions{position:relative;flex:0 0 auto;width:max-content;max-width:100%}.sort-menu{position:relative;width:max-content;max-width:100%}.sort-menu summary{display:flex;align-items:center;gap:4px;width:max-content;max-width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:9px;background:#fff;color:var(--ink);font-size:11px;font-weight:650;cursor:pointer;list-style:none;white-space:nowrap}.sort-menu summary::-webkit-details-marker{display:none}.sort-menu summary:hover,.sort-menu summary:focus-visible{border-color:#62a5ed;outline:none;box-shadow:0 0 0 3px rgba(23,105,210,.10)}.sort-options{position:absolute;right:0;top:calc(100% + 6px);z-index:10;width:max-content;min-width:244px;max-width:min(310px,calc(100vw - 32px));padding:6px;border:1px solid var(--line);border-radius:10px;background:#fff;box-shadow:0 10px 28px rgba(27,42,78,.16)}.sort-option{display:block;width:100%;padding:8px 9px;border:0;border-radius:7px;background:transparent;color:var(--ink);font:inherit;font-size:11px;text-align:left;cursor:pointer;white-space:nowrap}.sort-option:hover,.sort-option:focus-visible{background:#edf6ff;color:#0b5fba;outline:none}.sort-option.is-selected{background:#e8f3ff;color:#0b4f9c;font-weight:750}.region-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.region-button{display:grid;grid-template-columns:42px minmax(0,1fr);align-items:center;width:100%;height:68px;padding:7px 10px;border:1px solid #e2e7ef;border-radius:10px;background:#fafbfc;color:var(--ink);font:inherit;text-align:left;cursor:pointer;overflow:hidden}.region-button:hover,.region-button:focus-visible{border-color:#62a5ed;background:#edf6ff;color:#0b5fba;outline:none}.region-button.is-active{border-color:#1769d2;background:#e8f3ff;color:#0b4f9c;box-shadow:inset 0 0 0 1px #1769d2}.region-code{font-size:13px;font-weight:800;color:var(--muted);font-variant-numeric:tabular-nums}.region-button.is-active .region-code,.region-button:hover .region-code{color:inherit}.region-button-name-wrap{min-width:0}.region-button-name{display:-webkit-box;overflow:hidden;-webkit-box-orient:vertical;-webkit-line-clamp:2;font-size:11px;line-height:1.25;font-weight:650}.region-button-meta{margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:9px;color:var(--muted);font-variant-numeric:tabular-nums}
.forecast-section,.city-section,.downloads{margin-top:14px;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 8px 28px rgba(27,42,78,.06)}.section-title{font-size:20px;margin:0 0 5px}.section-lead{max-width:1120px;font-size:12px;line-height:1.55;color:#475467}.forecast-method{margin:12px 0;padding:11px 13px;border-left:4px solid #1769d2;background:#f4f8ff;font-size:11px;line-height:1.5;color:#344054}.forecast-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:16px}.forecast-table{width:100%;border-collapse:collapse;font-size:11px}.forecast-table th{text-align:left;color:var(--muted);font-size:10px;padding:7px 6px;border-bottom:1px solid var(--line)}.forecast-table td{padding:8px 6px;border-bottom:1px solid #edf0f5;vertical-align:top}.forecast-table .num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}.forecast-name{font-weight:700}.forecast-parent{display:block;color:var(--muted);font-size:9px;margin-top:2px}.forecast-badge{display:inline-block;min-width:36px;padding:4px 5px;border-radius:7px;color:#fff;text-align:center;font-weight:800}.city-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:14px}.city-card{padding:14px;border:1px solid #dfe6ef;border-radius:12px;background:#fafcff}.city-card h3{margin:0 0 5px;font-size:16px}.city-summary{font-size:11px;line-height:1.5;color:#475467;margin-bottom:10px}.city-row{padding:9px 0;border-top:1px solid #e8edf4;font-size:11px;line-height:1.45}.city-row:first-of-type{border-top:0}.city-row-head{display:flex;justify-content:space-between;gap:10px;font-weight:750}.download-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin-top:14px}.download-link{display:block;padding:11px;border:1px solid #dce4ef;border-radius:10px;background:#fafcff;color:#0b5fba;text-decoration:none;font-size:11px;line-height:1.35;overflow-wrap:anywhere}.download-link:hover,.download-link:focus-visible{background:#eaf4ff;border-color:#62a5ed;outline:none}.download-link strong{display:block;color:var(--ink);font-size:12px;margin-bottom:3px}.source-links{margin-top:14px;font-size:11px;line-height:1.7;color:#475467}.source-links a{color:#0b5fba}
@media(max-width:1100px){.region-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.download-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:950px){.method{grid-template-columns:1fr;height:190px;align-content:center}.method-region{min-height:22px;max-height:44px}.layout{grid-template-columns:1fr}.map-card{height:470px}.map-card svg{height:470px}.panel{height:620px;min-height:620px}.analytics-grid,.forecast-grid,.city-grid{grid-template-columns:1fr}}@media(max-width:720px){.region-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.rank-head{grid-template-columns:1fr}.directory-head,.analytics-head{flex-direction:column}.directory-actions,.sort-menu,.sort-menu summary{width:max-content;max-width:100%}.sort-options{left:0;right:auto}.download-grid{grid-template-columns:1fr}}@media(max-width:560px){.shell{padding:10px}.title{font-size:20px}.method{height:218px;padding:12px 14px}.method-text{-webkit-line-clamp:3}.map-card{height:360px}.map-card svg{height:360px}.official-grid{grid-template-columns:1fr}.region-grid{grid-template-columns:1fr}.analytics{padding:13px}.salary-label{font-size:9px}.chart-axis text{font-size:9px}.forecast-section,.city-section,.downloads{padding:14px}.forecast-table{font-size:10px}}
.help-toolbar{display:flex;justify-content:flex-start;margin:0 0 12px}.help-button{appearance:none;border:1px solid #7db8f0;border-radius:10px;background:#eaf4ff;color:#0b4f9c;padding:9px 13px;font:inherit;font-size:13px;font-weight:750;cursor:pointer}.help-button:hover,.help-button:focus-visible{background:#dceeff;border-color:#1769d2;outline:none;box-shadow:0 0 0 3px rgba(23,105,210,.12)}
.index-guide{margin-top:14px;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:22px;box-shadow:0 8px 28px rgba(27,42,78,.06);scroll-margin-top:16px}.index-guide.flash{animation:guideFlash 1.1s ease}.guide-title{font-size:21px;margin:0 0 7px}.guide-lead{font-size:14px;line-height:1.55;color:#344054;max-width:1050px}.guide-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:16px}.guide-card{padding:14px;border:1px solid #e1e7f0;border-radius:12px;background:#fafcff}.guide-card h3{font-size:14px;margin:0 0 6px}.guide-card p,.guide-card li{font-size:12px;line-height:1.55;color:#475467}.guide-card p{margin:0}.guide-card ul{margin:6px 0 0;padding-left:18px}.guide-wide{grid-column:1/-1;background:#f4f8ff;border-color:#cfe3f8}.guide-warning{grid-column:1/-1;background:#fff9eb;border-color:#f2d59a}.plain-formula{display:inline-block;margin-top:7px;padding:7px 9px;border-radius:8px;background:#eef4fb;color:#17365d;font-weight:750}.guide-scale{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-top:9px}.guide-scale span{padding:8px;border-radius:8px;text-align:center;font-size:11px;font-weight:750}.guide-scale .low{background:#dff3df;color:#17602b}.guide-scale .mid{background:#fff2bd;color:#795500}.guide-scale .high{background:#ffe0dd;color:#8a1520}
.year-controls{display:flex;flex-wrap:wrap;gap:7px;margin:13px 0 4px}.year-button{appearance:none;border:1px solid #b9c9dd;border-radius:9px;background:#fff;color:#24415f;padding:7px 12px;font:inherit;font-size:12px;font-weight:750;cursor:pointer}.year-button:hover,.year-button:focus-visible{border-color:#1769d2;background:#eef6ff;outline:none}.year-button.is-active{background:#1769d2;border-color:#1769d2;color:#fff}.table-responsive{width:100%;overflow-x:auto}.table-responsive .forecast-table{min-width:650px}.index-delta{display:block;font-size:9px;color:#52657a;margin-top:2px}.forecast-line{margin-top:20px;padding-top:16px;border-top:1px solid var(--line)}.forecast-line svg{width:100%;height:390px;display:block}.median-actual{stroke:#ed7d31;fill:none;stroke-width:3}.median-predicted{stroke:#1769d2;fill:none;stroke-width:3}.median-estimate{stroke:#7a8699;fill:none;stroke-width:2;stroke-dasharray:5 4}.median-point.actual{fill:#ed7d31}.median-point.predicted{fill:#1769d2}.median-point.estimate{fill:#fff;stroke:#7a8699;stroke-width:3}.selected-year-ring{fill:none;stroke:#0b4f9c;stroke-width:3}
.table-responsive{overflow:visible}.table-responsive .forecast-table{width:100%;min-width:0;table-layout:fixed}.table-responsive .forecast-table th,.table-responsive .forecast-table td{overflow-wrap:anywhere}.table-responsive .forecast-table th:nth-child(1),.table-responsive .forecast-table td:nth-child(1){width:6%;text-align:center}.table-responsive .forecast-table th:nth-child(2),.table-responsive .forecast-table td:nth-child(2){width:32%}.table-responsive .forecast-table th:nth-child(3),.table-responsive .forecast-table td:nth-child(3){width:11%}.table-responsive .forecast-table th:nth-child(4),.table-responsive .forecast-table td:nth-child(4){width:14%}.table-responsive .forecast-table th:nth-child(5),.table-responsive .forecast-table td:nth-child(5){width:11%}.table-responsive .forecast-table th:nth-child(6),.table-responsive .forecast-table td:nth-child(6){width:26%}.table-responsive .forecast-table .num{white-space:normal}.salary-comparison{display:block;line-height:1.35}.salary-current{display:block;color:#667085;font-size:9px}.salary-forecast{display:block;font-weight:800}
.growth-chart-legend{display:flex;flex-wrap:wrap;gap:14px;margin:9px 0;font-size:11px;color:#475467}.growth-chart-legend span{display:inline-flex;align-items:center;gap:6px}.growth-chart-legend i{display:inline-block;width:20px;height:3px}.growth-chart-legend .wage{background:#1769d2}.growth-chart-legend .inflation{background:#d94841}.growth-chart-legend .wage-band{height:10px;background:rgba(23,105,210,.15);border:1px solid rgba(23,105,210,.5)}.growth-chart-legend .inflation-band{height:10px;background:rgba(217,72,65,.15);border:1px solid rgba(217,72,65,.5)}.growth-wage-line{fill:none;stroke:#1769d2;stroke-width:3}.growth-inflation-line{fill:none;stroke:#d94841;stroke-width:3}.growth-wage-band{fill:#1769d2;opacity:.13;stroke:none}.growth-inflation-band{fill:#d94841;opacity:.13;stroke:none}.growth-forecast{stroke-dasharray:7 5}.growth-wage-point{fill:#1769d2}.growth-inflation-point{fill:#d94841}.growth-future-point{fill:#fff;stroke-width:3}.growth-future-point.growth-wage-point{stroke:#1769d2}.growth-future-point.growth-inflation-point{stroke:#d94841}.selected-year-line{stroke:#0b4f9c;stroke-width:2;stroke-dasharray:3 3}
@keyframes guideFlash{0%,100%{box-shadow:0 8px 28px rgba(27,42,78,.06)}50%{box-shadow:0 0 0 5px rgba(23,105,210,.18)}}
@media(max-width:720px){.guide-grid{grid-template-columns:1fr}.guide-wide,.guide-warning{grid-column:auto}.guide-scale{grid-template-columns:1fr}}
</style>
</head>
<body><main class="shell">
<h1 class="title">Насколько трудно найти приемлемо оплачиваемую работу в субъектах РФ</h1>
<div class="subtitle">Рынок труда — официальные данные Росстата за 2024 год; медианная зарплата и прожиточный минимум — 2025 год. 1 — сравнительно легче, 10 — труднее. Наведите курсор на субъект или щёлкните, чтобы закрепить панель.</div>
<section class="method" id="method" aria-live="polite"></section>
<div class="help-toolbar"><button type="button" class="help-button" id="index-help-button" aria-controls="index-guide">Что такое индекс и как он считается?</button></div>
<section class="layout"><div class="map-card" id="map"></div><aside class="panel" id="panel"></aside></section>
<section class="analytics" aria-labelledby="analytics-title"><div class="analytics-head"><div><h2 class="analytics-title" id="analytics-title">Зарплаты и положение регионов на рынке труда</h2><div class="analytics-note">Все графики построены по тем же официальным данным. Нажатие на регион на графике закрепляет его на карте и в правой панели.</div></div></div><div class="salary-chart-wrap"><h3 class="chart-title">Медианная начисленная зарплата по регионам</h3><div class="chart-note">Апрель 2025 года, рублей в месяц. Медиана означает, что половина работников получает меньше, а половина — больше.</div><div class="chart-controls" role="group" aria-label="Выбор регионов на диаграмме зарплат"><button type="button" class="chart-control is-active" data-salary-view="highest">20 самых высоких</button><button type="button" class="chart-control" data-salary-view="lowest">20 самых низких</button><button type="button" class="chart-control" data-salary-view="all">Все регионы</button></div><svg class="chart-svg" id="salary-chart" role="img" aria-label="Диаграмма медианной зарплаты по субъектам России"></svg></div><div class="analytics-grid"><div class="analysis-plot"><h3 class="chart-title">Зарплатный запас и сложность поиска работы</h3><div class="chart-note">По горизонтали — сколько рублей остаётся от медианной зарплаты после вычитания регионального прожиточного минимума; по вертикали — индекс сложности. Это условный запас, а не фактический доход после всех расходов. Размер круга отражает население.</div><svg class="chart-svg" id="relationship-chart" role="img" aria-label="Связь зарплатного запаса и сложности поиска работы"></svg></div><div class="analysis-plot"><h3 class="chart-title">Как распределён индекс сложности</h3><div class="chart-note">Гистограмма показывает, сколько субъектов попало в каждый диапазон. Оранжевым отмечён диапазон выбранного региона.</div><svg class="chart-svg" id="distribution-chart" role="img" aria-label="Распределение субъектов по индексу сложности поиска работы"></svg></div></div></section>
<section class="region-directory"><div class="directory-head"><div><h2 class="directory-title">Все субъекты на карте</h2><div class="directory-note">Код перед названием — код автомобильного региона. Население — оценка Росстата на 01.01.2025; площадь — Росстат/Росреестр на 01.01.2025. Нажмите, чтобы закрепить; повторно — чтобы открепить.</div></div><div class="directory-actions"><details class="sort-menu" id="sort-menu"><summary>Сортировка: <span id="sort-label">По алфавиту</span></summary><div class="sort-options" role="menu"><button type="button" class="sort-option is-selected" data-sort="alphabetical">По алфавиту</button><button type="button" class="sort-option" data-sort="code">По коду региона</button><button type="button" class="sort-option" data-sort="population-desc">Население, сначала больше</button><button type="button" class="sort-option" data-sort="population-asc">Население, сначала меньше</button><button type="button" class="sort-option" data-sort="area-desc">Площадь, сначала большие</button><button type="button" class="sort-option" data-sort="area-asc">Площадь, сначала малые</button><button type="button" class="sort-option" data-sort="score-desc">Трудоустройство, сначала труднее</button><button type="button" class="sort-option" data-sort="score-asc">Трудоустройство, сначала легче</button></div></details></div></div><div class="region-grid" id="region-grid"></div></section>
<section class="forecast-section" aria-labelledby="forecast-title"><h2 class="section-title" id="forecast-title">Какие работы могут быть востребованы и привлекательны</h2><p class="section-lead" id="forecast-lead">Выберите год. Будущий сценарий учитывает последний профессиональный срез Росстата, макропрогноз ЦБ и различия отраслей. Текущий индекс карты при этом не меняется.</p><div class="year-controls" id="year-controls" role="group" aria-label="Выбор года прогноза"></div><div class="forecast-method" id="forecast-method"></div><div class="forecast-grid"><div><h3 class="chart-title">Крупные профессиональные группы</h3><div class="chart-note">Надёжнее для общего выбора направления: региональные данные доступны по девяти группам.</div><div class="table-responsive" id="group-forecast"></div></div><div><h3 class="chart-title">Детальные профессии</h3><div class="chart-note">Более конкретно, но менее надёжно: детализация опубликована только по России в целом.</div><div class="table-responsive" id="occupation-forecast"></div></div></div><div class="forecast-line"><h3 class="chart-title">Медианная зарплата в России: фактические данные и сценарий</h3><div class="chart-note">Оранжевые точки — опубликованная Росстатом годовая медиана за 2020–2025 годы. Значения 2026–2029 рассчитаны по медианному прогнозу роста номинальной зарплаты из июльского макроопроса Банка России. 2030–2032 — осторожное техническое продолжение с замедлением роста, а не официальный прогноз ЦБ. Суммы номинальные.</div><svg id="median-forecast-chart" role="img" aria-label="Фактическая и сценарная медианная зарплата в России"></svg></div></section>
<section class="city-section" aria-labelledby="city-title"><h2 class="section-title" id="city-title">Почему на некоторые работы трудно устроиться в Москве и Санкт-Петербурге</h2><p class="section-lead">В целом оба города относятся к сравнительно доступным рынкам труда. Но внутри отдельных групп ситуация может быть обратной: уже работающих специалистов много относительно числа свободных мест, а часть вакансий после поправки на оплату учитывается слабее. Ниже приведены только выводы, которые следуют из данных Росстата; миграцию, опыт, образование и удалённую работу этот набор не измеряет.</p><div class="city-grid" id="city-explanations"></div></section>
<section class="index-guide" id="index-guide" aria-labelledby="guide-title">
<h2 class="guide-title" id="guide-title">Как понимать индекс сложности трудоустройства</h2>
<p class="guide-lead"><strong>Коротко:</strong> индекс показывает, где обычному соискателю сравнительно труднее найти подходящую работу. Высокий балл означает, что людей, которым нужна работа, относительно много, доступных вакансий мало, профессиональная структура спроса хуже совпадает со структурой занятости и поиск чаще затягивается. Это сравнительный рейтинг регионов, а не вероятность трудоустройства конкретного человека.</p>
<div class="guide-grid">
<article class="guide-card"><h3>Кто считается безработным, а кто рабочей силой?</h3><p>Рабочая сила — занятые плюс безработные. Безработным по методологии МОТ считается человек без работы, который активно ищет её и готов приступить. Человек, который хочет работать, но перестал искать из-за отсутствия вариантов, формально не безработный: он относится к потенциальной рабочей силе.</p></article>
<article class="guide-card"><h3>Что здесь называется вакансией?</h3><p>Это число работников, которых обследованные Росстатом организации сообщили как необходимых для свободных рабочих мест. Один требуемый работник считается одной вакансией.</p></article>
<article class="guide-card"><h3>Широкое недоиспользование труда — 34%</h3><p>Это доля безработных и потенциальной рабочей силы в расширенной рабочей силе. Показатель замечает людей, которые хотят работать, но уже не ведут активный поиск. Чем он выше, тем больше конкуренция и скрытая нехватка доступной работы.</p></article>
<article class="guide-card"><h3>Нехватка вакансий — 30%</h3><p>Доля вакансий равна вакансиям, делённым на занятые и свободные места вместе. Для соискателя направление обратное: чем ниже доля свободных мест, тем труднее найти вариант.</p><span class="plain-formula">Доля вакансий = вакансии ÷ (занятые места + вакансии)</span></article>
<article class="guide-card"><h3>Профессиональное несовпадение — 13%</h3><p>Сравниваются доли девяти крупных групп среди занятых и среди вакансий. Если работников определённого профиля много, а вакансии сосредоточены в других группах, общий рынок хуже подходит людям региона. Маленькая ниша влияет только своей небольшой долей.</p></article>
<article class="guide-card"><h3>Длительность поиска — 8%</h3><p>Поровну учитываются среднее число месяцев поиска и доля безработных, ищущих работу год и дольше. Вес небольшой: показатель наблюдается только среди продолжающих активный поиск и сильнее колеблется в малых выборках.</p></article>
<article class="guide-card"><h3>Низкая оплата — 15%</h3><p>Используется зарплатный запас: из медианной зарплаты региона вычитается местный прожиточный минимум трудоспособного населения. Такой расчёт различает регионы с одинаковым отношением зарплаты к минимуму, но разными денежными суммами. Чем меньше запас в рублях, тем труднее найти работу с приемлемой оплатой.</p></article>
<article class="guide-card"><h3>Что такое привлекательность зарплаты?</h3><p>Это отдельный сравнительный балл 1–10 по величине зарплатного запаса. Например, при зарплате 70 тыс. руб. и минимуме 20 тыс. руб. запас равен 50 тыс. руб. Отношение зарплаты к минимуму показывается рядом только как дополнительный контекст. Запас не является реальным располагаемым доходом: прожиточный минимум не включает полный бюджет семьи, стоимость жилья и индивидуальные расходы.</p></article>
<article class="guide-card guide-wide"><h3>Как пять показателей становятся числом от 1 до 10?</h3><ul><li>Каждый субъект получает относительное место по каждому компоненту — процентиль от 0 до 1.</li><li>Процентиль не позволяет Москве получить преимущество просто из-за численности населения и не даёт одному экстремальному значению растянуть всю шкалу.</li><li>Компоненты складываются с весами 34% + 30% + 13% + 8% + 15%, затем результат переводится в шкалу 1–10.</li><li>Рядом показывается диапазон при сценариях, где зарплата получает от 10% до 20% веса. Если диапазон широк, точное место нестабильно.</li></ul><p>Баллы являются порядковой сравнительной шкалой: 8 не означает, что искать работу ровно вдвое труднее, чем при 4.</p></article>
<article class="guide-card"><h3>Что означает индекс профессиональной группы?</h3><p>Он показывает, где труднее найти приемлемо оплачиваемую работу. Низкооплачиваемая вакансия учитывается частично; вакансия с оплатой не ниже региональной медианы считается полностью, но никогда больше чем одной. Поэтому высокая зарплата не «размножает» вакансии.</p></article>
<article class="guide-card"><h3>Что происходит с нулём вакансий?</h3><p>Ноль не всегда означает максимальную конкуренцию. Если профессиональная группа занимает менее 0,5% работников региона или меньше 5% размера типичной группы и спрос равен нулю, она получает статус «локальный рынок почти отсутствует» без места в рейтинге. Если рынок мал, но вакансии есть, строка сохраняется со сглаживанием.</p></article>
<article class="guide-card"><h3>Зачем нужно сглаживание малых групп?</h3><p>Если в группе мало рабочих мест, одна вакансия может резко изменить процент. Поэтому доля маленькой группы осторожно приближается к среднему по России. Сила поправки <strong>κ (каппа)</strong> оценивается из данных; это модельная стабилизация, а не официальная погрешность Росстата. В опубликованной таблице нет региональных стандартных ошибок, поэтому рядом показывается чувствительность результата при κ вдвое меньше и больше.</p></article>
<article class="guide-card guide-wide"><h3>Как читать шкалу</h3><div class="guide-scale"><span class="low">1–3: сравнительно легче</span><span class="mid">4–6: средняя сложность</span><span class="high">7–10: сравнительно труднее</span></div><p style="margin-top:9px">Разница между 8 и 4 не означает, что нанять работника ровно в два раза труднее. Шкала показывает положение относительно других субъектов РФ.</p></article>
<article class="guide-card guide-warning"><h3>Ограничения, которые нельзя забывать</h3><ul><li>Вакансии 1-Т(проф) относятся к организациям без субъектов малого предпринимательства; малый бизнес, фермерские хозяйства и неформальная занятость охвачены неполно. Поэтому ноль сельскохозяйственных вакансий в Чечне означает ноль в обследованном контуре, а не отсутствие сельского хозяйства вообще.</li><li>Медианная зарплата опубликована по региону, а по профессиональным группам Росстат публикует среднюю зарплату только по России. Зарплата группы в регионе поэтому является прозрачной оценкой: региональная медиана умножается на официальный общероссийский коэффициент группы.</li><li>Региональные профессиональные данные есть только по девяти крупным группам. Детальные профессии опубликованы по России в целом, поэтому их региональный балл — ориентир.</li><li>__VALIDATION_SUMMARY__ Это проверка согласованности, а не доказательство причинности.</li></ul></article>
</div>
</section>
<section class="downloads" aria-labelledby="downloads-title"><h2 class="section-title" id="downloads-title">Скачать данные и проверить источники</h2><p class="section-lead">Локальные файлы позволяют воспроизвести расчёты без повторного скачивания. Рядом приведены официальные страницы и прямые ссылки Росстата, использованные в проекте.</p><div class="download-grid"><a class="download-link" href="professional_group_forecast_2027_2032.csv" download><strong>Прогноз по группам на 2027–2032 годы</strong>CSV с баллами спроса, оплаты и итоговым ориентиром</a><a class="download-link" href="occupation_forecast_2027_2032.csv" download><strong>Прогноз по профессиям на 2027–2032 годы</strong>CSV с детализированным сценарным рейтингом</a><a class="download-link" href="rosstat_regional_job_search_difficulty.csv" download><strong>Индекс по регионам</strong>Региональные компоненты и итоговый балл</a><a class="download-link" href="rosstat_professional_job_search_difficulty_by_region.csv" download><strong>Профессиональные группы регионов</strong>Спрос, занятость, оплата и сглаживание</a><a class="download-link" href="Rosstat_professional_demand_2024.xlsx" download><strong>Профессиональная потребность Росстата</strong>Исходная форма 1‑Т(проф), 31.10.2024</a><a class="download-link" href="official_sources/Rosstat_monthly_wages_by_region_2013_2026-06.xlsx" download><strong>Зарплаты регионов 2013–2026</strong>Помесячный официальный ряд до июня 2026 года, обновлён 02.09.2026</a><a class="download-link" href="official_sources/Rosstat_professional_wage_2025_6.xlsx" download><strong>Зарплаты профессиональных групп</strong>Обследование за октябрь 2025 года</a><a class="download-link" href="official_sources/Sevastopol_labor_force_2026.xlsx" download><strong>Рабочая сила Севастополя</strong>Последний срез апрель–июнь 2026 года</a><a class="download-link" href="official_sources/source_manifest.csv" download><strong>Реестр источников</strong>URL, периоды и контрольные SHA‑256</a></div><div class="source-links"><strong>Официальные страницы:</strong> <a href="https://www.rosstat.gov.ru/labor_market_employment_salaries" target="_blank" rel="noopener">рынок труда и заработная плата</a> · <a href="https://82.rosstat.gov.ru/folder/27489" target="_blank" rel="noopener">рынок труда Севастополя</a> · <a href="https://rosstat.gov.ru/folder/210/document/13211" target="_blank" rel="noopener">«Рабочая сила, занятость и безработица — 2026»</a> · <a href="https://rosstat.gov.ru/storage/mediabank/Region_Pokaz_2025.pdf" target="_blank" rel="noopener">«Регионы России — 2025»</a>.<br><strong>Важно:</strong> 1‑Т(проф) проводится раз в два года по чётным годам. На дату формирования проекта результаты обследования за октябрь 2026 года ещё не опубликованы, поэтому профессиональная структура спроса берётся из последнего официального среза 2024 года. Для Севастополя полный сопоставимый срез девяти групп не опубликован, поэтому показан честный частичный индекс без вымышленных вакансий.</div></section>
</main>
<script>
const geojson=__GEOJSON__;
for(const feature of geojson.features){const g=feature.geometry;if(!g)continue;if(g.type==="Polygon")g.coordinates.forEach(r=>r.reverse());if(g.type==="MultiPolygon")g.coordinates.forEach(p=>p.forEach(r=>r.reverse()));}
const regionData=__REGION_DATA__;
const noDataRegions=__NO_DATA_REGIONS__;
const occupationCatalog=__OCCUPATION_CATALOG__;
const forecastByYear=__FORECAST_BY_YEAR__;
const groupForecastByYear=__GROUP_FORECAST_BY_YEAR__;
const nationalMedianSeries=__NATIONAL_MEDIAN_SERIES__;
const wageInflationSeries=__WAGE_INFLATION_SERIES__;
const embeddedDownloads=__EMBEDDED_DOWNLOADS__;
const officialSourceLinks={
 "Rosstat_professional_demand_2024.xlsx":"https://www.rosstat.gov.ru/compendium/document/13266",
 "official_sources/Rosstat_monthly_wages_by_region_2013_2026-06.xlsx":"https://rosstat.gov.ru/storage/mediabank/tab2-zpl_06-2026.xlsx",
 "official_sources/Rosstat_professional_wage_2025_6.xlsx":"https://rosstat.gov.ru/storage/mediabank/sr-zpl6_2025.xlsx",
 "official_sources/Rosstat_57T_detailed_wages_2025.rar":"https://rosstat.gov.ru/storage/mediabank/sved_57-t_2025.rar",
 "official_sources/Rosstat_subsistence_minimum_by_region.xlsx":"https://rosstat.gov.ru/storage/mediabank/nb_pm_4-1.xlsx",
 "official_sources/Rosstat_labor_force_15plus_2026-07-15.xlsx":"https://rosstat.gov.ru/storage/mediabank/trud-1_15-s.xlsx",
 "official_sources/Rosstat_unemployment_15plus_2026-07-15.xlsx":"https://rosstat.gov.ru/storage/mediabank/trud_3_15-s.xlsx",
 "official_sources/Rosstat_median_wage_2019_2025.xlsx":"https://rosstat.gov.ru/storage/mediabank/tab9-zpl_2025.xlsx",
 "official_sources/Rosstat_CPI_1992_2025.xlsx":"https://rosstat.gov.ru/storage/mediabank/ipc_s_1992-2025.xlsx",
 "official_sources/Rosstat_CPI_monthly_1991_2026-08.xlsx":"https://rosstat.gov.ru/storage/mediabank/ipc_mes_08-2026.xlsx",
 "official_sources/Sevastopol_labor_force_2026.xlsx":"https://82.rosstat.gov.ru/folder/27489",
 "official_sources/Rosstat_CPI_December_2025.html":"https://rosstat.gov.ru/storage/mediabank/3_16-01-2026.html",
 "official_sources/Bank_of_Russia_inflation_forecast_2026-07-24.html":"https://www.cbr.ru/press/pr/?file=24072026_133000key.htm",
 "official_sources/Bank_of_Russia_medium_term_forecast_2026-07-24.pdf":"https://www.cbr.ru/Content/Document/File/194258/forecast_260724.pdf",
 "official_sources/Bank_of_Russia_macroeconomic_survey_2026-07.xlsx":"https://www.cbr.ru/Content/Document/File/144490/full.xlsx",
 "official_sources/Bank_of_Russia_macroeconomic_survey_2026-07.pdf":"https://www.cbr.ru/Content/Document/File/193892/Survey_pdf.pdf"
};
const cityJobExplanations=__CITY_JOB_EXPLANATIONS__;
const forecastMeta=__FORECAST_META__;
function ensureGrowthInflationChart(){
 let host=document.getElementById("wage-inflation-chart");if(host)return host;
 const medianHost=document.getElementById("median-forecast-chart");if(!medianHost)return null;
 medianHost.parentElement.insertAdjacentHTML("afterend",`<div class="forecast-line"><h3 class="chart-title">Рост номинальной зарплаты и среднегодовая инфляция</h3><div class="chart-note">Это не самостоятельно построенный прогноз и не целевая линия ЦБ. Значения взяты из июльского макроэкономического опроса Банка России, проведённого ${forecastMeta.cbr_survey_dates} среди ${forecastMeta.cbr_respondents} экономистов. 2021–2025 годы — фактические значения из сводной таблицы; 2026–${forecastMeta.cbr_table_last_year} годы — медианы прогнозов участников. Полупрозрачные области показывают диапазон от 10-го до 90-го процентиля: внутри него находятся прогнозы основной части опрошенных. Для сопоставимости с зарплатой используется среднегодовой ИПЦ. Банк России не публикует в этом опросе прогноз медианной зарплаты, поэтому синяя линия означает рост номинальной зарплаты в среднем за год, а не рост медианы.</div><div class="growth-chart-legend"><span><i class="wage"></i>Рост номинальной зарплаты</span><span><i class="inflation"></i>Среднегодовая инфляция</span><span><i class="wage-band"></i>10–90% прогнозов зарплаты</span><span><i class="inflation-band"></i>10–90% прогнозов инфляции</span><span>сплошная линия — факт · пунктир — медианный прогноз</span></div><svg id="wage-inflation-chart" role="img" aria-label="Фактические значения и медианные прогнозы аналитиков из макроэкономического опроса Банка России для роста номинальной зарплаты и среднегодовой инфляции"></svg></div>`);
 return document.getElementById("wage-inflation-chart");
}
function renderWageInflationChart(selectedYear=forecastYear){
 const host=ensureGrowthInflationChart();if(!host)return;
 const width=Math.max(360,host.getBoundingClientRect().width||1050),height=390,margin={top:24,right:34,bottom:48,left:70};
 const svg=d3.select(host).attr("viewBox",`0 0 ${width} ${height}`).attr("height",height);svg.selectAll("*").remove();
 const data=wageInflationSeries.map(d=>Object.assign({},d,{year:Number(d.year),salary:Number(d.salary_growth_pct),inflation:Number(d.inflation_pct),salaryLower:Number(d.salary_lower_pct),salaryUpper:Number(d.salary_upper_pct),inflationLower:Number(d.inflation_lower_pct),inflationUpper:Number(d.inflation_upper_pct),real:Number(d.real_wage_growth_pct)})).filter(d=>Number.isFinite(d.salary)&&Number.isFinite(d.inflation));
 const allValues=data.reduce((values,d)=>values.concat([d.salary,d.inflation,Number.isFinite(d.salaryLower)?d.salaryLower:d.salary,Number.isFinite(d.salaryUpper)?d.salaryUpper:d.salary,Number.isFinite(d.inflationLower)?d.inflationLower:d.inflation,Number.isFinite(d.inflationUpper)?d.inflationUpper:d.inflation]),[]),extent=d3.extent(allValues),pad=Math.max(1,(extent[1]-extent[0])*.12),x=d3.scaleLinear().domain(d3.extent(data,d=>d.year)).range([margin.left,width-margin.right]),y=d3.scaleLinear().domain([Math.min(0,extent[0]-pad),extent[1]+pad]).nice().range([height-margin.bottom,margin.top]),lineFor=key=>d3.line().x(d=>x(d.year)).y(d=>y(d[key]));
 svg.append("rect").attr("class","chart-frame").attr("x",margin.left).attr("y",margin.top).attr("width",width-margin.left-margin.right).attr("height",height-margin.top-margin.bottom);
 svg.append("g").attr("class","chart-grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
 const actual=data.filter(d=>d.year<=2025),future=data.filter(d=>d.year>=2025),forecastBand=data.filter(d=>d.year>=2026&&Number.isFinite(d.salaryLower)&&Number.isFinite(d.salaryUpper)&&Number.isFinite(d.inflationLower)&&Number.isFinite(d.inflationUpper));
 if(forecastBand.length){svg.append("path").datum(forecastBand).attr("class","growth-wage-band").attr("d",d3.area().x(d=>x(d.year)).y0(d=>y(d.salaryLower)).y1(d=>y(d.salaryUpper)));svg.append("path").datum(forecastBand).attr("class","growth-inflation-band").attr("d",d3.area().x(d=>x(d.year)).y0(d=>y(d.inflationLower)).y1(d=>y(d.inflationUpper)));}
 for(const [key,lineClass,pointClass] of [["salary","growth-wage-line","growth-wage-point"],["inflation","growth-inflation-line","growth-inflation-point"]]){
  svg.append("path").datum(actual).attr("class",lineClass).attr("d",lineFor(key));svg.append("path").datum(future).attr("class",`${lineClass} growth-forecast`).attr("d",lineFor(key));
  svg.selectAll(`circle.${pointClass}`).data(data).join("circle").attr("class",d=>`${pointClass}${d.year>=2026?" growth-future-point":""}`).attr("cx",d=>x(d.year)).attr("cy",d=>y(d[key])).attr("r",4.5).on("mouseenter",(event,d)=>{const ranges=d.year>=2026?`<br>10–90% прогнозов: зарплата ${fmt(d.salaryLower,1)}–${fmt(d.salaryUpper,1)}%; инфляция ${fmt(d.inflationLower,1)}–${fmt(d.inflationUpper,1)}%`:"";showChartTooltip(event,`<strong>${d.year} год</strong>Рост номинальной зарплаты: ${fmt(d.salary,1)}%<br>Среднегодовая инфляция: ${fmt(d.inflation,1)}%<br>Расчётный рост реальной зарплаты: ${fmt(d.real,1)}%${ranges}<br>${esc(d.status)}`);}).on("mousemove",moveChartTooltip).on("mouseleave",hideChartTooltip);
 }
 if(data.some(d=>d.year===selectedYear))svg.append("line").attr("class","selected-year-line").attr("x1",x(selectedYear)).attr("x2",x(selectedYear)).attr("y1",margin.top).attr("y2",height-margin.bottom);
 svg.append("g").attr("class","chart-axis").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).tickValues(data.map(d=>d.year)).tickFormat(d3.format("d")));svg.append("g").attr("class","chart-axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(d=>`${fmt(d,0)}%`));
 svg.append("text").attr("class","chart-axis-title").attr("x",(margin.left+width-margin.right)/2).attr("y",height-7).attr("text-anchor","middle").text("Год");svg.append("text").attr("class","chart-axis-title").attr("transform","rotate(-90)").attr("x",-(margin.top+height-margin.bottom)/2).attr("y",14).attr("text-anchor","middle").text("Изменение за год, %");
}
const scoreColor=d3.scaleLinear().domain([1,4,6,8,10]).range(["#238b45","#86c440","#ffd54f","#ef3b2c","#7f001f"]).clamp(true);
const map=document.getElementById("map"),panel=document.getElementById("panel"),method=document.getElementById("method"),regionGrid=document.getElementById("region-grid"),sortMenu=document.getElementById("sort-menu"),sortLabel=document.getElementById("sort-label"),salaryChart=document.getElementById("salary-chart"),relationshipChart=document.getElementById("relationship-chart"),distributionChart=document.getElementById("distribution-chart");
const WIDTH=1100,HEIGHT=620;
const indexHelpButton=document.getElementById("index-help-button"),indexGuide=document.getElementById("index-guide");
let pinnedIso=null,currentIso=null,expanded=false,openGroup=null,filterQuery="",sliderValue=null,salaryView="highest",forecastYear=2028;
function esc(value){return String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));}
function fmt(value,digits=0){return value==null?"нет данных":Number(value).toLocaleString("ru-RU",{maximumFractionDigits:digits,minimumFractionDigits:digits});}
function normalizeSearch(value){return String(value).toLocaleLowerCase("ru-RU").replaceAll("ё","е").replace(/\s+/g," ").trim();}
function readableText(color){const c=d3.color(color);if(!c)return "#14213d";const luminance=(.299*c.r+.587*c.g+.114*c.b)/255;return luminance>.58?"#14213d":"#fff";}
function signed(value,digits=1){const n=Number(value);return `${n>=0?"+":""}${fmt(n,digits)}`;}
function forecastTable(rows,isOccupation=false,year=forecastYear){return `<table class="forecast-table"><thead><tr><th>№</th><th>${isOccupation?"Профессия":"Группа"}</th><th class="num">Спрос</th><th class="num">${isOccupation?"Оплата группы":"Оплата"}</th><th class="num">Итог</th><th class="num">Зарплата</th></tr></thead><tbody>${rows.map(row=>`<tr><td>${row.rank}</td><td><span class="forecast-name">${esc(row.name)}</span>${isOccupation?`<span class="forecast-parent">${esc(row.parent_group)}</span>`:""}<span class="forecast-parent">${esc(row.scenario_note||"")}</span></td><td class="num">${fmt(row.demand_score,1)}<span class="forecast-parent">${fmt(row.projected_required_workers)} усл. мест</span></td><td class="num">${fmt(row.salary_score,1)}</td><td class="num"><span class="forecast-badge" style="background:${scoreColor(row.opportunity_score)}">${fmt(row.opportunity_score,1)}</span></td><td class="num"><span class="salary-comparison" title="${esc(row.salary_source||'')}"><span class="salary-current">${row.salary_base_year||2025}: ${fmt(row.current_salary_2025)} ₽${row.salary_observations?` · n=${fmt(row.salary_observations)}`:""}</span><span class="salary-forecast">${year}: ${fmt(row.predicted_salary)} ₽</span><span class="index-delta">номинальный рост ${signed(row.salary_growth_pct,1)}%</span></span></td></tr>`).join("")}</tbody></table>`;}
function renderMedianForecastChart(selectedYear=forecastYear){const host=document.getElementById("median-forecast-chart");if(!host)return;const width=Math.max(360,host.getBoundingClientRect().width||1050),height=390,margin={top:24,right:34,bottom:48,left:78};const svg=d3.select(host).attr("viewBox",`0 0 ${width} ${height}`);svg.selectAll("*").remove();const data=nationalMedianSeries.map(d=>Object.assign({},d,{year:Number(d.year),value:Number(d.median_wage)})).filter(d=>Number.isFinite(d.value));if(!data.length)return;const x=d3.scaleLinear().domain(d3.extent(data,d=>d.year)).range([margin.left,width-margin.right]),extent=d3.extent(data,d=>d.value),pad=Math.max(1,(extent[1]-extent[0])*.12),y=d3.scaleLinear().domain([Math.max(0,extent[0]-pad),extent[1]+pad]).nice().range([height-margin.bottom,margin.top]);svg.append("rect").attr("class","chart-frame").attr("x",margin.left).attr("y",margin.top).attr("width",width-margin.left-margin.right).attr("height",height-margin.top-margin.bottom);svg.append("g").attr("class","chart-grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickSize(-(width-margin.left-margin.right)).tickFormat(""));const actual=data.filter(d=>d.status==="Фактические данные Росстата"),estimate=data.filter(d=>d.year===2026),predicted=data.filter(d=>d.year>=2026),lastActual=actual.length?actual[actual.length-1]:null;const line=d3.line().x(d=>x(d.year)).y(d=>y(d.value));svg.append("path").datum(actual).attr("class","median-actual").attr("d",line);if(lastActual&&estimate.length)svg.append("path").datum([lastActual].concat(estimate)).attr("class","median-estimate").attr("d",line);svg.append("path").datum(predicted).attr("class","median-predicted").attr("d",line);svg.selectAll("circle.median-point").data(data).join("circle").attr("class",d=>`median-point ${d.status==="Фактические данные Росстата"?"actual":d.year===2026?"estimate":"predicted"}`).attr("cx",d=>x(d.year)).attr("cy",d=>y(d.value)).attr("r",5).on("mouseenter",(event,d)=>showChartTooltip(event,`<strong>${d.year} год</strong>${esc(d.status)}<br>Медианная зарплата: ${rubles(d.value)}${d.year===2026?"<br>Оценка полного года, не опубликованный факт Росстата.":""}`)).on("mousemove",moveChartTooltip).on("mouseleave",hideChartTooltip);const selected=data.find(d=>d.year===selectedYear);if(selected)svg.append("circle").attr("class","selected-year-ring").attr("cx",x(selected.year)).attr("cy",y(selected.value)).attr("r",10);svg.append("g").attr("class","chart-axis").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).tickValues(data.map(d=>d.year)).tickFormat(d3.format("d")));svg.append("g").attr("class","chart-axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(d=>`${fmt(d/1000,0)} тыс.`));svg.append("text").attr("class","chart-axis-title").attr("x",(margin.left+width-margin.right)/2).attr("y",height-7).attr("text-anchor","middle").text("Год");svg.append("text").attr("class","chart-axis-title").attr("transform","rotate(-90)").attr("x",-(margin.top+height-margin.bottom)/2).attr("y",15).attr("text-anchor","middle").text("Медианная зарплата, руб./мес.");}
function renderForecast(){const years=forecastMeta.years||Object.keys(forecastByYear).map(Number);document.getElementById("year-controls").innerHTML=years.map(year=>`<button type="button" class="year-button${year===forecastYear?" is-active":""}" data-year="${year}">${year}</button>`).join("");document.getElementById("forecast-title").textContent=`Какие работы могут быть востребованы и привлекательны в ${forecastYear} году`;document.getElementById("forecast-lead").textContent=`Ориентир на ${forecastYear} год на 65% зависит от сценарного спроса работодателей и на 35% — от сравнительной оплаты. Чем выше балл, тем сильнее сочетание вероятной потребности в кадрах и зарплатной привлекательности. Это сценарий, а не обещание количества вакансий.`;document.getElementById("forecast-method").innerHTML=`Основа спроса — обследование Росстата 1‑Т(проф) на 31.10.2024. К нему применена умеренная отраслевая поправка: в строительстве и связанных работах учтён спад 2026 года и дорогой кредит; для квалифицированных рабочих и специалистов — официальный кадровый приоритет до 2030 года; для стандартных офисных функций — постепенная цифровизация. Зарплата 2026–2029 следует медиане июльского макроопроса Банка России, после 2029 года показано техническое замедляющееся продолжение. Текущий индекс карты не изменён. <a href="https://www.cbr.ru/press/pr/?file=24072026_133000key.htm" target="_blank" rel="noopener noreferrer">Сценарий ЦБ</a> · <a href="https://www.rosstat.gov.ru/storage/mediabank/89_17-06-2026.html" target="_blank" rel="noopener noreferrer">ВВП и строительство</a> · <a href="https://mintrud.gov.ru/employment/260" target="_blank" rel="noopener noreferrer">прогноз Минтруда</a>.`;document.getElementById("group-forecast").innerHTML=forecastTable((groupForecastByYear[String(forecastYear)]||[]).slice(0,9),false,forecastYear);document.getElementById("occupation-forecast").innerHTML=forecastTable((forecastByYear[String(forecastYear)]||[]).slice(0,10),true,forecastYear);document.querySelectorAll(".year-button").forEach(button=>button.addEventListener("click",()=>{forecastYear=Number(button.dataset.year);renderForecast();}));renderMedianForecastChart(forecastYear);}
function cityReason(row){const rawRate=row.staff_count+row.required_workers>0?100*row.required_workers/(row.staff_count+row.required_workers):0;const payEffect=row.salary_adjusted_vacancy_rate_pct<row.adjusted_vacancy_rate_pct-.05?` После учёта уровня оплаты доступность снижается с ${fmt(row.adjusted_vacancy_rate_pct,1)}% до ${fmt(row.salary_adjusted_vacancy_rate_pct,1)}%.`:" Уровень оплаты почти не уменьшает число учитываемых возможностей.";return `На ${fmt(row.staff_count)} занятых приходится ${fmt(row.required_workers)} заявленных свободных мест — около ${fmt(rawRate,1)}% рабочих мест группы. Поэтому высокая численность города сама по себе не означает лёгкий вход именно в эту группу.${payEffect}`;}
function renderCities(){const host=document.getElementById("city-explanations");host.innerHTML=Object.entries(cityJobExplanations).map(([city,data])=>`<article class="city-card"><h3>${esc(city)}</h3><div class="city-summary">Общий индекс города — <strong>${fmt(data.regional_score,1)} из 10</strong>: в целом рынок сравнительно доступен. Широкое недоиспользование труда — ${fmt(data.broad_underutilization_rate,1)}%, несовпадение структуры профессий — ${fmt(data.mismatch_pct,1)}%, условный зарплатный запас — ${fmt(data.salary_buffer)} ₽. Трудность ниже относится не ко всему городу, а к конкретным группам.</div>${data.rows.map(row=>`<div class="city-row"><div class="city-row-head"><span>${esc(row.name)}</span><span style="color:${scoreColor(row.score)}">${fmt(row.score,1)} / 10</span></div><div>${cityReason(row)}</div><div class="forecast-parent">Примеры профессий внутри группы: ${row.examples.map(esc).join(", ")}.</div></div>`).join("")}</article>`).join("");}
function updateDownloads(){const grid=document.querySelector(".download-grid"),links=grid.querySelectorAll("a");if(links[0]){links[0].href="professional_group_forecast_2027_2032.csv";links[0].innerHTML="<strong>Прогноз по группам на 2027–2032 годы</strong>CSV с годом, оценкой оплаты и ростом зарплаты";}if(links[1]){links[1].href="occupation_forecast_2027_2032.csv";links[1].innerHTML="<strong>Прогноз по профессиям на 2027–2032 годы</strong>Детализированный многолетний сценарий";}grid.insertAdjacentHTML("beforeend",`<a class="download-link" href="cbr_macro_survey_wages_inflation_2021_2029.csv" download><strong>Зарплата и инфляция — опрос ЦБ</strong>Факты 2021–2025, медианы и диапазоны прогнозов 2026–2029</a><a class="download-link" href="official_sources/Bank_of_Russia_macroeconomic_survey_2026-07.xlsx" download><strong>Агрегированные результаты опроса ЦБ</strong>Официальный Excel с историей и диапазонами прогнозов</a><a class="download-link" href="official_sources/Bank_of_Russia_macroeconomic_survey_2026-07.pdf" download><strong>Графики макроопроса Банка России</strong>Официальная презентация от 15.07.2026</a><a class="download-link" href="rosstat_current_labor_context_2026.csv" download><strong>Актуальный рынок труда, март–май 2026</strong>Рабочая сила и безработица по субъектам</a><a class="download-link" href="official_sources/Rosstat_labor_force_15plus_2026-07-15.xlsx" download><strong>Рабочая сила 15+, Росстат</strong>Официальный Excel, обновлён 15.07.2026</a><a class="download-link" href="official_sources/Rosstat_unemployment_15plus_2026-07-15.xlsx" download><strong>Безработица 15+, Росстат</strong>Официальный Excel, обновлён 15.07.2026</a><a class="download-link" href="official_sources/Rosstat_median_wage_2019_2025.xlsx" download><strong>История медианной зарплаты</strong>Фактические годовые данные 2019–2025</a><a class="download-link" href="official_sources/Bank_of_Russia_inflation_forecast_2026-07-24.html" download><strong>Решение Банка России</strong>Базовый прогноз самого регулятора от 24.07.2026</a><a class="download-link" href="official_sources/Bank_of_Russia_medium_term_forecast_2026-07-24.pdf" download><strong>Среднесрочный прогноз Банка России</strong>Официальная таблица базового сценария</a>`);const sources=document.querySelector(".source-links");if(sources)sources.insertAdjacentHTML("beforeend",`<br><strong>Прогноз зарплаты и инфляции:</strong> <a href="https://www.cbr.ru/statistics/ddkp/mo_br/" target="_blank" rel="noopener noreferrer">макроэкономический опрос Банка России — таблица и интерактивные графики</a> · <a href="https://www.cbr.ru/Content/Document/File/144490/full.xlsx" target="_blank" rel="noopener noreferrer">агрегированные результаты (Excel)</a> · <a href="https://www.cbr.ru/Content/Document/File/193892/Survey_pdf.pdf" target="_blank" rel="noopener noreferrer">официальные графики опроса (PDF)</a>.`);}
function addOccupationSalaryDownloads(){const grid=document.querySelector(".download-grid");if(!grid||grid.querySelector('[href="trudvsem_top_occupation_salary_snapshot_2026.csv"]'))return;grid.insertAdjacentHTML("beforeend",`<a class="download-link" href="trudvsem_top_occupation_salary_snapshot_2026.csv" download><strong>Медианные зарплатные предложения профессий</strong>Срез официального API «Работа России» с размером выборки</a><a class="download-link" href="official_sources/Rosstat_57T_detailed_wages_2025.rar" download><strong>Полный архив Росстата 57-Т</strong>Зарплаты по составным профессиональным группам ОКЗ за октябрь 2025</a>`);}
function activateEmbeddedDownloads(){document.querySelectorAll("a[download]").forEach(link=>{const key=link.getAttribute("href");link.dataset.downloadFile=key;if(officialSourceLinks[key]){link.href=officialSourceLinks[key];link.target="_blank";link.rel="noopener noreferrer";link.removeAttribute("download");link.dataset.officialSource="true";return;}if(location.protocol==="file:"){link.href=new URL(key,location.href).href;return;}const dataUri=embeddedDownloads[key];if(dataUri){link.href="#download";link.addEventListener("click",event=>{event.preventDefault();event.stopPropagation();try{const comma=dataUri.indexOf(","),meta=dataUri.slice(5,comma),base64=dataUri.slice(comma+1),mime=meta.split(";")[0]||"application/octet-stream",binary=atob(base64),bytes=new Uint8Array(binary.length);for(let index=0;index<binary.length;index++)bytes[index]=binary.charCodeAt(index);const objectUrl=URL.createObjectURL(new Blob([bytes],{type:mime})),download=document.createElement("a");download.href=objectUrl;download.download=key.split("/").pop();document.body.appendChild(download);download.click();download.remove();setTimeout(()=>URL.revokeObjectURL(objectUrl),5000);}catch(error){link.href=key;location.href=key;}});return;}link.href=key;});}
function occupationRegionalScore(occupation,d){const parent=(d.ranking||[]).find(row=>row.name===occupation.parent_group);const context=parent&&parent.score!=null?parent.score:d.score;return Math.max(1,Math.min(10,.75*occupation.national_score+.25*context));}
function renderMethod(d){method.innerHTML=`<div><div class="eyebrow">Сложность поиска приемлемо оплачиваемой работы</div><div class="method-region">${esc(d.region)} · ${d.score.toFixed(1)} из 10</div></div><div><div class="method-title">${esc(d.interpretation)}</div><div class="method-text">Высокий балл означает: желающих работать больше, вакансий меньше, профессии хуже совпадают, поиск длится дольше или зарплатный запас меньше. Веса: 34% + 30% + 13% + 8% + 15%. При других сценариях: ${d.score_min.toFixed(1)}–${d.score_max.toFixed(1)}. Свежий официальный контекст (${esc(d.current_labor_period)}): рабочая сила ${fmt(d.labor_force_thousand_current,1)} тыс. человек, безработица ${fmt(d.unemployment_rate_current,1)}%; он показан для актуальности, но не смешивается с индексом за 2024 год.</div></div>`;}
function renderPanel(d,pinned=false,preserve=false){
 if(!preserve){expanded=false;openGroup=null;filterQuery="";sliderValue=d.score;}
 const allRows=(d.ranking||[]).map((row,index)=>({row,index}));
 const normalizedQuery=normalizeSearch(filterQuery);
 const filteredRows=normalizedQuery?allRows.filter(({row})=>normalizeSearch(row.name).includes(normalizedQuery)):allRows;
 const visibleRows=normalizedQuery?filteredRows:filteredRows.slice(0,expanded?9:5);
 const groupRows=visibleRows.map(({row:r,index})=>{
  const key=`g:${index}`;
  const rankText=r.rank??"—",scoreBadge=r.score==null?`<div class="badge" title="Локальный рынок не ранжируется" style="background:#8b95a5">нет</div>`:`<div class="badge" style="background:${scoreColor(r.score)}">${r.score.toFixed(1)}</div>`;
  const marketExplanation=r.structural_zero_market?`Локальный рынок почти отсутствует, поэтому нулю вакансий не присвоен ложный максимальный балл.`:`После поправки на оплату ${fmt(r.required_workers)} вакансий эквивалентны ${fmt(r.salary_adjusted_required_workers,1)} вакансиям с оплатой не ниже местной медианы.`;
  const detail=openGroup===key?`<div class="group-detail"><strong>${esc(r.name)}</strong><br>Организации сообщили, что им требуется ${fmt(r.required_workers)} работников; занято ${fmt(r.staff_count)} человек. ${marketExplanation}<br>Оценочная зарплата группы: ${fmt(r.estimated_professional_wage)} руб.; условный запас после прожиточного минимума: ${fmt(r.estimated_professional_wage_buffer)} руб.; привлекательность оплаты: ${fmt(r.salary_attractiveness_score,1)} из 10. Это оценка из региональной медианы и официального зарплатного коэффициента группы по России.<br>Доля свободных мест после сглаживания: ${fmt(r.adjusted_vacancy_rate_pct,1)}%; с учётом качества оплаты: ${fmt(r.salary_adjusted_vacancy_rate_pct,1)}%.<br>На группу приходится ${fmt(r.staff_share_pct,1)}% занятых и ${fmt(r.vacancy_share_pct,1)}% вакансий региона. Баллы при других настройках: ${fmt(r.score_min,1)}–${fmt(r.score_max,1)}; место: ${fmt(r.rank_min)}–${fmt(r.rank_max)}. ${esc(r.observation_status)}. Для сглаживания использована κ = ${fmt(r.kappa_mom,0)}; собственные данные региона имеют вес ${fmt(r.credibility_weight_pct,0)}%.</div>`:"";
  return `<div><div class="rank-row"><div class="rank-num">${rankText}</div><div><button type="button" class="rank-name" data-key="${key}" aria-expanded="${openGroup===key}">${esc(r.name)}</button><div class="rank-meta">${fmt(r.required_workers)} вакансий · с учётом оплаты ${fmt(r.salary_adjusted_required_workers,1)}</div></div>${scoreBadge}</div>${detail}</div>`;
 }).join("");
 const rankedOccupations=occupationCatalog.map((row,catalogIndex)=>({row,catalogIndex,score:occupationRegionalScore(row,d)})).sort((a,b)=>b.score-a.score||b.row.vacancy_rate_pct-a.row.vacancy_rate_pct||b.row.required_workers-a.row.required_workers).map((item,index)=>({...item,rank:index+1}));
 const occupationMatches=normalizedQuery?rankedOccupations.filter(({row})=>normalizeSearch(row.name).includes(normalizedQuery)):[];
 const occupationRows=occupationMatches.map(({row:r,catalogIndex,score,rank})=>{
  const key=`o:${catalogIndex}`,parentEntry=allRows.find(({row})=>row.name===r.parent_group),parentRank=parentEntry?.row.rank??null,parentRankText=parentRank?`${parentRank}-е место среди наблюдаемых групп`:`локальный рынок группы не ранжируется`;
  const detail=openGroup===key?`<div class="group-detail"><strong>${esc(r.name)}</strong><br>Профессия относится к группе «${esc(r.parent_group)}», которая занимает ${parentRankText} по сложности трудоустройства среди девяти групп региона.<br>Профессия находится на месте № ${rank} из ${rankedOccupations.length} в расчётном списке: места идут от самых трудных для соискателя к более доступным.<br>По России организациям требуется ${fmt(r.required_workers)} работников этой профессии. Свободные места составляют ${fmt(r.vacancy_rate_pct,1)}% всех рабочих мест профессии; после сглаживания малой базы — ${fmt(r.adjusted_vacancy_rate_pct,1)}%.<br>Ориентировочная оценка для региона «${esc(d.region)}»: ${score.toFixed(1)} из 10. Детальные профессии Росстат публикует только по России в целом, поэтому это модельный ориентир, а не отдельное региональное наблюдение.</div>`:"";
  return `<div><div class="rank-row"><div class="rank-num">${rank}</div><div><button type="button" class="rank-name" data-key="${key}" aria-expanded="${openGroup===key}">${esc(r.name)}</button><div class="rank-meta">${esc(r.parent_group)} · область № ${parentRank??"—"} · РФ: ${fmt(r.required_workers)} чел.</div></div><div class="badge" title="Расчётная оценка для региона" style="background:${scoreColor(score)}">${score.toFixed(1)}</div></div>${detail}</div>`;
 }).join("");
 const rows=normalizedQuery?`${groupRows?`<div class="rank-subtitle"><strong>Совпавшие группы региона</strong></div>${groupRows}`:""}${occupationRows?`<div class="rank-subtitle" style="margin-top:10px"><strong>Детальные профессии Росстата</strong> · показатели профессий опубликованы по России в целом</div>${occupationRows}`:""}`:(groupRows||`<div class="rank-empty">Для этого субъекта Росстат не опубликовал полный сопоставимый срез девяти профессиональных групп. Общий индекс рассчитан только по доступным официальным показателям; поиск профессий ниже остаётся общероссийским ориентиром.</div>`);
 const empty=normalizedQuery&&!groupRows&&!occupationRows?`<div class="rank-empty">По запросу «${esc(filterQuery)}» совпадений нет.</div>`:"";
 const more=!normalizedQuery&&allRows.length>5?`<div class="more-wrap"><button type="button" class="more">${expanded?"Свернуть":"Показать ещё"}</button></div>`:"";
 const selected=Number(sliderValue??d.score),previewColor=scoreColor(selected);
 panel.innerHTML=`<div class="eyebrow">Субъект РФ · ${pinned?"закреплено":"наведение"}</div><div class="region-name">${esc(d.region)}</div><div class="overall"><span>Сложность поиска приемлемо оплачиваемой работы<br><small>1 — легче, 10 — труднее среди регионов</small></span><strong style="color:${scoreColor(d.score)}">${d.score.toFixed(1)}</strong></div><div class="official-grid"><div><span>Медианная зарплата, апрель 2025</span><strong>${fmt(d.median_wage_2025)} руб.</strong></div><div><span>Привлекательность зарплаты</span><strong>${fmt(d.salary_attractiveness_score,1)} из 10</strong></div><div><span>Условный запас после минимума</span><strong>${fmt(d.median_wage_buffer)} руб.</strong></div><div><span>Покрытие прожиточного минимума</span><strong>${fmt(d.median_wage_to_subsistence_ratio,2)} раза</strong></div><div><span>Доля вакантных мест</span><strong>${fmt(d.overall_vacancy_rate_pct,1)}%</strong></div><div><span>Безработные + потенциальная рабочая сила</span><strong>${fmt(d.broad_underutilization_rate,1)}%</strong></div><div><span>Среднее время поиска</span><strong>${fmt(d.average_job_search_months,1)} мес.</strong></div><div><span>Ищут год и дольше</span><strong>${fmt(d.long_term_unemployment_share,1)}%</strong></div><div><span>Несовпадение профструктуры</span><strong>${fmt(d.occupational_mismatch_pct,1)}%</strong></div></div><div class="rank-head"><label class="rank-search-label">Поиск по всем профессиям<input class="rank-search" type="search" value="${esc(filterQuery)}" placeholder="Например: программист" aria-label="Поиск профессиональной области или профессии"></label><div class="rank-title">Группы работ, где труднее найти достойно оплачиваемую работу</div></div><div class="rank-subtitle">Рейтинг учитывает доступность вакансий и оплату. Локальный рынок почти нулевого размера показывается в поиске, но не получает искусственный балл 10 и место в рейтинге.</div>${rows}${empty}${more}<div class="legend"><div class="gradient-scale"><div class="gradient"></div><input class="gradient-slider" type="range" min="1" max="10" step="0.1" value="${selected.toFixed(1)}" aria-label="Выбрать индекс сложности трудоустройства"></div><div class="legend-labels"><span>1 · сравнительно легче</span><span>5 · средне</span><span>10 · сравнительно труднее</span></div><div class="scale-readout"><span class="index-preview" style="background:${previewColor};color:${readableText(previewColor)}">Выбрано: ${selected.toFixed(1)}</span></div></div><div class="note">Рынок труда и вакансии — 2024 год; зарплата и прожиточный минимум — 2025 год. ${esc(d.coverage_note)} ${esc(d.denominator_stability)} ${esc(d.validation_summary)}</div>`;
 panel.querySelectorAll(".rank-name").forEach(button=>button.addEventListener("click",()=>{const key=button.dataset.key;openGroup=openGroup===key?null:key;renderPanel(d,pinned,true);}));
 const moreButton=panel.querySelector(".more");if(moreButton)moreButton.addEventListener("click",()=>{expanded=!expanded;openGroup=null;renderPanel(d,pinned,true);});
 const searchInput=panel.querySelector(".rank-search");if(searchInput)searchInput.addEventListener("input",()=>{filterQuery=searchInput.value;openGroup=null;renderPanel(d,pinned,true);const next=panel.querySelector(".rank-search");if(next){next.focus();next.setSelectionRange(next.value.length,next.value.length);}});
 const slider=panel.querySelector(".gradient-slider");if(slider)slider.addEventListener("input",()=>{sliderValue=Number(slider.value);const preview=panel.querySelector(".index-preview"),color=scoreColor(sliderValue);preview.textContent=`Выбрано: ${sliderValue.toFixed(1)}`;preview.style.background=color;preview.style.color=readableText(color);});
}
const analyticsData=Object.entries(regionData).map(([iso,d])=>({iso,...d}));
const chartTooltip=d3.select("body").append("div").attr("class","chart-tooltip").attr("role","tooltip");
function showChartTooltip(event,html){chartTooltip.html(html).style("display","block");moveChartTooltip(event);}
function moveChartTooltip(event){const node=chartTooltip.node(),pad=14,w=node.offsetWidth,h=node.offsetHeight;let left=event.clientX+pad,top=event.clientY+pad;if(left+w>window.innerWidth-8)left=event.clientX-w-pad;if(top+h>window.innerHeight-8)top=event.clientY-h-pad;chartTooltip.style("left",`${Math.max(8,left)}px`).style("top",`${Math.max(8,top)}px`);}
function hideChartTooltip(){chartTooltip.style("display","none");}
function rubles(value){return `${fmt(value)} руб.`;}
function renderSalaryChart(){
 const sorted=[...analyticsData].sort((a,b)=>b.median_wage_2025-a.median_wage_2025||a.region.localeCompare(b.region,"ru"));
 const data=salaryView==="highest"?sorted.slice(0,20):salaryView==="lowest"?sorted.slice(-20).reverse():sorted;
 const width=Math.max(320,salaryChart.getBoundingClientRect().width||900),rowHeight=salaryView==="all"?22:25,height=Math.max(460,data.length*rowHeight+65),margin={top:18,right:86,bottom:38,left:Math.min(235,Math.max(135,width*.23))};
 const svg=d3.select(salaryChart).attr("viewBox",`0 0 ${width} ${height}`).attr("height",height);svg.selectAll("*").remove();
 const x=d3.scaleLinear().domain([0,d3.max(analyticsData,d=>d.median_wage_2025)*1.08]).nice().range([margin.left,width-margin.right]);
 const y=d3.scaleBand().domain(data.map(d=>d.iso)).range([margin.top,height-margin.bottom]).padding(.2);
 svg.append("rect").attr("class","chart-frame").attr("x",margin.left).attr("y",margin.top).attr("width",width-margin.left-margin.right).attr("height",height-margin.top-margin.bottom);
 const ticks=width<600?4:6;svg.append("g").attr("class","chart-grid").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).ticks(ticks).tickSize(-(height-margin.top-margin.bottom)).tickFormat(""));
 const regionsMedian=d3.median(analyticsData,d=>d.median_wage_2025);svg.append("line").attr("class","reference-line").attr("x1",x(regionsMedian)).attr("x2",x(regionsMedian)).attr("y1",margin.top).attr("y2",height-margin.bottom);svg.append("text").attr("class","reference-label").attr("x",x(regionsMedian)+4).attr("y",margin.top+11).text(`медиана субъектов: ${fmt(regionsMedian)} руб.`);
 svg.append("g").selectAll("rect").data(data).join("rect").attr("class",d=>`salary-bar${d.iso===currentIso?" selected-bar":""}`).attr("x",x(0)).attr("y",d=>y(d.iso)).attr("width",d=>Math.max(1,x(d.median_wage_2025)-x(0))).attr("height",y.bandwidth()).on("mouseenter",(event,d)=>showChartTooltip(event,`<strong>${esc(d.region)}</strong>Медианная зарплата: ${rubles(d.median_wage_2025)}<br>Прожиточный минимум: ${rubles(d.working_age_subsistence_2025)}<br>Условный запас: ${rubles(d.median_wage_buffer)}<br>Покрытие минимума: ${fmt(d.median_wage_to_subsistence_ratio,2)} раза<br>Сложность поиска работы: ${fmt(d.score,1)} из 10`)).on("mousemove",moveChartTooltip).on("mouseleave",hideChartTooltip).on("click",(event,d)=>pinIso(d.iso));
 svg.append("g").selectAll("text.salary-label").data(data).join("text").attr("class","salary-label").attr("x",margin.left-7).attr("y",d=>y(d.iso)+y.bandwidth()/2+3).attr("text-anchor","end").text(d=>d.region.length>31?`${d.region.slice(0,29)}…`:d.region);
 svg.append("g").selectAll("text.salary-value").data(data).join("text").attr("class","salary-value").attr("x",d=>x(d.median_wage_2025)+5).attr("y",d=>y(d.iso)+y.bandwidth()/2+3).text(d=>fmt(d.median_wage_2025));
 svg.append("g").attr("class","chart-axis").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).ticks(ticks).tickFormat(d=>`${fmt(d/1000)} тыс.`));svg.append("text").attr("class","chart-axis-title").attr("x",(margin.left+width-margin.right)/2).attr("y",height-5).attr("text-anchor","middle").text("Медианная зарплата, рублей в месяц");
}
function renderRelationshipChart(){
 const width=Math.max(320,relationshipChart.getBoundingClientRect().width||680),height=410,margin={top:20,right:25,bottom:52,left:62};const svg=d3.select(relationshipChart).attr("viewBox",`0 0 ${width} ${height}`).attr("height",height);svg.selectAll("*").remove();
 const xExtent=d3.extent(analyticsData,d=>d.median_wage_buffer),xPad=(xExtent[1]-xExtent[0])*.08,x=d3.scaleLinear().domain([Math.max(0,xExtent[0]-xPad),xExtent[1]+xPad]).nice().range([margin.left,width-margin.right]),y=d3.scaleLinear().domain([1,10]).range([height-margin.bottom,margin.top]),r=d3.scaleSqrt().domain(d3.extent(analyticsData,d=>d.population_people)).range([3.5,13]);
 const xMedian=d3.median(analyticsData,d=>d.median_wage_buffer),yMedian=d3.median(analyticsData,d=>d.score);svg.append("rect").attr("class","chart-frame").attr("x",margin.left).attr("y",margin.top).attr("width",width-margin.left-margin.right).attr("height",height-margin.top-margin.bottom);svg.append("line").attr("class","quadrant-line").attr("x1",x(xMedian)).attr("x2",x(xMedian)).attr("y1",margin.top).attr("y2",height-margin.bottom);svg.append("line").attr("class","quadrant-line").attr("x1",margin.left).attr("x2",width-margin.right).attr("y1",y(yMedian)).attr("y2",y(yMedian));
 svg.append("g").attr("class","chart-grid").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).ticks(width<600?4:6).tickSize(-(height-margin.top-margin.bottom)).tickFormat(""));svg.append("g").attr("class","chart-grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
 svg.append("g").selectAll("circle").data(analyticsData.slice().sort((a,b)=>b.population_people-a.population_people)).join("circle").attr("class",d=>`scatter-point${d.iso===currentIso?" is-selected":""}`).attr("cx",d=>x(d.median_wage_buffer)).attr("cy",d=>y(d.score)).attr("r",d=>r(d.population_people)).on("mouseenter",(event,d)=>showChartTooltip(event,`<strong>${esc(d.region)}</strong>Условный зарплатный запас: ${rubles(d.median_wage_buffer)}<br>Медианная зарплата: ${rubles(d.median_wage_2025)}<br>Прожиточный минимум: ${rubles(d.working_age_subsistence_2025)}<br>Покрытие минимума: ${fmt(d.median_wage_to_subsistence_ratio,2)} раза<br>Сложность поиска: ${fmt(d.score,1)} из 10<br>Население: ${fmt(d.population_people)} чел.`)).on("mousemove",moveChartTooltip).on("mouseleave",hideChartTooltip).on("click",(event,d)=>pinIso(d.iso));
 const selected=analyticsData.find(d=>d.iso===currentIso);if(selected)svg.append("text").attr("class","selected-label").attr("x",x(selected.median_wage_buffer)+(x(selected.median_wage_buffer)>width*.72?-8:8)).attr("y",y(selected.score)-9).attr("text-anchor",x(selected.median_wage_buffer)>width*.72?"end":"start").text(selected.region);
 svg.append("g").attr("class","chart-axis").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).ticks(width<600?4:6).tickFormat(d=>`${fmt(d/1000,0)} тыс.`));svg.append("g").attr("class","chart-axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(5));svg.append("text").attr("class","chart-axis-title").attr("x",(margin.left+width-margin.right)/2).attr("y",height-6).attr("text-anchor","middle").text("Условный зарплатный запас, рублей в месяц");svg.append("text").attr("class","chart-axis-title").attr("transform","rotate(-90)").attr("x",-(margin.top+height-margin.bottom)/2).attr("y",14).attr("text-anchor","middle").text("Сложность поиска работы, 1–10");
}
function renderDistributionChart(){
 const width=Math.max(320,distributionChart.getBoundingClientRect().width||430),height=410,margin={top:20,right:18,bottom:52,left:54};const svg=d3.select(distributionChart).attr("viewBox",`0 0 ${width} ${height}`).attr("height",height);svg.selectAll("*").remove();const bins=d3.bin().value(d=>d.score).domain([1,10]).thresholds(d3.range(1,11,1))(analyticsData),x=d3.scaleLinear().domain([1,10]).range([margin.left,width-margin.right]),y=d3.scaleLinear().domain([0,d3.max(bins,d=>d.length)]).nice().range([height-margin.bottom,margin.top]);const selected=analyticsData.find(d=>d.iso===currentIso);
 svg.append("rect").attr("class","chart-frame").attr("x",margin.left).attr("y",margin.top).attr("width",width-margin.left-margin.right).attr("height",height-margin.top-margin.bottom);svg.append("g").attr("class","chart-grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
 svg.append("g").selectAll("rect").data(bins).join("rect").attr("class",b=>`hist-bar${selected&&selected.score>=b.x0&&(selected.score<b.x1||b.x1===10&&selected.score<=10)?" is-selected":""}`).attr("x",b=>x(b.x0)+2).attr("y",b=>y(b.length)).attr("width",b=>Math.max(1,x(b.x1)-x(b.x0)-4)).attr("height",b=>y(0)-y(b.length)).on("mouseenter",(event,b)=>showChartTooltip(event,`<strong>Индекс ${fmt(b.x0,0)}–${fmt(b.x1,0)}</strong>Субъектов: ${b.length}<br>${b.slice(0,6).map(d=>esc(d.region)).join("<br>")}${b.length>6?"<br>…":""}`)).on("mousemove",moveChartTooltip).on("mouseleave",hideChartTooltip);
 svg.append("g").selectAll("text").data(bins).join("text").attr("class","hist-value").attr("x",b=>(x(b.x0)+x(b.x1))/2).attr("y",b=>y(b.length)-6).text(b=>b.length);svg.append("g").attr("class","chart-axis").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).ticks(9));svg.append("g").attr("class","chart-axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(5));svg.append("text").attr("class","chart-axis-title").attr("x",(margin.left+width-margin.right)/2).attr("y",height-6).attr("text-anchor","middle").text("Индекс сложности поиска работы");svg.append("text").attr("class","chart-axis-title").attr("transform","rotate(-90)").attr("x",-(margin.top+height-margin.bottom)/2).attr("y",13).attr("text-anchor","middle").text("Количество субъектов");
}
function renderAnalytics(){renderSalaryChart();renderRelationshipChart();renderDistributionChart();}
const projection=d3.geoConicEqualArea().parallels([50,70]).rotate([-100,0]).fitExtent([[24,24],[WIDTH-24,HEIGHT-24]],geojson);
const path=d3.geoPath(projection);
const svg=d3.select(map).append("svg").attr("viewBox",`0 0 ${WIDTH} ${HEIGHT}`).attr("preserveAspectRatio","xMidYMid meet").attr("role","img").attr("aria-label","Карта сложности поиска приемлемо оплачиваемой работы по субъектам России");
svg.selectAll("path.region").data(geojson.features).join("path").attr("class","region").attr("d",path).attr("fill",f=>{const d=regionData[f.properties.shapeISO];return d?scoreColor(d.score):"#e5e8ee";}).attr("stroke","#fff").attr("stroke-width",.6).append("title").text(f=>{const iso=f.properties.shapeISO,d=regionData[iso],missing=noDataRegions[iso];return d?`${d.region}: ${d.score.toFixed(1)} из 10`:`${missing?.region||f.properties.shapeName}: нет полного набора данных для индекса`;});
const hoverOutline=svg.append("path").attr("class","region-outline").style("display","none");
const pinnedOutline=svg.append("path").attr("class","region-outline pinned").style("display","none");
function featureByIso(iso){return geojson.features.find(feature=>feature.properties.shapeISO===iso);}
function updateDirectoryActive(){regionGrid.querySelectorAll(".region-button").forEach(button=>button.classList.toggle("is-active",button.dataset.iso===pinnedIso));}
function renderNoDataRegion(d,isPinned=false){method.innerHTML=`<div><div class="eyebrow">Субъект без полного набора данных</div><div class="method-region">${esc(d.region)} · индекс не рассчитан</div></div><div><div class="method-title">Недостаточно сопоставимых официальных показателей</div><div class="method-text">${esc(d.message)}</div></div>`;panel.innerHTML=`<div class="eyebrow">Субъект РФ · ${isPinned?"закреплено":"наведение"}</div><div class="region-name">${esc(d.region)}</div><div class="overall"><span>Индекс сложности поиска работы</span><strong style="color:#7b8494">нет данных</strong></div><div class="official-grid"><div><span>Код региона</span><strong>${esc(d.region_code||"—")}</strong></div><div><span>Площадь</span><strong>${d.area_thousand_km2==null?"нет данных":`${fmt(d.area_thousand_km2,1)} тыс. км²`}</strong></div><div><span>Население</span><strong>${d.population_people==null?"нет данных":`${fmt(d.population_people)} чел.`}</strong></div></div><div class="rank-empty">${esc(d.message)}</div><div class="note">Серый цвет означает отсутствие полного набора исходных показателей, а не лёгкость или сложность трудоустройства.</div>`;}
function showIso(iso,isPinned=false){const d=regionData[iso],missing=noDataRegions[iso],feature=featureByIso(iso);if(!feature||(!d&&!missing))return;currentIso=iso;if(d){renderMethod(d);renderPanel(d,isPinned);renderAnalytics();}else{renderNoDataRegion(missing,isPinned);}hoverOutline.attr("d",path(feature)).style("display",isPinned?"none":null);updateDirectoryActive();}
function pinIso(iso){const feature=featureByIso(iso);if(!feature||(!regionData[iso]&&!noDataRegions[iso]))return;pinnedIso=iso;pinnedOutline.attr("d",path(feature)).style("display",null);hoverOutline.style("display","none");showIso(iso,true);}
function togglePin(iso){if(pinnedIso===iso){pinnedIso=null;pinnedOutline.style("display","none");showIso(iso,false);}else{pinIso(iso);}}
const directoryEntries=Object.entries({...noDataRegions,...regionData}),sortLabels={alphabetical:"По алфавиту",code:"По коду региона","population-desc":"Население, сначала больше","population-asc":"Население, сначала меньше","area-desc":"Площадь, сначала большие","area-asc":"Площадь, сначала малые","score-desc":"Трудоустройство, сначала труднее","score-asc":"Трудоустройство, сначала легче"};
let directorySort="alphabetical";
function sortedDirectoryEntries(){const entries=[...directoryEntries],byName=(a,b)=>a[1].region.localeCompare(b[1].region,"ru");return entries.sort((a,b)=>{if(directorySort==="code")return Number(a[1].region_code)-Number(b[1].region_code)||byName(a,b);if(directorySort==="population-desc")return b[1].population_people-a[1].population_people||byName(a,b);if(directorySort==="population-asc")return a[1].population_people-b[1].population_people||byName(a,b);if(directorySort==="area-desc")return b[1].area_thousand_km2-a[1].area_thousand_km2||byName(a,b);if(directorySort==="area-asc")return a[1].area_thousand_km2-b[1].area_thousand_km2||byName(a,b);if(directorySort==="score-desc")return b[1].score-a[1].score||byName(a,b);if(directorySort==="score-asc")return a[1].score-b[1].score||byName(a,b);return byName(a,b);});}
function fmtPopulation(value){return value>=1000000?`${fmt(value/1000000,2)} млн чел.`:`${fmt(value/1000,0)} тыс. чел.`;}
function renderDirectory(){const entries=sortedDirectoryEntries();regionGrid.innerHTML=entries.map(([iso,d])=>{const hasScore=Number.isFinite(d.score),meta=hasScore?`${fmtPopulation(d.population_people)} · ${fmt(d.area_thousand_km2,1)} тыс. км² · индекс ${d.score.toFixed(1)}`:`${d.population_people==null?"население: нет данных":fmtPopulation(d.population_people)} · индекс не рассчитан`;return `<button type="button" class="region-button${pinnedIso===iso?" is-active":""}${hasScore?"":" no-data"}" data-iso="${iso}" aria-label="Регион ${esc(d.region)}, код ${esc(d.region_code)}, ${hasScore?`индекс сложности ${d.score.toFixed(1)}`:"индекс не рассчитан"}"><span class="region-code">${esc(d.region_code)}</span><span class="region-button-name-wrap"><span class="region-button-name">${esc(d.region)}</span><span class="region-button-meta">${meta}</span></span></button>`;}).join("");regionGrid.querySelectorAll(".region-button").forEach(button=>{const iso=button.dataset.iso;button.addEventListener("mouseenter",()=>{const feature=featureByIso(iso);if(feature)hoverOutline.attr("d",path(feature)).style("display",null);if(!pinnedIso)showIso(iso,false);});button.addEventListener("mouseleave",()=>hoverOutline.style("display","none"));button.addEventListener("click",()=>togglePin(iso));});updateDirectoryActive();}
document.querySelectorAll(".sort-option").forEach(button=>button.addEventListener("click",()=>{directorySort=button.dataset.sort;sortLabel.textContent=sortLabels[directorySort];document.querySelectorAll(".sort-option").forEach(option=>option.classList.toggle("is-selected",option===button));renderDirectory();sortMenu.open=false;}));
document.querySelectorAll(".chart-control").forEach(button=>button.addEventListener("click",()=>{salaryView=button.dataset.salaryView;document.querySelectorAll(".chart-control").forEach(option=>option.classList.toggle("is-active",option===button));renderSalaryChart();}));
const renderForecastWithoutGrowth=renderForecast;renderForecast=function(){renderForecastWithoutGrowth();renderWageInflationChart(forecastYear);};renderForecast();renderCities();updateDownloads();addOccupationSalaryDownloads();activateEmbeddedDownloads();
renderDirectory();
svg.selectAll("path.region").on("mouseenter",function(event,feature){const iso=feature.properties.shapeISO;if(!regionData[iso]&&!noDataRegions[iso])return;hoverOutline.attr("d",path(feature)).style("display",pinnedIso?"none":null);if(!pinnedIso)showIso(iso,false);}).on("mouseleave",function(){if(!pinnedIso)hoverOutline.style("display","none");}).on("click",function(event,feature){const iso=feature.properties.shapeISO;if(regionData[iso]||noDataRegions[iso])togglePin(iso);});
indexHelpButton.addEventListener("click",()=>{indexGuide.scrollIntoView({behavior:"smooth",block:"start"});indexGuide.classList.remove("flash");void indexGuide.offsetWidth;indexGuide.classList.add("flash");});
let resizeTimer=null;window.addEventListener("resize",()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{renderAnalytics();renderMedianForecastChart(forecastYear);renderWageInflationChart(forecastYear);},120);});
const firstIso=Object.entries(regionData).sort((a,b)=>b[1].score-a[1].score)[0][0];showIso(firstIso,false);
</script></body></html>'''

embedded_download_paths = [
    GROUP_FORECAST_FILE, OCCUPATION_FORECAST_FILE, WAGE_INFLATION_FILE,
    REGIONAL_REPORT_FILE, PROFESSIONAL_REPORT_FILE, PROFESSIONAL_DEMAND_FILE,
    WAGE_VALIDATION_FILE, PROFESSIONAL_WAGE_FILE, SUBSISTENCE_MINIMUM_FILE,
    SOURCE_MANIFEST_FILE, CURRENT_LABOR_CONTEXT_FILE, CURRENT_LABOR_FORCE_FILE,
    CURRENT_UNEMPLOYMENT_FILE, MEDIAN_WAGE_HISTORY_FILE, ROSSTAT_CPI_FILE,
    ROSSTAT_CPI_HISTORY_FILE, ROSSTAT_CPI_MONTHLY_FILE,
    CBR_INFLATION_FORECAST_FILE, CBR_MEDIUM_TERM_FORECAST_FILE,
    CBR_MACRO_SURVEY_FILE, CBR_MACRO_SURVEY_PDF_FILE,
    OCCUPATION_SALARY_SNAPSHOT_FILE, DETAILED_WAGE_ARCHIVE_FILE,
]
mime_by_suffix = {
    ".csv": "text/csv;charset=utf-8",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".html": "text/html;charset=utf-8",
    ".pdf": "application/pdf",
}
embedded_downloads = {}
for download_path in embedded_download_paths:
    if download_path.exists():
        relative_name = download_path.relative_to(PROJECT_DIR).as_posix()
        encoded = base64.b64encode(download_path.read_bytes()).decode("ascii")
        embedded_downloads[relative_name] = (
            f"data:{mime_by_suffix.get(download_path.suffix.lower(), 'application/octet-stream')};"
            f"base64,{encoded}"
        )

map_html = (
    html_template
    .replace("__GEOJSON__", json.dumps(geojson, ensure_ascii=False, separators=(",", ":")))
    .replace("__REGION_DATA__", json.dumps(map_data, ensure_ascii=False, separators=(",", ":")))
    .replace("__NO_DATA_REGIONS__", json.dumps(no_data_regions, ensure_ascii=False, separators=(",", ":")))
    .replace("__OCCUPATION_CATALOG__", json.dumps(occupation_catalog, ensure_ascii=False, separators=(",", ":")))
    .replace("__FORECAST_BY_YEAR__", json.dumps(forecast_by_year, ensure_ascii=False, separators=(",", ":")))
    .replace("__GROUP_FORECAST_BY_YEAR__", json.dumps(group_forecast_by_year, ensure_ascii=False, separators=(",", ":")))
    .replace("__NATIONAL_MEDIAN_SERIES__", json.dumps(national_median_series.to_dict(orient="records"), ensure_ascii=False, separators=(",", ":")))
    .replace("__WAGE_INFLATION_SERIES__", json.dumps(wage_inflation_series.to_dict(orient="records"), ensure_ascii=False, separators=(",", ":")))
    .replace("__EMBEDDED_DOWNLOADS__", json.dumps(embedded_downloads, ensure_ascii=False, separators=(",", ":")))
    .replace("__CITY_JOB_EXPLANATIONS__", json.dumps(city_job_explanations, ensure_ascii=False, separators=(",", ":")))
    .replace("__FORECAST_META__", json.dumps({
        "months": wage_forecast_months,
        "mape_pct": round(wage_forecast_mape * 100, 2),
        "alpha": round(wage_ridge_alpha, 6),
        "cbr_survey_date": inflation_model_meta["publication_date"],
        "cbr_survey_dates": inflation_model_meta["survey_dates"],
        "cbr_respondents": inflation_model_meta["respondents"],
        "cbr_table_last_year": inflation_model_meta["table_last_year"],
        "years": list(FORECAST_YEARS),
    }, ensure_ascii=False, separators=(",", ":")))
    .replace("__VALIDATION_SUMMARY__", validation_summary)
)
MAP_FILE.write_text(map_html, encoding="utf-8")

# Режим оперативных данных сайтов вакансий хранится отдельно от Росстата и
# добавляется только при наличии проверенного локального среза.
online_regional_file = PROJECT_DIR / "online_job_market_by_region.csv"
if online_regional_file.exists():
    from inject_online_job_mode import inject_online_mode
    inject_online_mode(MAP_FILE)

# GitHub Pages открывает index.html, поэтому публичная копия всегда
# синхронизируется с основным HTML при каждой сборке.
(PROJECT_DIR / "index.html").write_text(MAP_FILE.read_text(encoding="utf-8"), encoding="utf-8")

print(f"Карта сохранена: {MAP_FILE}")
print(f"Регионов на карте: {len(map_data)}")
print(f"Профессиональных строк Росстата: {len(professional_export)}")
print(f"Детальных профессий в поиске: {len(occupation_catalog)}")
print(f"Рейтинг сохранён: {PROFESSIONAL_REPORT_FILE}")
