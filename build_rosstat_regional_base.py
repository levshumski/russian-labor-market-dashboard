from pathlib import Path
import re

import pandas as pd


PROJECT_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd().resolve()
)

FILES = {
    "unemployment": PROJECT_DIR / "Unemployment(15-72yo).xlsx",
    "employment": PROJECT_DIR / "Chislennost_rabochey_sili_i_yroven_zanatosti_15-72yo.xlsx",
    "labor_force": PROJECT_DIR / "Chislennost_rabochey_sili_15-72yo.xlsx",
    "qualification": PROJECT_DIR / "Chislennost_rabochey_sili_15+yo.xlsx",
}
MAP_CSV = PROJECT_DIR / "rosstat_regions_for_map.csv"
CONTEXT_CSV = PROJECT_DIR / "rosstat_regional_context_2026.csv"
SEVASTOPOL_LABOR_FILE = (
    PROJECT_DIR / "official_sources" / "Sevastopol_labor_force_2026.xlsx"
)


def normalize_region(value):
    name = re.sub(r"\s+", " ", str(value)).strip()
    aliases = {
        "г. Москва": "Москва", "г.Москва": "Москва",
        "г. Санкт-Петербург": "Санкт-Петербург", "г.Санкт-Петербург": "Санкт-Петербург",
        "г. Севастополь": "Севастополь", "г.Севастополь": "Севастополь",
        "Республика Северная Осетия - Алания": "Республика Северная Осетия-Алания",
        "Еврейская авт.область": "Еврейская автономная область",
        "Чукотский авт.округ": "Чукотский автономный округ",
        "Ямало-Ненецкий авт.округ": "Ямало-Ненецкий автономный округ",
        "в том числе: Ханты-Мансийский авт.округ": "Ханты-Мансийский автономный округ — Югра",
        "в том числе: Ханты-Мансийский автономный округ - Югра": "Ханты-Мансийский автономный округ — Югра",
        "в том числе: Ненецкий авт.округ": "Ненецкий автономный округ",
        "в том числе: Ненецкий автономный округ": "Ненецкий автономный округ",
        "в том числе Ненецкий автономный округ": "Ненецкий автономный округ",
    }
    return aliases.get(name, name)


def latest_indicator(file_path, sheet_name, value_name):
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    period_columns = raw.iloc[4][raw.iloc[4].notna()].index
    if len(period_columns) == 0:
        raise ValueError(f"Не найдены периоды: {file_path.name}, лист {sheet_name}")
    value_column = int(period_columns[-1])
    period = str(raw.iloc[4, value_column]).replace("\n", " ").strip()
    result = raw.iloc[5:, [0, value_column]].copy()
    result.columns = ["subject", value_name]
    result["subject"] = result["subject"].map(normalize_region)
    result[value_name] = pd.to_numeric(result[value_name], errors="coerce")
    result = result.dropna(subset=[value_name]).drop_duplicates("subject", keep="last")
    return result, period


for source_file in FILES.values():
    if not source_file.exists():
        raise FileNotFoundError(f"Не найден официальный файл Росстата: {source_file.name}")
if not MAP_CSV.exists():
    raise FileNotFoundError(
        "Не найден rosstat_regions_for_map.csv с таблицей соответствия ISO-кодов геометрии."
    )

# Из существующего CSV берём только постоянное соответствие ISO-кода и названия субъекта.
crosswalk = pd.read_csv(MAP_CSV, usecols=["Код_ISO_субъекта", "Субъект_РФ"])
crosswalk.columns = ["shapeISO", "subject"]
crosswalk["subject"] = crosswalk["subject"].map(normalize_region)
if "Севастополь" not in set(crosswalk["subject"]):
    crosswalk.loc[len(crosswalk)] = ["RU-SEV", "Севастополь"]

unemployment, unemployment_period = latest_indicator(FILES["unemployment"], "4", "unemployment_rate")
employment, employment_period = latest_indicator(FILES["employment"], "4", "employment_rate")
labor_force, labor_force_period = latest_indicator(FILES["labor_force"], "3", "labor_force_thousand")
participation, participation_period = latest_indicator(FILES["labor_force"], "4", "labor_force_participation_rate")
qualification, qualification_period = latest_indicator(FILES["qualification"], "7", "qualified_labor_force_share")

official = crosswalk.copy()
for frame in [unemployment, employment, labor_force, participation, qualification]:
    official = official.merge(frame, on="subject", how="left")

# Центральные таблицы пока не включают Севастополь. Берём опубликованный
# Крымстатом (территориальным органом Росстата) последний срез апрель–июнь 2026.
if not SEVASTOPOL_LABOR_FILE.exists():
    raise FileNotFoundError(f"Не найден официальный файл: {SEVASTOPOL_LABOR_FILE}")
sev_raw = pd.read_excel(SEVASTOPOL_LABOR_FILE, header=None)
sev_numeric = sev_raw.iloc[:, 1:7].apply(pd.to_numeric, errors="coerce")
sev_rows = sev_numeric.loc[sev_numeric.iloc[:, :3].notna().all(axis=1)]
sev_latest = sev_rows.iloc[-1]
sev_mask = official["subject"].eq("Севастополь")
official.loc[sev_mask, [
    "labor_force_thousand", "employment_rate", "unemployment_rate",
    "labor_force_participation_rate",
]] = [float(sev_latest.iloc[0]), float(sev_latest.iloc[4]),
      float(sev_latest.iloc[5]), float(sev_latest.iloc[3])]
# В территориальном оперативном файле нет сопоставимой доли образования.
official.loc[sev_mask, "qualified_labor_force_share"] = pd.NA

missing = official.loc[
    official[["unemployment_rate", "employment_rate", "labor_force_thousand"]].isna().any(axis=1),
    "subject",
]
if not missing.empty:
    raise ValueError("Не сопоставлены субъекты: " + ", ".join(missing.astype(str)))

context_export = official[[
    "shapeISO", "subject", "unemployment_rate", "employment_rate",
    "labor_force_thousand", "labor_force_participation_rate",
    "qualified_labor_force_share",
]].rename(columns={
    "shapeISO": "Код_ISO_субъекта",
    "subject": "Субъект_РФ",
    "unemployment_rate": "Уровень_безработицы_pct",
    "employment_rate": "Уровень_занятости_pct",
    "labor_force_thousand": "Рабочая_сила_тыс_человек",
    "labor_force_participation_rate": "Участие_в_рабочей_силе_pct",
    "qualified_labor_force_share": "Профильное_или_высшее_образование_pct",
})
context_export.insert(2, "Период", "март–май 2026")
context_export.loc[
    context_export["Субъект_РФ"].eq("Севастополь"), "Период"
] = "апрель–июнь 2026"
context_export.to_csv(CONTEXT_CSV, index=False, encoding="utf-8-sig")
context_export.to_csv(MAP_CSV, index=False, encoding="utf-8-sig")

print(f"Справочный региональный контекст Росстата обновлён: {len(context_export)} субъектов")
print(f"Период безработицы: {unemployment_period}")
print(f"Период занятости: {employment_period}")
