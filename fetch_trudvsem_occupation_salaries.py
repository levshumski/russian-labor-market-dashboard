from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
import math
import re
import time

import numpy as np
import pandas as pd
import requests


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = PROJECT_DIR / "trudvsem_top_occupation_salary_snapshot_2026.csv"
API_URL = "https://opendata.trudvsem.ru/api/v1/vacancies"
MAX_PAGES = 12
PAGE_SIZE = 100


OCCUPATIONS = [
    {
        "occupation": "Врачи скорой медицинской помощи и парамедики",
        "query": "врач скорой медицинской помощи",
        "pattern": r"врач.*скорой.*медицинской.*помощи|парамедик",
        "fallback_57t_2025": 136745,
        "fallback_level": "точная составная группа ОКЗ, 57-Т",
    },
    {
        "occupation": "Провизоры",
        "query": "провизор",
        "pattern": r"\bпровизор",
        "fallback_57t_2025": 87669,
        "fallback_level": "укрупнённая группа других специалистов здравоохранения, 57-Т",
    },
    {
        "occupation": "Диетологи и специалисты по рациональному питанию",
        "query": "диетолог",
        "pattern": r"диетолог|врач.*по.*питани",
        "fallback_57t_2025": 87669,
        "fallback_level": "укрупнённая группа других специалистов здравоохранения, 57-Т",
    },
    {
        "occupation": "Врачи общей практики",
        "query": "врач общей практики",
        "pattern": r"врач.*общей практики|семейный врач",
        "fallback_57t_2025": 120041,
        "fallback_level": "составная группа «Врачи», 57-Т",
    },
    {
        "occupation": "Преподаватели иностранных языков на курсах и частные",
        "query": "преподаватель иностранного языка",
        "pattern": r"преподавател.*(иностран|англий|немец|француз|китай|испан)",
        "fallback_57t_2025": 72647,
        "fallback_level": "составная группа других специалистов образования, 57-Т",
    },
    {
        "occupation": "Физиотерапевты",
        "query": "физиотерапевт",
        "pattern": r"физиотерапевт",
        "fallback_57t_2025": 87669,
        "fallback_level": "укрупнённая группа других специалистов здравоохранения, 57-Т",
    },
    {
        "occupation": "Специалисты в области медицинских аспектов охраны труда и окружающей среды",
        "query": "специалист по охране труда",
        "pattern": r"специалист.*охран.*труда|инженер.*охран.*труда|специалист.*промышленной.*безопасности",
        "fallback_57t_2025": 87669,
        "fallback_level": "укрупнённая группа других специалистов здравоохранения, 57-Т",
    },
    {
        "occupation": "Ветеринарные врачи",
        "query": "ветеринарный врач",
        "pattern": r"ветеринар.*врач|врач.*ветеринар",
        "fallback_57t_2025": 75830,
        "fallback_level": "точная составная группа ОКЗ, 57-Т",
    },
    {
        "occupation": "Специалисты по сбыту продукции (исключая информационно-коммуникационные технологии)",
        "query": "специалист по сбыту",
        "pattern": r"специалист.*сбыт|менеджер.*сбыт|специалист.*продаж",
        "fallback_57t_2025": 130649,
        "fallback_level": "составная группа сбыта и маркетинга, 57-Т",
    },
    {
        "occupation": "Операторы машин по производству текстильной, меховой и кожаной продукции",
        "query": "оператор текстильного производства",
        "pattern": r"оператор.*(текстил|швей|ткац|кож|мех)|оператор.*вязаль",
        "fallback_57t_2025": 55382,
        "fallback_level": "точная составная группа ОКЗ, 57-Т",
    },
]


def get_json(params: dict) -> dict:
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0 labor-market-research/1.0"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as error:
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Ошибка API после трёх попыток: {last_error}")


def fetch_page(query: str, offset: int) -> dict:
    return get_json({"text": query, "limit": PAGE_SIZE, "offset": offset})


def salary_midpoint(vacancy: dict) -> float | None:
    if str(vacancy.get("currency", "")).lower() not in {"", "«руб.»", "руб.", "руб", "rur"}:
        return None
    salary_min = pd.to_numeric(vacancy.get("salary_min"), errors="coerce")
    salary_max = pd.to_numeric(vacancy.get("salary_max"), errors="coerce")
    values = [float(value) for value in (salary_min, salary_max) if pd.notna(value) and value > 0]
    if not values:
        return None
    return float(np.mean(values))


def collect_occupation(config: dict) -> dict:
    first = fetch_page(config["query"], 0)
    total = int(first.get("meta", {}).get("total", 0))
    pages = min(MAX_PAGES, max(1, math.ceil(total / PAGE_SIZE)))
    payloads = [first]
    if pages > 1:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(fetch_page, config["query"], offset): offset
                for offset in range(1, pages)
            }
            for future in as_completed(futures):
                payloads.append(future.result())

    pattern = re.compile(config["pattern"], flags=re.IGNORECASE)
    seen = set()
    salaries = []
    matched_vacancies = 0
    for payload in payloads:
        for wrapper in payload.get("results", {}).get("vacancies", []):
            vacancy = wrapper.get("vacancy", {})
            vacancy_id = vacancy.get("id") or vacancy.get("vac_url")
            if vacancy_id in seen:
                continue
            seen.add(vacancy_id)
            if not pattern.search(str(vacancy.get("job-name", ""))):
                continue
            matched_vacancies += 1
            midpoint = salary_midpoint(vacancy)
            if midpoint is not None:
                salaries.append(midpoint)

    use_api = len(salaries) >= 8
    median_salary = float(np.median(salaries)) if use_api else float(config["fallback_57t_2025"])
    return {
        "Профессия": config["occupation"],
        "Поисковый_запрос": config["query"],
        "Найдено_API_всего": total,
        "Проверено_вакансий": len(seen),
        "Совпало_по_названию": matched_vacancies,
        "Наблюдений_с_зарплатой": len(salaries),
        "Медианная_предлагаемая_зарплата_2026_руб": round(median_salary),
        "Источник_зарплаты": (
            "Медиана середины зарплатных вилок вакансий API «Работа России»"
            if use_api else f"Росстат 57-Т за октябрь 2025: {config['fallback_level']}"
        ),
        "Дата_среза": date.today().isoformat(),
        "URL": f"{API_URL}?text={requests.utils.quote(config['query'])}",
    }


def main() -> None:
    rows = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(collect_occupation, config): config for config in OCCUPATIONS}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
    order = {item["occupation"]: index for index, item in enumerate(OCCUPATIONS)}
    rows.sort(key=lambda row: order[row["Профессия"]])
    for row in rows:
        print(
            f"{row['Профессия']}: {row['Медианная_предлагаемая_зарплата_2026_руб']:,} руб.; "
            f"наблюдений {row['Наблюдений_с_зарплатой']}"
        )
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Сохранено: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
