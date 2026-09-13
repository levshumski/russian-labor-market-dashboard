"""Собирает воспроизводимый срез вакансий из доступных официальных интерфейсов.

HH используется через публичные страницы поиска, потому что endpoint вакансий
api.hh.ru блокирует текущий адрес выполнения. SuperJob поддерживается через
официальный API при наличии SUPERJOB_API_KEY.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
REGIONS_FILE = ROOT / "rosstat_regions_for_map.csv"
VACANCIES_FILE = ROOT / "online_job_vacancies_combined.csv"
REGIONAL_FILE = ROOT / "online_job_market_by_region.csv"
GROUP_FILE = ROOT / "online_job_market_by_region_and_group.csv"
STATUS_FILE = ROOT / "online_job_sources_status.csv"
CACHE_DIR = ROOT / "online_job_cache"
CACHE_DIR.mkdir(exist_ok=True)

COLLECTED_AT = datetime.now().astimezone().isoformat(timespec="seconds")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
    )
}

GROUP_PATTERNS = [
    ("Информационные технологии", r"программист|разработчик|developer|python|java|frontend|backend|devops|data|аналитик данных|тестировщик|\bqa\b|системн(?:ый|ая) администратор"),
    ("Инженерия и производство", r"инженер|технолог|конструктор|механик|электрик|электромонт|сварщик|слесарь|токарь|фрезеровщик|оператор станка|производств"),
    ("Строительство и недвижимость", r"строител|прораб|сметчик|архитектор|монтажник|бетонщик|каменщик|недвижим|риелтор|риэлтор"),
    ("Транспорт и логистика", r"водитель|курьер|логист|кладовщик|комплектовщик|экспедитор|диспетчер|машинист"),
    ("Медицина и фармацевтика", r"врач|медицин|медсестр|фельдшер|фармацевт|провизор|стоматолог|ветеринар"),
    ("Образование и наука", r"учитель|преподаватель|педагог|воспитатель|научн|лаборант"),
    ("Продажи и клиентский сервис", r"продавец|кассир|продаж|клиент|официант|бариста|администратор|оператор call|менеджер.*магазин"),
    ("Финансы, бухгалтерия и право", r"бухгалтер|экономист|финанс|аудитор|юрист|налог|банк|кредит"),
    ("Маркетинг, медиа и дизайн", r"маркет|реклам|дизайнер|редактор|контент|smm|копирайтер|pr\b"),
    ("Сельское и лесное хозяйство", r"агроном|тракторист|фермер|животновод|дояр|леснич|лесозаготов|сельск"),
    ("Охрана и безопасность", r"охранник|безопасност|полицей|пожарн|контролер"),
    ("Рабочий и обслуживающий персонал", r"уборщик|разнорабоч|грузчик|дворник|санитар|повар|пекарь|швея|мастер"),
    ("Управление", r"директор|руководитель|начальник|управляющий|product manager|project manager"),
]

REGION_ALIASES = {
    "Кемеровская область - Кузбасс": "Кемеровская область",
    "Республика Северная Осетия — Алания": "Республика Северная Осетия-Алания",
    "Республика Северная Осетия - Алания": "Республика Северная Осетия-Алания",
    "Ханты-Мансийский АО - Югра": "Ханты-Мансийский автономный округ — Югра",
    "Ханты-Мансийский автономный округ - Югра": "Ханты-Мансийский автономный округ — Югра",
    "Ненецкий АО": "Ненецкий автономный округ",
    "Ямало-Ненецкий АО": "Ямало-Ненецкий автономный округ",
    "Чукотский АО": "Чукотский автономный округ",
    "Еврейская АО": "Еврейская автономная область",
}


def clean_url(value: str) -> str:
    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def classify_title(title: str) -> str:
    normalized = str(title).lower().replace("ё", "е")
    for label, pattern in GROUP_PATTERNS:
        if re.search(pattern, normalized, flags=re.I):
            return label
    return "Прочие профессии"


def hh_regions(session: requests.Session) -> dict[str, str]:
    response = session.get("https://api.hh.ru/areas", timeout=45)
    response.raise_for_status()
    root = next(item for item in response.json() if item["id"] == "113")
    lookup = {}
    for item in root["areas"]:
        name = REGION_ALIASES.get(item["name"], item["name"])
        lookup[name] = item["id"]
    return lookup


def parse_hh_card(card, requested_region: str) -> dict:
    title_node = card.select_one('[data-qa="serp-item__title"]')
    employer_node = card.select_one('[data-qa="vacancy-serp__vacancy-employer"]')
    address_node = card.select_one('[data-qa="vacancy-serp__vacancy-address"]')
    salary_values = []
    salary_currency = None
    for data in card.select("data[value]"):
        value = data.get("value", "")
        if value.isdigit():
            salary_values.append(float(value))
        elif value in {"RUB", "RUR", "USD", "EUR", "KZT", "BYR"}:
            salary_currency = "RUR" if value in {"RUB", "RUR"} else value
    salary_values = salary_values[:2]
    salary_from = salary_values[0] if salary_values else np.nan
    salary_to = salary_values[-1] if salary_values else np.nan
    salary_mid = np.nanmean([salary_from, salary_to]) if salary_values else np.nan
    experience = next((x.get_text(" ", strip=True) for x in card.select('[data-qa*="work-experience"]')), None)
    title = title_node.get_text(" ", strip=True) if title_node else ""
    return {
        "source": "hh.ru",
        "source_vacancy_id": card.select_one("[id]").get("id") if card.select_one("[id]") else None,
        "title": title,
        "professional_group": classify_title(title),
        "region": requested_region,
        "location_text": address_node.get_text(" ", strip=True) if address_node else requested_region,
        "employer": employer_node.get_text(" ", strip=True) if employer_node else None,
        "salary_from": salary_from,
        "salary_to": salary_to,
        "salary_currency": salary_currency,
        "salary_mid_rur": salary_mid if salary_currency == "RUR" else np.nan,
        "salary_gross": "до вычета налогов" in card.get_text(" ", strip=True).lower(),
        "experience": experience,
        "employment": None,
        "schedule": "Удалённая работа" if card.select_one('[data-qa="vacancy-label-work-schedule-remote"]') else None,
        "published_at": None,
        "collected_at": COLLECTED_AT,
        "url": clean_url(title_node.get("href")) if title_node and title_node.get("href") else None,
    }


def fetch_hh(session: requests.Session, regions: pd.DataFrame, pages: int = 2) -> tuple[list[dict], list[dict]]:
    area_ids = hh_regions(session)
    rows, totals = [], []
    for position, region in enumerate(regions.itertuples(index=False), 1):
        area_id = area_ids.get(region.subject)
        if not area_id:
            totals.append({"source": "hh.ru", "region": region.subject, "reported_total": np.nan, "status": "нет area_id"})
            continue
        reported_total = None
        for page in range(pages):
            cache = CACHE_DIR / f"hh_{area_id}_{page}.html"
            if cache.exists() and (time.time() - cache.stat().st_mtime) < 6 * 3600:
                html = cache.read_text(encoding="utf-8")
            else:
                response = session.get(
                    "https://hh.ru/search/vacancy",
                    params={"area": area_id, "page": page, "items_on_page": 20},
                    timeout=45,
                )
                if response.status_code != 200:
                    totals.append({"source": "hh.ru", "region": region.subject, "reported_total": np.nan, "status": f"HTTP {response.status_code}"})
                    break
                html = response.text
                cache.write_text(html, encoding="utf-8")
                time.sleep(0.65)
            soup = BeautifulSoup(html, "html.parser")
            if page == 0:
                heading = soup.select_one("h1")
                numbers = re.findall(r"\d+", heading.get_text("", strip=True).replace("\xa0", "")) if heading else []
                reported_total = int("".join(numbers)) if numbers else None
            cards = soup.select('[data-qa="vacancy-serp__vacancy"]')
            rows.extend(parse_hh_card(card, region.subject) for card in cards)
            if len(cards) < 10:
                break
        totals.append({"source": "hh.ru", "region": region.subject, "reported_total": reported_total, "status": "ok" if reported_total is not None else "не распознано"})
        print(f"HH {position:02d}/{len(regions)} {region.subject}: total={reported_total}, sample={sum(x['region']==region.subject for x in rows)}")
    return rows, totals


def fetch_superjob(session: requests.Session, regions: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    key = os.getenv("SUPERJOB_API_KEY")
    if not key:
        return [], [{"source": "superjob.ru", "region": "Россия", "reported_total": np.nan, "status": "нужен SUPERJOB_API_KEY"}]
    # Официальный API требует Secret key приложения. География API сопоставляется
    # текстом; строки всё равно проходят нормализацию и дедупликацию ниже.
    rows, totals = [], []
    headers = {**session.headers, "X-Api-App-Id": key}
    for region in regions.itertuples(index=False):
        response = session.get("https://api.superjob.ru/2.0/vacancies/", params={"town": region.subject, "count": 100}, headers=headers, timeout=45)
        if response.status_code != 200:
            totals.append({"source": "superjob.ru", "region": region.subject, "reported_total": np.nan, "status": f"HTTP {response.status_code}"})
            continue
        payload = response.json()
        totals.append({"source": "superjob.ru", "region": region.subject, "reported_total": payload.get("total"), "status": "ok"})
        for item in payload.get("objects", []):
            title = item.get("profession") or ""
            lo, hi = item.get("payment_from") or np.nan, item.get("payment_to") or np.nan
            if pd.isna(lo): lo = hi
            if pd.isna(hi): hi = lo
            currency = item.get("currency")
            rows.append({"source":"superjob.ru","source_vacancy_id":item.get("id"),"title":title,"professional_group":classify_title(title),"region":region.subject,"location_text":(item.get("town") or {}).get("title"),"employer":item.get("firm_name"),"salary_from":lo,"salary_to":hi,"salary_currency":currency,"salary_mid_rur":np.nanmean([lo,hi]) if currency=="rub" and not pd.isna(lo) else np.nan,"salary_gross":None,"experience":(item.get("experience") or {}).get("title"),"employment":(item.get("type_of_work") or {}).get("title"),"schedule":(item.get("place_of_work") or {}).get("title"),"published_at":datetime.fromtimestamp(item["date_published"]).astimezone().isoformat() if item.get("date_published") else None,"collected_at":COLLECTED_AT,"url":item.get("link")})
        time.sleep(0.65)
    return rows, totals


def normalize_for_dedup(value) -> str:
    return re.sub(r"[^a-zа-я0-9]+", " ", str(value).lower().replace("ё", "е")).strip()


def build_outputs() -> None:
    regions = pd.read_csv(REGIONS_FILE, encoding="utf-8-sig").rename(columns={"Субъект_РФ": "subject", "Код_ISO_субъекта": "shapeISO"})
    session = requests.Session()
    session.headers.update(HEADERS)
    hh_rows, hh_totals = fetch_hh(session, regions, pages=2)
    sj_rows, sj_totals = fetch_superjob(session, regions)
    vacancies = pd.DataFrame(hh_rows + sj_rows)
    for column in ["title", "employer", "region"]:
        vacancies[f"_{column}"] = vacancies[column].map(normalize_for_dedup)
    vacancies["_salary"] = vacancies["salary_mid_rur"].round(-3)
    vacancies["duplicate_group"] = vacancies.groupby(["_title", "_employer", "_region", "_salary"], dropna=False).ngroup()
    vacancies["is_cross_platform_duplicate"] = vacancies.duplicated("duplicate_group", keep=False)
    vacancies = vacancies.drop(columns=["_title", "_employer", "_region", "_salary"])
    vacancies.to_csv(VACANCIES_FILE, index=False, encoding="utf-8-sig")

    totals = pd.DataFrame(hh_totals + sj_totals)
    totals.to_csv(STATUS_FILE, index=False, encoding="utf-8-sig")
    ok_totals = totals.loc[totals.status == "ok", ["source", "region", "reported_total"]].copy()
    totals_frame = ok_totals.groupby("region", as_index=False)["reported_total"].sum(min_count=1)
    sources_frame = ok_totals.groupby("region")["source"].agg(lambda x: ", ".join(sorted(set(x)))).rename("sources_available").reset_index()
    salary_summary = vacancies.groupby("region", as_index=False).agg(sample_vacancies=("title","size"),salary_observations=("salary_mid_rur","count"),median_salary_offer_rur=("salary_mid_rur","median"),salary_q25_rur=("salary_mid_rur",lambda x:x.quantile(.25)),salary_q75_rur=("salary_mid_rur",lambda x:x.quantile(.75)))
    regional = regions[["subject","shapeISO"]].rename(columns={"subject":"region"}).merge(totals_frame,on="region",how="left").merge(sources_frame,on="region",how="left").merge(salary_summary,on="region",how="left")
    regional["snapshot_at"] = COLLECTED_AT
    regional.to_csv(REGIONAL_FILE,index=False,encoding="utf-8-sig")
    groups = vacancies.groupby(["region","professional_group"],as_index=False).agg(sample_vacancies=("title","size"),salary_observations=("salary_mid_rur","count"),median_salary_offer_rur=("salary_mid_rur","median"))
    groups["sample_share_pct"] = groups["sample_vacancies"] / groups.groupby("region")["sample_vacancies"].transform("sum") * 100
    groups.to_csv(GROUP_FILE,index=False,encoding="utf-8-sig")
    print(json.dumps({"vacancies":len(vacancies),"regions_with_totals":int(regional.reported_total.notna().sum()),"salary_observations":int(vacancies.salary_mid_rur.notna().sum()),"sources":totals.groupby('source').status.first().to_dict()},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    build_outputs()
