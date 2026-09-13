"""Добавляет в готовый дашборд отдельный режим данных сайтов вакансий."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REGIONAL = ROOT / "online_job_market_by_region.csv"
GROUPS = ROOT / "online_job_market_by_region_and_group.csv"
VACANCIES = ROOT / "online_job_vacancies_combined.csv"
ROSSTAT = ROOT / "rosstat_regional_job_search_difficulty.csv"
STATUS = ROOT / "online_job_sources_status.csv"


def percentile(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(pct=True, method="average", ascending=ascending).fillna(0.5)


def safe_int(value, default: int = 0) -> int:
    """Convert numeric CSV values without failing on an empty regional sample."""
    return default if pd.isna(value) else int(round(float(value)))


def prepare_online_data():
    online = pd.read_csv(REGIONAL, encoding="utf-8-sig")
    groups = pd.read_csv(GROUPS, encoding="utf-8-sig")
    vacancies = pd.read_csv(VACANCIES, encoding="utf-8-sig")
    official = pd.read_csv(ROSSTAT, encoding="utf-8-sig").rename(columns={
        "Код_ISO_субъекта": "shapeISO",
        "Субъект_РФ": "region",
        "Рабочая_сила_окт_дек_2024_тыс": "labor_force_thousand",
        "Медианная_зарплата_апрель_2025_руб": "official_median_wage",
        "Прожиточный_минимум_трудоспособных_2025_руб": "subsistence",
        "Индекс_сложности_трудоустройства_1_10": "official_difficulty_score",
    })
    current_labor = pd.read_csv(ROOT / "rosstat_regions_for_map.csv", encoding="utf-8-sig").rename(columns={
        "Код_ISO_субъекта": "shapeISO",
        "Субъект_РФ": "region",
        "Рабочая_сила_тыс_человек": "labor_force_current_thousand",
    })
    online = online.merge(official[["shapeISO", "region", "labor_force_thousand", "official_median_wage", "subsistence", "official_difficulty_score"]], on=["shapeISO", "region"], how="left")
    online = online.merge(current_labor[["shapeISO", "region", "labor_force_current_thousand"]], on=["shapeISO", "region"], how="left")
    online["labor_force_thousand"] = online["labor_force_current_thousand"].combine_first(online["labor_force_thousand"])
    online["listings_per_10k_labor_force"] = online["reported_total"] / (online["labor_force_thousand"] * 1000) * 10000
    online["salary_ratio_to_official_median"] = online["median_salary_offer_rur"] / online["official_median_wage"]
    online["availability_difficulty"] = percentile(online["listings_per_10k_labor_force"], ascending=False)
    online["pay_difficulty"] = percentile(online["salary_ratio_to_official_median"], ascending=False)
    online["online_difficulty_score"] = (1 + 9 * (0.65 * online["availability_difficulty"] + 0.35 * online["pay_difficulty"])).round(2)
    online["salary_coverage_pct"] = online["salary_observations"] / online["sample_vacancies"] * 100

    national_group = groups.groupby("professional_group", as_index=False).agg(national_count=("sample_vacancies", "sum"), national_salary=("median_salary_offer_rur", "median"))
    national_group["national_share"] = national_group["national_count"] / national_group["national_count"].sum()
    groups = groups.merge(national_group, on="professional_group", how="left")
    groups = groups.merge(online[["region", "official_median_wage"]], on="region", how="left")
    region_sample = groups.groupby("region")["sample_vacancies"].transform("sum")
    alpha = 10.0
    groups["smoothed_share"] = (groups["sample_vacancies"] + alpha * groups["national_share"]) / (region_sample + alpha)
    groups["salary_ratio"] = groups["median_salary_offer_rur"] / groups["official_median_wage"]
    groups["scarcity_signal"] = groups.groupby("region")["smoothed_share"].rank(pct=True, ascending=False)
    groups["low_pay_signal"] = groups.groupby("region")["salary_ratio"].rank(pct=True, ascending=False).fillna(0.5)
    groups["difficulty_score"] = (1 + 9 * (0.7 * groups["scarcity_signal"] + 0.3 * groups["low_pay_signal"])).round(2)
    groups["rank"] = groups.groupby("region")["difficulty_score"].rank(method="first", ascending=False).astype(int)

    examples = vacancies.sort_values(["region", "professional_group", "salary_mid_rur"], ascending=[True, True, False]).groupby(["region", "professional_group"], as_index=False).head(3)
    example_lookup = {(r, g): frame[["title", "salary_mid_rur", "url", "source"]].replace({np.nan: None}).to_dict("records") for (r, g), frame in examples.groupby(["region", "professional_group"])}
    group_lookup = {}
    for region, frame in groups.groupby("region"):
        rows = []
        for row in frame.sort_values("rank").itertuples(index=False):
            rows.append({"name": row.professional_group, "rank": int(row.rank), "score": float(row.difficulty_score), "sample": int(row.sample_vacancies), "salary_n": int(row.salary_observations), "median_salary": None if pd.isna(row.median_salary_offer_rur) else int(round(row.median_salary_offer_rur)), "examples": example_lookup.get((region, row.professional_group), [])})
        group_lookup[region] = rows

    payload = {}
    for row in online.dropna(subset=["reported_total", "online_difficulty_score"]).itertuples(index=False):
        payload[row.shapeISO] = {
            "region": row.region,
            "score": float(row.online_difficulty_score),
            "reported_total": int(row.reported_total),
            "sample": safe_int(row.sample_vacancies),
            "salary_n": safe_int(row.salary_observations),
            "median_salary": None if pd.isna(row.median_salary_offer_rur) else int(round(row.median_salary_offer_rur)),
            "q25": None if pd.isna(row.salary_q25_rur) else int(round(row.salary_q25_rur)),
            "q75": None if pd.isna(row.salary_q75_rur) else int(round(row.salary_q75_rur)),
            "salary_coverage": round(float(row.salary_coverage_pct), 1),
            "listings_per_10k": round(float(row.listings_per_10k_labor_force), 1),
            "salary_ratio": round(float(row.salary_ratio_to_official_median), 2),
            "official_median": int(round(row.official_median_wage)),
            "official_score": None if pd.isna(row.official_difficulty_score) else round(float(row.official_difficulty_score), 2),
            "snapshot_at": row.snapshot_at,
            "groups": group_lookup.get(row.region, []),
        }
    raw_status = pd.read_csv(STATUS, encoding="utf-8-sig")
    status = []
    raw_status = raw_status[raw_status["source"].isin(["hh.ru", "superjob.ru"])]
    for source, frame in raw_status.groupby("source", sort=False):
        ok = int(frame["status"].eq("ok").sum())
        if source == "hh.ru":
            message = f"Данные получены: {ok} из {len(frame)} регионов"
            state = "ok" if ok else "недоступно"
        else:
            raw_message = str(frame.iloc[0]["status"])
            state = "ok" if raw_message == "ok" else "ограничено"
            message = f"Данные получены: {ok} из {len(frame)} регионов" if state == "ok" else "Источник временно недоступен"
        status.append({"source": source, "status": state, "message": message, "regions": ok})
    return payload, status


ONLINE_CSS = r'''
<style id="online-mode-style">
.data-mode-switch{display:grid;grid-template-columns:repeat(2,minmax(220px,360px));gap:10px;align-items:stretch;margin:4px 0 16px}.data-mode-switch button{position:relative;text-align:left;border:1px solid #cbd6e5;border-radius:14px;background:#fff;color:#274060;padding:13px 16px 13px 44px;font:inherit;font-size:12px;font-weight:800;cursor:pointer;box-shadow:0 4px 14px rgba(28,52,90,.05);transition:.18s ease}.data-mode-switch button:before{content:'';position:absolute;left:15px;top:50%;width:16px;height:16px;border-radius:50%;border:2px solid #9fb1c9;transform:translateY(-50%)}.data-mode-switch button:hover{border-color:#5d9de6;background:#f6faff;transform:translateY(-1px)}.data-mode-switch button.active{background:linear-gradient(135deg,#0f5fbe,#237edc);color:#fff;border-color:#1769d2;box-shadow:0 7px 20px rgba(23,105,210,.22)}.data-mode-switch button.active:before{border-color:#fff;background:#fff;box-shadow:inset 0 0 0 4px #1769d2}.online-dashboard{display:none}.online-mode .official-dashboard{display:none}.online-mode .online-dashboard{display:block}.online-grid{display:grid;grid-template-columns:minmax(0,2.25fr) minmax(330px,.95fr);gap:14px}.online-map-wrap,.online-panel,.online-section{background:#fff;border:1px solid #d7dee9;border-radius:16px;box-shadow:0 8px 28px rgba(27,42,78,.07)}.online-map-wrap{height:620px;overflow:hidden}.online-map-wrap svg{width:100%;height:620px;display:block}.online-panel{height:620px;overflow:auto;padding:18px}.online-section{padding:18px;margin-top:14px}.online-metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.online-metrics>div{background:#f6f8fb;border-radius:10px;padding:9px}.online-metrics span{display:block;color:#667085;font-size:10px}.online-metrics strong{font-size:14px}.online-rank{display:grid;grid-template-columns:24px minmax(0,1fr) 45px;gap:8px;align-items:center;border-bottom:1px solid #edf0f5;padding:9px 2px}.online-rank small{display:block;color:#667085;margin-top:2px}.online-example{margin:0 0 8px 30px;padding:8px 10px;background:#f4f8ff;border-left:3px solid #6ca9ea;font-size:11px}.online-example a{color:#1769d2}.source-status{display:grid;grid-template-columns:repeat(2,minmax(240px,1fr));gap:12px;margin:12px 0}.source-status>div{position:relative;padding:14px 16px 14px 44px;background:linear-gradient(145deg,#f8fbff,#f3f6fb);border:1px solid #dce5f1;border-radius:14px;font-size:12px;min-height:62px}.source-status>div:before{content:'';position:absolute;left:16px;top:17px;width:15px;height:15px;border-radius:50%;background:#d29b31;box-shadow:0 0 0 5px rgba(210,155,49,.12)}.source-status>div.ok:before{background:#27a45b;box-shadow:0 0 0 5px rgba(39,164,91,.12)}.source-status b{font-size:14px}.source-ok{color:#19713b}.source-missing{color:#8a5a16}.online-chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.online-compare-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.online-chart{width:100%;min-height:410px}.online-chart.compact{min-height:360px}.online-note{font-size:11px;line-height:1.5;color:#667085}.online-downloads{display:flex;flex-wrap:wrap;gap:8px}.online-downloads a{color:#1769d2}.online-region{cursor:pointer;vector-effect:non-scaling-stroke}.online-region:hover{stroke:#1769d2;stroke-width:2px}.online-region.selected{stroke:#0b4f9c;stroke-width:3px}.online-tooltip{position:fixed;display:none;pointer-events:none;z-index:100;background:#fff;border:1px solid #ccd6e3;border-radius:8px;padding:8px 10px;box-shadow:0 7px 20px rgba(0,0,0,.15);font-size:11px}
.online-method{margin-bottom:14px}.online-grid{grid-template-columns:minmax(0,2.35fr) minmax(330px,.9fr);align-items:start}.online-map-wrap{position:relative}.online-panel{min-height:620px;overflow-y:auto;scrollbar-gutter:stable}.online-section{background:var(--panel);border-color:var(--line);box-shadow:0 8px 28px rgba(27,42,78,.06)}.online-section>h3{font-size:20px;margin:0 0 5px}.online-section>p{max-width:1120px}.online-metrics{margin-bottom:13px}.online-metrics>div{background:#f7f8fb;padding:8px 10px}.online-metrics span{color:var(--muted)}.online-metrics strong{display:block;margin-top:2px}.online-rank{grid-template-columns:22px minmax(0,1fr) 46px;padding:8px 3px}.online-rank small{font-size:10px;margin-top:3px}.online-example{margin:0 3px 8px 30px;padding:9px 10px;background:#f5f9ff;border-left-color:#7ebaf4;line-height:1.45}.source-status{gap:9px}.source-status>div{padding:11px 13px 11px 39px;background:#fafcff;border-color:#dfe6ef;border-radius:12px;font-size:11px;min-height:56px}.source-status>div:before{left:14px;top:16px;width:12px;height:12px;box-shadow:0 0 0 4px rgba(210,155,49,.12)}.source-status>div.ok:before{box-shadow:0 0 0 4px rgba(39,164,91,.12)}.source-status b{font-size:13px}.online-compare-grid h4,.online-chart-grid h4{font-size:15px;margin:12px 0 3px}.online-note{line-height:1.45;color:var(--muted)}.online-downloads{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin-top:12px}.online-downloads a{display:block;padding:11px;border:1px solid #dce4ef;border-radius:10px;background:#fafcff;color:#0b5fba;text-decoration:none;font-size:11px;line-height:1.35}.online-downloads a:hover{background:#eaf4ff;border-color:#62a5ed}
@media(max-width:1050px){.online-compare-grid{grid-template-columns:1fr 1fr}}@media(max-width:900px){.online-grid,.online-chart-grid,.online-compare-grid,.data-mode-switch{grid-template-columns:1fr}.online-panel{height:auto}.source-status{grid-template-columns:1fr}.online-map-wrap{height:470px}.online-map-wrap svg{height:470px}}
</style>
'''

ONLINE_HTML = r'''
<section id="online-dashboard" class="online-dashboard" aria-live="polite">
  <section class="method online-method" aria-live="polite"><div><div class="eyebrow">ТЕКУЩИЕ ВАКАНСИИ</div><div class="method-region" id="online-method-region">Выберите регион</div></div><div><div class="method-title" id="online-method-title">Доступность вакансий с типичным для региона уровнем оплаты</div><div class="method-text" id="online-method-text">Оценка учитывает число опубликованных вакансий в расчёте на рабочую силу и сопоставляет предлагаемую оплату с медианной заработной платой региона.</div></div></section>
  <section class="online-section" style="margin-top:0"><h3>Данные сервисов поиска работы</h3><p class="section-lead">Оперативный срез открытых объявлений hh.ru и SuperJob. Он дополняет официальную статистику и показывает текущие предложения работодателей.</p><div class="source-status" id="online-source-status"></div></section>
  <div class="online-grid" style="margin-top:14px"><div class="online-map-wrap" id="online-map"></div><aside class="online-panel" id="online-panel"></aside></div>
  <section class="online-section"><h3>Сравнение официальной статистики и онлайн-объявлений</h3><p class="online-note">Графики сопоставляют последний официальный срез с актуальными объявлениями hh.ru и SuperJob. Различия отражают не только изменение рынка, но и разный охват источников.</p><div class="online-compare-grid"><div><h4>Где оценки расходятся сильнее</h4><div class="online-chart compact" id="online-gap-chart"></div></div><div><h4>Зарплата в объявлениях к медиане Росстата</h4><div class="online-chart compact" id="online-salary-ratio-chart"></div></div><div><h4>Покрытие регионов источниками</h4><div class="online-chart compact" id="online-source-chart"></div></div></div></section>
  <section class="online-section"><h3>Регионы и текущие объявления</h3><div class="online-chart-grid"><div><h4>Медианная зарплата в выборке</h4><div class="online-chart" id="online-salary-chart"></div></div><div><h4>Доступность объявлений и зарплата</h4><div class="online-chart" id="online-scatter-chart"></div></div></div></section>
  <section class="online-section"><h3>Как понимать оценку онлайн-вакансий</h3><p class="online-note">Высокий балл означает, что объявлений относительно мало и/или предлагаемая зарплата ниже типичной зарплаты региона. Оценка нужна для сравнения регионов и не показывает личную вероятность получить работу. Рейтинг профессий строится по доступной выборке объявлений, поэтому отражает общую картину, но не охватывает абсолютно все вакансии.</p><div class="online-downloads"><a href="online_job_vacancies_combined.csv" download>Вакансии, CSV</a><a href="online_job_market_by_region.csv" download>Показатели регионов, CSV</a><a href="online_job_market_by_region_and_group.csv" download>Профессиональные группы, CSV</a><a href="online_job_sources_status.csv" download>Покрытие источников, CSV</a></div></section>
</section>
'''

ONLINE_JS = r'''
<script id="online-mode-script">
const onlineRegionData=__ONLINE_DATA__,onlineSourceStatus=__ONLINE_STATUS__;
const modeOfficial=document.getElementById("mode-official"),modeOnline=document.getElementById("mode-online"),onlinePanel=document.getElementById("online-panel");
const onlineMethodRegion=document.getElementById("online-method-region"),onlineMethodTitle=document.getElementById("online-method-title"),onlineMethodText=document.getElementById("online-method-text");
const pageTitle=document.querySelector('.title'),pageSubtitle=document.querySelector('.subtitle'),officialTitle=pageTitle.textContent,officialSubtitle=pageSubtitle.textContent;
function setDataMode(mode){const online=mode==="online";document.body.classList.toggle("online-mode",online);modeOfficial.classList.toggle("active",!online);modeOnline.classList.toggle("active",online);modeOfficial.setAttribute("aria-pressed",!online);modeOnline.setAttribute("aria-pressed",online);pageTitle.textContent=online?'Доступность вакансий на онлайн-платформах по регионам России':officialTitle;pageSubtitle.textContent=online?'Оперативный срез объявлений hh.ru и SuperJob по состоянию на 13 сентября 2026 года. Значение 1 соответствует наиболее благоприятным условиям, значение 10 — наименее благоприятным. Для отображения сведений выберите субъект на карте.':officialSubtitle;if(online)drawOnlineDashboard();}
modeOfficial.onclick=()=>setDataMode("official");modeOnline.onclick=()=>setDataMode("online");
const onlineFmt=new Intl.NumberFormat("ru-RU"),onlineColor=d3.scaleLinear().domain([1,3,5.5,7.5,10]).range(["#238b45","#86c440","#ffd54f","#ef4b2d","#7f001f"]).clamp(true);let onlineReady=false,onlineSelected=null;
function onlineEsc(x){return String(x??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function renderOnlinePanel(iso){const d=onlineRegionData[iso];if(!d){onlineMethodRegion.textContent='Данные по региону отсутствуют';onlineMethodTitle.textContent='Оценка онлайн-вакансий не рассчитана';onlineMethodText.textContent='Отсутствие сведений не позволяет сделать вывод об условиях поиска работы.';onlinePanel.innerHTML='<div class="eyebrow">ДАННЫЕ ОНЛАЙН-ПЛАТФОРМ ОТСУТСТВУЮТ</div><div class="region-name">Сопоставимые объявления по региону не найдены</div><p class="online-note">Отсутствие сведений не является признаком благоприятных или неблагоприятных условий поиска работы.</p>';return;}onlineSelected=iso;onlineMethodRegion.textContent=`${d.region} · ${d.score.toFixed(1)} из 10`;onlineMethodTitle.textContent=d.score>=7?'Низкая доступность вакансий с типичным для региона уровнем оплаты':d.score>=4?'Средняя доступность вакансий относительно других регионов':'Высокая доступность вакансий с типичным для региона уровнем оплаты';onlineMethodText.textContent=`На 10 тыс. человек рабочей силы приходится ${d.listings_per_10k.toFixed(1)} объявления; медианная предлагаемая заработная плата — ${d.median_salary?onlineFmt.format(d.median_salary)+' ₽':'не указана'}.`;document.querySelectorAll('.online-region').forEach(x=>x.classList.toggle('selected',x.dataset.iso===iso));onlinePanel.innerHTML=`<div class="eyebrow">ВАКАНСИИ · ${onlineEsc(d.snapshot_at.slice(0,10))}</div><div class="region-name">${onlineEsc(d.region)}</div><div class="overall"><span>Доступность вакансий с типичным уровнем оплаты<br><small>1 — наиболее благоприятные условия, 10 — наименее благоприятные</small></span><strong style="color:${onlineColor(d.score)}">${d.score.toFixed(1)}</strong></div><div class="online-metrics"><div><span>Объявления на онлайн-платформах</span><strong>${onlineFmt.format(d.reported_total)}</strong></div><div><span>Объявления на 10 тыс. человек рабочей силы</span><strong>${d.listings_per_10k.toFixed(1)}</strong></div><div><span>Медианная предлагаемая заработная плата</span><strong>${d.median_salary?onlineFmt.format(d.median_salary)+' ₽':'нет данных'}</strong></div><div><span>Доля объявлений с указанной оплатой</span><strong>${d.salary_coverage.toFixed(0)}% выборки</strong></div></div><h3 class="rank-title" style="margin-top:15px">Профессиональные направления с наиболее ограниченным предложением вакансий</h3><p class="online-note">Для малых выборок применяется статистическая корректировка, снижающая влияние единичных объявлений на итоговый рейтинг.</p>${d.groups.slice(0,9).map(g=>`<div class="online-rank"><b>${g.rank}</b><button class="rank-name" data-online-group="${onlineEsc(g.name)}">${onlineEsc(g.name)}<small>${g.sample} объявлений в выборке · медианная предлагаемая зарплата ${g.median_salary?onlineFmt.format(g.median_salary)+' ₽':'не указана'}</small></button><span class="badge" style="background:${onlineColor(g.score)}">${g.score.toFixed(1)}</span></div><div class="online-example" data-online-detail="${onlineEsc(g.name)}" hidden>${g.examples.map(v=>`<a href="${onlineEsc(v.url)}" target="_blank" rel="noopener">${onlineEsc(v.title)}</a>${v.salary_mid_rur?' · '+onlineFmt.format(v.salary_mid_rur)+' ₽':''}`).join('<br>')||'Примеры в выборке отсутствуют'}</div>`).join('')}`;onlinePanel.querySelectorAll('[data-online-group]').forEach(b=>b.onclick=()=>{const x=onlinePanel.querySelector(`[data-online-detail="${CSS.escape(b.dataset.onlineGroup)}"]`);x.hidden=!x.hidden;});}
function drawOnlineDashboard(){if(onlineReady)return;onlineReady=true;document.getElementById('online-source-status').innerHTML=onlineSourceStatus.map(s=>`<div class="${s.status==='ok'?'ok':''}"><b>${onlineEsc(s.source)}</b><br><span class="${s.status==='ok'?'source-ok':'source-missing'}">${onlineEsc(s.message)}</span></div>`).join('');const host=document.getElementById('online-map'),w=980,h=620,p=d3.geoConicEqualArea().parallels([50,70]).rotate([-100,0]).fitExtent([[20,20],[w-20,h-20]],geojson),path=d3.geoPath(p),svg=d3.select(host).append('svg').attr('viewBox',`0 0 ${w} ${h}`).attr('preserveAspectRatio','xMidYMid meet');svg.selectAll('path').data(geojson.features).join('path').attr('class','online-region').attr('data-iso',f=>f.properties.shapeISO).attr('d',path).attr('fill',f=>onlineRegionData[f.properties.shapeISO]?onlineColor(onlineRegionData[f.properties.shapeISO].score):'#e5e8ee').attr('stroke','#fff').attr('stroke-width',.65).on('mouseenter',(e,f)=>renderOnlinePanel(f.properties.shapeISO)).on('click',(e,f)=>renderOnlinePanel(f.properties.shapeISO)).append('title').text(f=>onlineRegionData[f.properties.shapeISO]?`${onlineRegionData[f.properties.shapeISO].region}: ${onlineRegionData[f.properties.shapeISO].score.toFixed(1)}`:`${f.properties.shapeName}: нет онлайн-среза`);drawOnlineCharts();const first=Object.entries(onlineRegionData).sort((a,b)=>b[1].score-a[1].score)[0];if(first)renderOnlinePanel(first[0]);}
function drawOnlineCharts(){const data=Object.entries(onlineRegionData).map(([iso,d])=>({iso,...d})).filter(d=>d.median_salary);const top=data.slice().sort((a,b)=>b.median_salary-a.median_salary).slice(0,20);drawBars(top);drawScatter(data);drawGapChart(data.filter(d=>d.official_score!=null));drawSalaryRatioChart(data);drawSourceChart();}
function drawBars(data){const host=document.getElementById('online-salary-chart'),w=Math.max(500,host.clientWidth||600),h=410,m={top:15,right:45,bottom:45,left:155},svg=d3.select(host).append('svg').attr('viewBox',`0 0 ${w} ${h}`),x=d3.scaleLinear().domain([0,d3.max(data,d=>d.median_salary)*1.08]).range([m.left,w-m.right]),y=d3.scaleBand().domain(data.map(d=>d.region)).range([m.top,h-m.bottom]).padding(.22);svg.selectAll('rect').data(data).join('rect').attr('x',m.left).attr('y',d=>y(d.region)).attr('width',d=>x(d.median_salary)-m.left).attr('height',y.bandwidth()).attr('fill','#4f91dc');svg.append('g').attr('transform',`translate(${m.left},0)`).call(d3.axisLeft(y).tickFormat(d=>d.length>22?d.slice(0,20)+'…':d));svg.append('g').attr('transform',`translate(0,${h-m.bottom})`).call(d3.axisBottom(x).ticks(5).tickFormat(d=>onlineFmt.format(d/1000)+' тыс.'));svg.append('text').attr('x',(m.left+w-m.right)/2).attr('y',h-6).attr('text-anchor','middle').text('Рублей в месяц, выборка объявлений');}
function drawScatter(data){const host=document.getElementById('online-scatter-chart'),w=Math.max(500,host.clientWidth||600),h=410,m={top:15,right:25,bottom:50,left:65},svg=d3.select(host).append('svg').attr('viewBox',`0 0 ${w} ${h}`),x=d3.scaleLinear().domain(d3.extent(data,d=>d.listings_per_10k)).nice().range([m.left,w-m.right]),y=d3.scaleLinear().domain(d3.extent(data,d=>d.salary_ratio)).nice().range([h-m.bottom,m.top]);svg.selectAll('circle').data(data).join('circle').attr('cx',d=>x(d.listings_per_10k)).attr('cy',d=>y(d.salary_ratio)).attr('r',5).attr('fill',d=>onlineColor(d.score)).attr('opacity',.8).on('click',(e,d)=>renderOnlinePanel(d.iso)).append('title').text(d=>`${d.region}: ${d.listings_per_10k} объявлений на 10 тыс.; зарплата ${d.salary_ratio} от официальной медианы`);svg.append('g').attr('transform',`translate(0,${h-m.bottom})`).call(d3.axisBottom(x).ticks(5));svg.append('g').attr('transform',`translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5));svg.append('text').attr('x',(m.left+w-m.right)/2).attr('y',h-6).attr('text-anchor','middle').text('Объявлений на 10 тыс. рабочей силы');svg.append('text').attr('transform','rotate(-90)').attr('x',-(m.top+h-m.bottom)/2).attr('y',14).attr('text-anchor','middle').text('Медиана объявлений / медиана региона');}
function shortRegion(s){return s.length>20?s.slice(0,18)+'…':s;}
function drawGapChart(data){const host=document.getElementById('online-gap-chart'),rows=data.map(d=>({...d,gap:d.score-d.official_score})).sort((a,b)=>Math.abs(b.gap)-Math.abs(a.gap)).slice(0,12),w=Math.max(360,host.clientWidth||430),h=360,m={top:12,right:25,bottom:42,left:130},svg=d3.select(host).append('svg').attr('viewBox',`0 0 ${w} ${h}`),limit=d3.max(rows,d=>Math.abs(d.gap))||1,x=d3.scaleLinear().domain([-limit,limit]).range([m.left,w-m.right]),y=d3.scaleBand().domain(rows.map(d=>d.region)).range([m.top,h-m.bottom]).padding(.24);svg.append('line').attr('x1',x(0)).attr('x2',x(0)).attr('y1',m.top).attr('y2',h-m.bottom).attr('stroke','#667085');svg.selectAll('rect').data(rows).join('rect').attr('x',d=>x(Math.min(0,d.gap))).attr('y',d=>y(d.region)).attr('width',d=>Math.abs(x(d.gap)-x(0))).attr('height',y.bandwidth()).attr('fill',d=>d.gap>0?'#e0674f':'#4688cf').on('click',(e,d)=>renderOnlinePanel(d.iso)).append('title').text(d=>`${d.region}: онлайн ${d.score.toFixed(1)}, Росстат ${d.official_score.toFixed(1)}, разница ${d.gap>0?'+':''}${d.gap.toFixed(1)}`);svg.append('g').attr('transform',`translate(${m.left},0)`).call(d3.axisLeft(y).tickFormat(shortRegion));svg.append('g').attr('transform',`translate(0,${h-m.bottom})`).call(d3.axisBottom(x).ticks(5));svg.append('text').attr('x',(m.left+w-m.right)/2).attr('y',h-7).attr('text-anchor','middle').attr('font-size',10).text('Онлайн-индекс минус индекс Росстата');}
function drawSalaryRatioChart(data){const host=document.getElementById('online-salary-ratio-chart'),rows=data.map(d=>({...d,diff:(d.salary_ratio-1)*100})).sort((a,b)=>Math.abs(b.diff)-Math.abs(a.diff)).slice(0,12),w=Math.max(360,host.clientWidth||430),h=360,m={top:12,right:25,bottom:42,left:130},svg=d3.select(host).append('svg').attr('viewBox',`0 0 ${w} ${h}`),lo=Math.min(-5,d3.min(rows,d=>d.diff)),hi=Math.max(5,d3.max(rows,d=>d.diff)),x=d3.scaleLinear().domain([lo,hi]).nice().range([m.left,w-m.right]),y=d3.scaleBand().domain(rows.map(d=>d.region)).range([m.top,h-m.bottom]).padding(.24);svg.append('line').attr('x1',x(0)).attr('x2',x(0)).attr('y1',m.top).attr('y2',h-m.bottom).attr('stroke','#667085');svg.selectAll('rect').data(rows).join('rect').attr('x',d=>x(Math.min(0,d.diff))).attr('y',d=>y(d.region)).attr('width',d=>Math.abs(x(d.diff)-x(0))).attr('height',y.bandwidth()).attr('fill',d=>d.diff>=0?'#35a56f':'#e47754').on('click',(e,d)=>renderOnlinePanel(d.iso)).append('title').text(d=>`${d.region}: ${d.diff>0?'+':''}${d.diff.toFixed(0)}% к официальной медиане`);svg.append('g').attr('transform',`translate(${m.left},0)`).call(d3.axisLeft(y).tickFormat(shortRegion));svg.append('g').attr('transform',`translate(0,${h-m.bottom})`).call(d3.axisBottom(x).ticks(5).tickFormat(d=>d+'%'));svg.append('text').attr('x',(m.left+w-m.right)/2).attr('y',h-7).attr('text-anchor','middle').attr('font-size',10).text('Отклонение медианы объявлений, %');}
function drawSourceChart(){const host=document.getElementById('online-source-chart'),rows=['hh.ru','superjob.ru'].map(name=>onlineSourceStatus.find(s=>s.source===name)||{source:name,regions:0,status:'ожидает подключения'}),w=Math.max(360,host.clientWidth||430),h=360,m={top:30,right:30,bottom:70,left:65},svg=d3.select(host).append('svg').attr('viewBox',`0 0 ${w} ${h}`),x=d3.scaleBand().domain(rows.map(d=>d.source)).range([m.left,w-m.right]).padding(.38),y=d3.scaleLinear().domain([0,84]).range([h-m.bottom,m.top]);svg.selectAll('rect').data(rows).join('rect').attr('x',d=>x(d.source)).attr('y',d=>y(d.regions||0)).attr('width',x.bandwidth()).attr('height',d=>y(0)-y(d.regions||0)).attr('fill',d=>d.regions?'#2879cf':'#d9dee7');svg.selectAll('.source-value').data(rows).join('text').attr('x',d=>x(d.source)+x.bandwidth()/2).attr('y',d=>y(d.regions||0)-7).attr('text-anchor','middle').attr('font-weight',800).text(d=>`${d.regions||0} / 84`);svg.append('g').attr('transform',`translate(0,${h-m.bottom})`).call(d3.axisBottom(x));svg.append('g').attr('transform',`translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5));svg.append('text').attr('x',(m.left+w-m.right)/2).attr('y',h-23).attr('text-anchor','middle').attr('font-size',10).text('Субъектов с сопоставимым срезом');}
</script>
'''


def inject_online_mode(map_file: Path | str) -> None:
    map_file = Path(map_file)
    html = map_file.read_text(encoding="utf-8")
    if 'id="online-mode-script"' in html:
        html = html.split('<style id="online-mode-style">')[0] + html.split('</style>', 1)[-1]
        raise RuntimeError("Повторная инъекция не поддерживается: сначала пересоберите основной HTML")
    data, status = prepare_online_data()
    html = html.replace("</head>", ONLINE_CSS + "\n</head>")
    html = html.replace('<div class="subtitle">', '<div class="subtitle">', 1)
    subtitle_end = html.find("</div>", html.find('<div class="subtitle">')) + len("</div>")
    switch = '<div class="data-mode-switch" role="group" aria-label="Источник данных"><button id="mode-official" class="active" aria-pressed="true">Официальная статистика Росстата</button><button id="mode-online" aria-pressed="false">Онлайн-сервисы вакансий</button></div><main class="official-dashboard">'
    html = html[:subtitle_end] + switch + html[subtitle_end:]
    body_script = html.rfind("<script>")
    # Keep both dashboards inside the same outer .shell container. The original
    # closing </main> belongs to .shell; first close .official-dashboard, insert
    # the online dashboard, and only then retain the outer closing tag.
    prefix = html[:body_script]
    shell_close = prefix.rfind("</main>")
    if shell_close < 0:
        raise RuntimeError("Не найден закрывающий тег основного контейнера .shell")
    html = prefix[:shell_close] + "</main>" + ONLINE_HTML + prefix[shell_close:] + html[body_script:]
    js = ONLINE_JS.replace("__ONLINE_DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))).replace("__ONLINE_STATUS__", json.dumps(status, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("</body>", js + "\n</body>")
    map_file.write_text(html, encoding="utf-8")
    print(f"Онлайн-режим добавлен: {len(data)} регионов")


if __name__ == "__main__":
    inject_online_mode(ROOT / "interactive_regional_deficit_map.html")
