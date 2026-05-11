from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PLOT_DIR = ROOT / "plots"


MISSION_STATS = [
    {
        "mission": "San Diego de Alcala",
        "founded": 1769,
        "years": 65,
        "baptisms": 6500,
        "deaths": 3884,
        "survived_pct": 39.82,
        "lat": 32.784,
        "lon": -117.107,
    },
    {
        "mission": "San Carlos Borromeo",
        "founded": 1770,
        "years": 64,
        "baptisms": 4134,
        "deaths": 2819,
        "survived_pct": 31.81,
        "lat": 36.542,
        "lon": -121.919,
    },
    {
        "mission": "San Antonio de Padua",
        "founded": 1771,
        "years": 63,
        "baptisms": 3984,
        "deaths": 3184,
        "survived_pct": 20.08,
        "lat": 35.991,
        "lon": -121.248,
    },
    {
        "mission": "San Gabriel Arcangel",
        "founded": 1771,
        "years": 63,
        "baptisms": 7854,
        "deaths": 6196,
        "survived_pct": 21.11,
        "lat": 34.097,
        "lon": -118.106,
    },
    {
        "mission": "San Luis Obispo de Tolosa",
        "founded": 1772,
        "years": 62,
        "baptisms": 2707,
        "deaths": 2043,
        "survived_pct": 24.53,
        "lat": 35.279,
        "lon": -120.664,
    },
    {
        "mission": "San Francisco de Asis",
        "founded": 1776,
        "years": 58,
        "baptisms": 7048,
        "deaths": 5680,
        "survived_pct": 19.41,
        "lat": 37.764,
        "lon": -122.426,
    },
    {
        "mission": "San Juan Capistrano",
        "founded": 1776,
        "years": 58,
        "baptisms": 4410,
        "deaths": 3340,
        "survived_pct": 24.26,
        "lat": 33.502,
        "lon": -117.662,
    },
    {
        "mission": "Santa Clara de Asis",
        "founded": 1777,
        "years": 57,
        "baptisms": 7432,
        "deaths": 5832,
        "survived_pct": 21.53,
        "lat": 37.349,
        "lon": -121.941,
    },
    {
        "mission": "San Buenaventura",
        "founded": 1782,
        "years": 52,
        "baptisms": 3252,
        "deaths": 2131,
        "survived_pct": 34.47,
        "lat": 34.280,
        "lon": -119.296,
    },
    {
        "mission": "Santa Barbara",
        "founded": 1786,
        "years": 48,
        "baptisms": 5679,
        "deaths": 4026,
        "survived_pct": 29.11,
        "lat": 34.438,
        "lon": -119.713,
    },
    {
        "mission": "La Purisima Concepcion",
        "founded": 1787,
        "years": 47,
        "baptisms": 4368,
        "deaths": 3656,
        "survived_pct": 16.30,
        "lat": 34.671,
        "lon": -120.421,
    },
    {
        "mission": "Santa Cruz",
        "founded": 1791,
        "years": 43,
        "baptisms": 2523,
        "deaths": 2081,
        "survived_pct": 17.51,
        "lat": 36.974,
        "lon": -122.030,
    },
    {
        "mission": "Nuestra Senora de la Soledad",
        "founded": 1791,
        "years": 43,
        "baptisms": 2234,
        "deaths": 1803,
        "survived_pct": 19.29,
        "lat": 36.404,
        "lon": -121.355,
    },
    {
        "mission": "San Jose",
        "founded": 1797,
        "years": 37,
        "baptisms": 6650,
        "deaths": 4722,
        "survived_pct": 29.00,
        "lat": 37.533,
        "lon": -121.919,
    },
    {
        "mission": "San Juan Bautista",
        "founded": 1797,
        "years": 37,
        "baptisms": 4118,
        "deaths": 2697,
        "survived_pct": 34.51,
        "lat": 36.845,
        "lon": -121.536,
    },
    {
        "mission": "San Miguel Arcangel",
        "founded": 1797,
        "years": 37,
        "baptisms": 2471,
        "deaths": 1805,
        "survived_pct": 26.95,
        "lat": 35.745,
        "lon": -120.698,
    },
    {
        "mission": "San Fernando Rey de Espana",
        "founded": 1797,
        "years": 37,
        "baptisms": 2959,
        "deaths": 2013,
        "survived_pct": 31.97,
        "lat": 34.273,
        "lon": -118.461,
    },
    {
        "mission": "San Luis Rey",
        "founded": 1798,
        "years": 36,
        "baptisms": 5600,
        "deaths": 3184,
        "survived_pct": 43.14,
        "lat": 33.232,
        "lon": -117.319,
    },
    {
        "mission": "Santa Ines",
        "founded": 1804,
        "years": 30,
        "baptisms": 1372,
        "deaths": 1124,
        "survived_pct": 18.08,
        "lat": 34.595,
        "lon": -120.137,
    },
    {
        "mission": "San Rafael Arcangel",
        "founded": 1817,
        "years": 17,
        "baptisms": 1907,
        "deaths": None,
        "survived_pct": None,
        "lat": 37.974,
        "lon": -122.530,
    },
    {
        "mission": "San Francisco Solano",
        "founded": 1823,
        "years": 11,
        "baptisms": 1018,
        "deaths": None,
        "survived_pct": None,
        "lat": 38.294,
        "lon": -122.455,
    },
]


POPULATION_BOUNDS = [
    {
        "year": 1769,
        "low": 279000,
        "mid": 310000,
        "high": 341000,
        "note": "Cook estimate of 310,000 plus/minus 10 percent",
    },
    {
        "year": 1845,
        "low": 100000,
        "mid": 150000,
        "high": 150000,
        "note": "NPS summary says perhaps no more than 150,000 by 1845",
    },
    {
        "year": 1846,
        "low": 120000,
        "mid": 150000,
        "high": 150000,
        "note": "Madley starting estimate for American conquest era",
    },
    {
        "year": 1870,
        "low": 30000,
        "mid": 30000,
        "high": 40000,
        "note": "Post Gold Rush-era estimate, about 30,000 in Madley/Kroeber framing",
    },
    {
        "year": 1900,
        "low": 15000,
        "mid": 16500,
        "high": 20000,
        "note": "Kroeber/NAHC nadir range; 16,500 commonly cited",
    },
]


MISSION_SUMMARY = [
    {
        "source": "NPS/ECPP summary",
        "through_year": 1834,
        "native_baptisms": 85840,
        "recorded_deaths": 59538,
        "births": "",
        "resident_peak": "",
        "resident_peak_years": "",
        "resident_1834": "",
        "note": "Death data missing for San Rafael and San Francisco Solano",
    },
    {
        "source": "Cook synthesis",
        "through_year": 1834,
        "native_baptisms": 81000,
        "recorded_deaths": 60600,
        "births": 29100,
        "resident_peak": 21000,
        "resident_peak_years": "1821-1824",
        "resident_1834": 15000,
        "note": "No Alta California mission was demographically self-sustaining",
    },
]


EPIDEMIC_EVENTS = [
    {
        "year": 1777,
        "event": "San Francisco Bay epidemic",
        "place": "San Francisco Bay mission area",
        "impact": "Early recorded disease crisis after missionization began",
    },
    {
        "year": 1801,
        "event": "Severe epidemic",
        "place": "Mission and nearby Native communities",
        "impact": "Reported as killing thousands across affected communities",
    },
    {
        "year": 1806,
        "event": "Measles epidemic",
        "place": "Mission San Jose and neighboring missions",
        "impact": "About 140 San Jose deaths in cited NPS summary",
    },
    {
        "year": 1814,
        "event": "Severe mortality year",
        "place": "Mission San Jose",
        "impact": "NPS summary notes 67 deaths",
    },
    {
        "year": 1825,
        "event": "Smallpox among Cahuilla",
        "place": "Southern inland California",
        "impact": "Documented outbreak beyond mission walls",
    },
    {
        "year": 1827,
        "event": "Measles epidemic",
        "place": "Mission communities",
        "impact": "Hundreds of deaths in 1827-1828",
    },
    {
        "year": 1833,
        "event": "Malaria epidemic",
        "place": "Central Valley and inland communities",
        "impact": "Major inland mortality, especially among valley peoples",
    },
    {
        "year": 1837,
        "event": "Smallpox epidemic",
        "place": "Northern California and adjacent frontier",
        "impact": "Major late mission/secularization-era mortality",
    },
    {
        "year": 1844,
        "event": "Smallpox near La Purisima",
        "place": "La Purisima Indian community",
        "impact": "NPS summary reports about 75 percent mortality",
    },
    {
        "year": 1846,
        "event": "American conquest-era violence",
        "place": "California",
        "impact": "Beginning of 1846-1873 genocide period in Madley's account",
    },
]


LAND_TRANSFER = [
    {
        "category": "Mission land claims by late 1700s",
        "value": 1000000,
        "unit": "acres",
        "note": "California 100 summary, approximate",
    },
    {
        "category": "Spanish/Mexican grants",
        "value": 813,
        "unit": "grants",
        "note": "California State Lands Commission",
    },
    {
        "category": "Confirmed Spanish/Mexican grants",
        "value": 604,
        "unit": "grants",
        "note": "California State Lands Commission",
    },
    {
        "category": "Surveyed acreage under confirmed grants",
        "value": 4321189,
        "unit": "acres",
        "note": "California State Lands Commission",
    },
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_all_csvs() -> None:
    write_csv(DATA_DIR / "mission_vital_stats.csv", MISSION_STATS)
    write_csv(DATA_DIR / "california_native_population_bounds.csv", POPULATION_BOUNDS)
    write_csv(DATA_DIR / "mission_system_summary.csv", MISSION_SUMMARY)
    write_csv(DATA_DIR / "epidemic_events.csv", EPIDEMIC_EVENTS)
    write_csv(DATA_DIR / "land_transfer_summary.csv", LAND_TRANSFER)


def mission_short_name(name: str) -> str:
    replacements = {
        "Nuestra Senora de la Soledad": "Soledad",
        "San Luis Obispo de Tolosa": "San Luis Obispo",
        "San Fernando Rey de Espana": "San Fernando",
        "La Purisima Concepcion": "La Purisima",
        "San Carlos Borromeo": "San Carlos",
        "San Diego de Alcala": "San Diego",
        "San Francisco de Asis": "San Francisco",
        "San Gabriel Arcangel": "San Gabriel",
        "San Antonio de Padua": "San Antonio",
    }
    return replacements.get(name, name)


def plot_mission_baptisms_deaths() -> None:
    rows = MISSION_STATS
    labels = [mission_short_name(row["mission"]) for row in rows]
    y_positions = list(range(len(rows)))
    baptisms = [row["baptisms"] for row in rows]
    deaths = [row["deaths"] or 0 for row in rows]

    plt.figure(figsize=(11, 10))
    plt.barh(y_positions, baptisms, color="#5176a3", label="Baptisms")
    plt.barh(
        y_positions,
        deaths,
        color="#b65d54",
        label="Recorded deaths",
        alpha=0.82,
    )
    for idx, row in enumerate(rows):
        if row["deaths"] is None:
            plt.text(
                row["baptisms"] + 120,
                idx,
                "deaths not in table",
                va="center",
                fontsize=8,
                color="#5f6368",
            )
    plt.yticks(y_positions, labels)
    plt.xlabel("People recorded")
    plt.title("Alta California mission baptisms and recorded deaths through 1834")
    plt.legend(loc="lower right")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mission_baptisms_deaths_by_mission.png", dpi=220)
    plt.close()


def plot_mission_survivorship() -> None:
    rows = [row for row in MISSION_STATS if row["survived_pct"] is not None]
    rows = sorted(rows, key=lambda row: row["survived_pct"])
    labels = [mission_short_name(row["mission"]) for row in rows]
    values = [row["survived_pct"] for row in rows]
    colors = ["#b65d54" if "Purisima" in label else "#5176a3" for label in labels]

    plt.figure(figsize=(10, 8))
    plt.barh(labels, values, color=colors)
    plt.axvline(24, color="#333333", linestyle="--", linewidth=1)
    plt.text(24.5, 0.5, "Approx. system average", fontsize=8, color="#333333")
    plt.xlabel("Survival share after recorded baptisms (%)")
    plt.title("Mission survivorship was low across the Alta California system")
    plt.xlim(0, 50)
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mission_survivorship_by_mission.png", dpi=220)
    plt.close()


def plot_population_bounds() -> None:
    rows = POPULATION_BOUNDS
    years = [row["year"] for row in rows]
    low = [row["low"] for row in rows]
    mid = [row["mid"] for row in rows]
    high = [row["high"] for row in rows]

    plt.figure(figsize=(10, 6))
    plt.fill_between(years, low, high, color="#b65d54", alpha=0.18, label="Range")
    plt.plot(
        years, mid, color="#8f2f2d", marker="o", linewidth=2.4, label="Reference line"
    )
    plt.axvline(1769, color="#555555", linewidth=1, linestyle="--")
    plt.axvline(1848, color="#555555", linewidth=1, linestyle="--")
    plt.text(1771, 325000, "Mission era begins", fontsize=8)
    plt.text(1850, 325000, "U.S. conquest / Gold Rush", fontsize=8)
    plt.ylabel("Estimated Native Californian population")
    plt.xlabel("Year")
    plt.title("Native Californian population decline with explicit uncertainty bounds")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "california_native_population_bounds.png", dpi=220)
    plt.close()


def plot_epidemic_timeline() -> None:
    rows = EPIDEMIC_EVENTS
    years = [row["year"] for row in rows]
    y_values = [idx % 2 for idx, _ in enumerate(rows)]

    plt.figure(figsize=(12, 4.8))
    plt.hlines(0.5, 1770, 1850, color="#9aa0a6", linewidth=1)
    plt.scatter(years, y_values, color="#8f2f2d", s=70, zorder=2)
    for row, y_value in zip(rows, y_values, strict=True):
        vertical = 0.13 if y_value == 0 else -0.13
        va = "bottom" if y_value == 0 else "top"
        label = f"{row['year']}: {row['event']}"
        plt.text(row["year"], y_value + vertical, label, ha="center", va=va, fontsize=8)
    plt.ylim(-0.55, 1.55)
    plt.yticks([])
    plt.xlabel("Year")
    plt.title("Selected disease and mortality shocks to use in the blog chronology")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "epidemic_timeline.png", dpi=220)
    plt.close()


def plot_mission_deaths_map() -> None:
    rows = [row for row in MISSION_STATS if row["deaths"] is not None]
    deaths = [row["deaths"] for row in rows]
    max_deaths = max(deaths)
    sizes = [80 + 620 * (death / max_deaths) for death in deaths]

    plt.figure(figsize=(7, 9))
    ca_lon = [-124.4, -120.0, -114.1, -117.1, -124.4, -124.4]
    ca_lat = [42.0, 42.0, 34.7, 32.5, 40.5, 42.0]
    plt.plot(ca_lon, ca_lat, color="#444444", linewidth=1)
    plt.scatter(
        [row["lon"] for row in rows],
        [row["lat"] for row in rows],
        s=sizes,
        c=deaths,
        cmap="Reds",
        alpha=0.72,
        edgecolor="#333333",
        linewidth=0.5,
    )
    for row in rows:
        if row["mission"] in {
            "La Purisima Concepcion",
            "San Francisco de Asis",
            "San Gabriel Arcangel",
            "San Diego de Alcala",
            "Santa Clara de Asis",
        }:
            plt.text(
                row["lon"] + 0.08,
                row["lat"] + 0.08,
                mission_short_name(row["mission"]),
                fontsize=8,
            )
    plt.colorbar(label="Recorded deaths")
    plt.xlim(-125.0, -113.8)
    plt.ylim(32.0, 42.5)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Recorded mission deaths by location (not decade-specific)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mission_deaths_map.png", dpi=220)
    plt.close()


def plot_land_transfer_scale() -> None:
    grant_rows = [
        row
        for row in LAND_TRANSFER
        if row["unit"] == "grants"
        or row["category"] == "Surveyed acreage under confirmed grants"
    ]
    labels = [
        "Spanish/Mexican grants",
        "Confirmed grants",
        "Surveyed acres under confirmed grants",
    ]
    values = [
        grant_rows[0]["value"],
        grant_rows[1]["value"],
        grant_rows[2]["value"] / 10000,
    ]
    units = ["grants", "grants", "10,000 acres"]

    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(labels, values, color=["#5176a3", "#72956f", "#b65d54"])
    for bar, value, unit in zip(bars, values, units, strict=True):
        label = f"{value:,.0f} {unit}"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.ylabel("Scaled value")
    plt.title("Spanish/Mexican land grant scale after secularization")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "land_transfer_scale.png", dpi=220)
    plt.close()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    save_all_csvs()
    plot_mission_baptisms_deaths()
    plot_mission_survivorship()
    plot_population_bounds()
    plot_epidemic_timeline()
    plot_mission_deaths_map()
    plot_land_transfer_scale()


if __name__ == "__main__":
    main()
