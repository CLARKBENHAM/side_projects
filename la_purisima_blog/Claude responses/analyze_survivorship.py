"""Analyze within-Alta-California mission survivorship variation.

Question: across the 21 Alta California missions, observed survivorship
ranges from ~16% (La Purisima) to ~43% (San Luis Rey). What predicts the
spread?

Inputs:  ../Codex responses/data/mission_vital_stats.csv
Outputs: mission_features.csv  -- enriched per-mission features
         (printed) correlations and ranked findings
"""

import csv
from pathlib import Path
from statistics import mean, stdev

HERE = Path(__file__).parent
SRC = HERE.parent / "Codex responses" / "data" / "mission_vital_stats.csv"
OUT = HERE / "mission_features.csv"


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    deny = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def main() -> None:
    rows: list[dict[str, str]] = []
    with SRC.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)

    enriched: list[dict[str, float | str]] = []
    for r in rows:
        founded = int(r["founded"])
        years = int(r["years"])
        baptisms = int(r["baptisms"])
        deaths_str = r["deaths"]
        survived_str = r["survived_pct"]
        lat = float(r["lat"])
        lon = float(r["lon"])
        if deaths_str == "" or survived_str == "":
            deaths: float | None = None
            survived: float | None = None
        else:
            deaths = float(deaths_str)
            survived = float(survived_str)
        baptisms_per_year = baptisms / years
        deaths_per_year = (deaths / years) if deaths is not None else None
        # death-to-baptism ratio is the key inverse of survivorship
        # but normalized differently than the published "survived_pct"
        d_over_b = (deaths / baptisms) if deaths is not None else None
        enriched.append(
            {
                "mission": r["mission"],
                "founded": founded,
                "years": years,
                "baptisms": baptisms,
                "deaths": deaths if deaths is not None else "",
                "survived_pct": survived if survived is not None else "",
                "lat": lat,
                "lon": lon,
                "baptisms_per_year": round(baptisms_per_year, 2),
                "deaths_per_year": (
                    round(deaths_per_year, 2) if deaths_per_year is not None else ""
                ),
                "deaths_over_baptisms": (
                    round(d_over_b, 4) if d_over_b is not None else ""
                ),
            }
        )

    # write enriched CSV
    fieldnames = [
        "mission",
        "founded",
        "years",
        "baptisms",
        "deaths",
        "survived_pct",
        "lat",
        "lon",
        "baptisms_per_year",
        "deaths_per_year",
        "deaths_over_baptisms",
    ]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in enriched:
            w.writerow(row)

    # only the 19 missions with death data
    have_data = [r for r in enriched if r["survived_pct"] != ""]
    print(f"Missions with death data: {len(have_data)} of {len(enriched)}")
    print(f"Missing: {[r['mission'] for r in enriched if r['survived_pct'] == '']}")
    print()

    surv = [float(r["survived_pct"]) for r in have_data]
    print(f"Survivorship %: min={min(surv):.2f}, max={max(surv):.2f}, "
          f"mean={mean(surv):.2f}, sd={stdev(surv):.2f}")
    print()

    # rank
    ranked = sorted(have_data, key=lambda r: float(r["survived_pct"]))
    print("Ranked by survivorship (lowest -> highest):")
    for r in ranked:
        print(
            f"  {float(r['survived_pct']):5.2f}%  {r['mission']:32s}  "
            f"founded {r['founded']}  lat {r['lat']:.2f}  "
            f"baptisms/yr {r['baptisms_per_year']:6.2f}  "
            f"deaths/yr {r['deaths_per_year']:6.2f}"
        )
    print()

    # correlations of survivorship with candidate predictors
    candidates = [
        "founded",
        "years",
        "lat",
        "lon",
        "baptisms",
        "deaths",
        "baptisms_per_year",
        "deaths_per_year",
    ]
    surv_vec = [float(r["survived_pct"]) for r in have_data]
    print("Pearson correlation with survivorship %:")
    for c in candidates:
        xs = [float(r[c]) for r in have_data]
        rho = pearson(xs, surv_vec)
        print(f"  {c:24s}  r = {rho:+.3f}")
    print()

    # split by latitude bands
    south = [r for r in have_data if float(r["lat"]) < 35.0]
    central = [r for r in have_data if 35.0 <= float(r["lat"]) < 36.5]
    north = [r for r in have_data if float(r["lat"]) >= 36.5]

    def band(name: str, rs: list) -> None:
        if not rs:
            print(f"  {name}: (no missions)")
            return
        ss = [float(r["survived_pct"]) for r in rs]
        print(
            f"  {name:8s}  n={len(rs)}  surv mean={mean(ss):5.2f}  "
            f"sd={stdev(ss) if len(ss) > 1 else 0:5.2f}  "
            f"min={min(ss):5.2f}  max={max(ss):5.2f}"
        )

    print("Survivorship by latitude band (split at 35.0 / 36.5):")
    band("south", south)
    band("central", central)
    band("north", north)
    print()

    # split by founding era
    early = [r for r in have_data if int(r["founded"]) < 1782]
    middle = [r for r in have_data if 1782 <= int(r["founded"]) < 1797]
    late = [r for r in have_data if int(r["founded"]) >= 1797]
    print("Survivorship by founding era:")
    band("<1782", early)
    band("1782-96", middle)
    band(">=1797", late)
    print()

    # split by mission scale (baptisms/yr)
    bpys = sorted(have_data, key=lambda r: float(r["baptisms_per_year"]))
    cut = len(bpys) // 3
    small = bpys[:cut]
    mid = bpys[cut: 2 * cut]
    big = bpys[2 * cut:]
    print("Survivorship by mission scale (baptisms/year terciles):")
    band("small", small)
    band("medium", mid)
    band("large", big)


if __name__ == "__main__":
    main()
