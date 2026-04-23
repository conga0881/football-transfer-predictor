# Run with: python3.12 fetch_fpl_data.py
# Downloads FPL 2023-24 and 2024-25 advanced stats (xG, xA, saves, clean sheets, etc.)
# and matches them onto cleaned_data.csv by player name + club.

import io, unicodedata, urllib.request
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

# ── FPL → Transfermarkt club name mapping ─────────────────────────────────────
FPL_TO_TM = {
    "Arsenal":        "Arsenal Football Club",
    "Aston Villa":    "Aston Villa Football Club",
    "Bournemouth":    "Association Football Club Bournemouth",
    "Brentford":      "Brentford Football Club",
    "Brighton":       "Brighton and Hove Albion Football Club",
    "Chelsea":        "Chelsea Football Club",
    "Crystal Palace": "Crystal Palace Football Club",
    "Everton":        "Everton Football Club",
    "Fulham":         "Fulham Football Club",
    "Ipswich":        "Ipswich Town",
    "Leicester":      "Leicester City",
    "Liverpool":      "Liverpool Football Club",
    "Man City":       "Manchester City Football Club",
    "Man Utd":        "Manchester United Football Club",
    "Newcastle":      "Newcastle United Football Club",
    "Nott'm Forest":  "Nottingham Forest Football Club",
    "Southampton":    "Southampton FC",
    "Spurs":          "Tottenham Hotspur Football Club",
    "West Ham":       "West Ham United Football Club",
    "Wolves":         "Wolverhampton Wanderers Football Club",
    # 2023-24 relegated clubs
    "Burnley":        "Burnley Football Club",
    "Luton":          "Luton Town",
    "Sheffield Utd":  "Sheffield United",
}

# FPL element_type → position group (for GK-specific stats)
FPL_POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def fetch_csv(url: str) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return pd.read_csv(io.BytesIO(r.read()))


def norm(s: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.lower().split())


def best_match(query: str, candidates: list[str], threshold=0.82) -> str | None:
    best_score, best_name = 0.0, None
    for c in candidates:
        score = SequenceMatcher(None, query, c).ratio()
        if score > best_score:
            best_score, best_name = score, c
    return best_name if best_score >= threshold else None


def token_overlap(a: str, b: str) -> bool:
    """True if all tokens of the shorter name appear in the longer name.
    Handles 'Bruno Fernandes' ⊂ 'Bruno Borges Fernandes' and reversed
    Korean/East-Asian names like 'Hwang Hee-chan' ↔ 'Hee-chan Hwang'."""
    tok_a = {t for t in a.replace("-", " ").split() if len(t) > 2}
    tok_b = {t for t in b.replace("-", " ").split() if len(t) > 2}
    if not tok_a or not tok_b:
        return False
    if len(tok_a) <= len(tok_b):
        return tok_a.issubset(tok_b)
    return tok_b.issubset(tok_a)


# ── Download FPL seasons ──────────────────────────────────────────────────────
BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

print("Downloading FPL data…")
fpl_seasons = {}
for season, fpl_yr, team_yr in [("2025", "2024-25", "2024-25"),
                                  ("2024", "2023-24", "2023-24")]:
    players = fetch_csv(f"{BASE}/{fpl_yr}/players_raw.csv")
    teams   = fetch_csv(f"{BASE}/{team_yr}/teams.csv")
    team_map = dict(zip(teams["id"], teams["name"]))
    players["fpl_team_name"] = players["team"].map(team_map)
    players["tm_club"]       = players["fpl_team_name"].map(FPL_TO_TM)
    players["fpl_season"]    = int(season)
    players["full_name"]     = players["first_name"].str.strip() + " " + players["second_name"].str.strip()
    players["full_name_norm"]= players["full_name"].apply(norm)
    fpl_seasons[season]      = players
    print(f"  FPL {fpl_yr}: {len(players)} players")

fpl_all = pd.concat(fpl_seasons.values(), ignore_index=True)

# Columns we care about
FPL_COLS = [
    "expected_goals", "expected_assists",
    "expected_goals_per_90", "expected_assists_per_90",
    "expected_goal_involvements_per_90",
    "expected_goals_conceded", "expected_goals_conceded_per_90",
    "clean_sheets", "saves", "saves_per_90",
    "starts",
]

# ── Load cleaned_data.csv ──────────────────────────────────────────────────────
df = pd.read_csv("cleaned_data.csv")
df["name_norm"] = df["name"].apply(norm)

# ── Match FPL → cleaned_data by season + name (+ club as tiebreaker) ──────────
print("\nMatching players…")

matched_records = []
unmatched = []

for season_str, fpl_df in [("2025", fpl_seasons["2025"]),
                             ("2024", fpl_seasons["2024"])]:
    tm_season = df[df["season"] == int(season_str)].copy()
    tm_names   = tm_season["name_norm"].tolist()
    tm_clubs   = tm_season["current_club_name"].tolist()

    for _, frow in fpl_df.iterrows():
        fn   = frow["full_name_norm"]
        wn   = norm(str(frow.get("web_name", "")))
        club = frow.get("tm_club", "")

        # 1. exact name + matching club
        mask_name = tm_season["name_norm"] == fn
        mask_club = tm_season["current_club_name"] == club
        exact_both = tm_season[mask_name & mask_club]
        if len(exact_both) == 1:
            rec = {"player_name_norm": fn, "season": int(season_str),
                   **{c: frow.get(c, np.nan) for c in FPL_COLS}}
            rec["_tm_idx"] = exact_both.index[0]
            matched_records.append(rec)
            continue

        # 2. exact name, any club (handles mid-season moves)
        exact_name = tm_season[mask_name]
        if len(exact_name) == 1:
            rec = {"player_name_norm": fn, "season": int(season_str),
                   **{c: frow.get(c, np.nan) for c in FPL_COLS}}
            rec["_tm_idx"] = exact_name.index[0]
            matched_records.append(rec)
            continue

        # 3. web_name substring in tm_name + matching club
        if wn and len(wn) > 3:
            mask_web  = tm_season["name_norm"].str.contains(wn, regex=False)
            web_club  = tm_season[mask_web & mask_club]
            if len(web_club) == 1:
                rec = {"player_name_norm": fn, "season": int(season_str),
                       **{c: frow.get(c, np.nan) for c in FPL_COLS}}
                rec["_tm_idx"] = web_club.index[0]
                matched_records.append(rec)
                continue

        # 4. fuzzy name match within same club (lowered threshold for compound names)
        if club:
            club_players = tm_season[mask_club]
            hit = best_match(fn, club_players["name_norm"].tolist(), threshold=0.78)
            if hit:
                idx = club_players[club_players["name_norm"] == hit].index[0]
                rec = {"player_name_norm": fn, "season": int(season_str),
                       **{c: frow.get(c, np.nan) for c in FPL_COLS}}
                rec["_tm_idx"] = idx
                matched_records.append(rec)
                continue

        # 4b. Token-overlap match within same club
        #     Catches "Bruno Fernandes" ⊂ "Bruno Borges Fernandes"
        #     and reversed names "Hwang Hee-chan" ↔ "Hee-chan Hwang"
        tok_matched = False
        if club:
            club_players = tm_season[mask_club]
            for tm_idx, tm_row in club_players.iterrows():
                if token_overlap(fn, tm_row["name_norm"]):
                    rec = {"player_name_norm": fn, "season": int(season_str),
                           **{c: frow.get(c, np.nan) for c in FPL_COLS}}
                    rec["_tm_idx"] = tm_idx
                    matched_records.append(rec)
                    tok_matched = True
                    break
        if tok_matched:
            continue

        # 5. TM name is contained in FPL full name (handles "Alisson" ⊂ "Alisson Ramses Becker")
        if club:
            club_players = tm_season[mask_club]
            for tm_idx, tm_row in club_players.iterrows():
                tname = tm_row["name_norm"]
                if len(tname) > 3 and (tname in fn or fn in tname):
                    rec = {"player_name_norm": fn, "season": int(season_str),
                           **{c: frow.get(c, np.nan) for c in FPL_COLS}}
                    rec["_tm_idx"] = tm_idx
                    matched_records.append(rec)
                    break
            else:
                # 6. last token of TM name matches last token of FPL name, same club
                fpl_last = fn.split()[-1] if fn else ""
                for tm_idx, tm_row in club_players.iterrows():
                    tm_last = tm_row["name_norm"].split()[-1] if tm_row["name_norm"] else ""
                    if fpl_last and tm_last and fpl_last == tm_last and len(fpl_last) > 3:
                        rec = {"player_name_norm": fn, "season": int(season_str),
                               **{c: frow.get(c, np.nan) for c in FPL_COLS}}
                        rec["_tm_idx"] = tm_idx
                        matched_records.append(rec)
                        break
                else:
                    unmatched.append((fn, frow.get("fpl_team_name", ""), season_str))
        else:
            unmatched.append((fn, frow.get("fpl_team_name", ""), season_str))

print(f"  Matched {len(matched_records)} FPL rows")
print(f"  Unmatched {len(unmatched)} FPL rows (not in cleaned_data = expected)")

# ── Attach FPL features back to cleaned_data ──────────────────────────────────
matched_df = pd.DataFrame(matched_records)

# Deduplicate: keep highest-minutes row per _tm_idx
if "_tm_idx" in matched_df.columns and "starts" in matched_df.columns:
    matched_df = matched_df.sort_values("starts", ascending=False).drop_duplicates("_tm_idx")

for col in FPL_COLS:
    df[col] = np.nan

for _, row in matched_df.iterrows():
    idx = int(row["_tm_idx"])
    for col in FPL_COLS:
        if col in row:
            df.at[idx, col] = row[col]

match_pct = df[FPL_COLS[0]].notna().mean() * 100
print(f"\nMatch rate: {match_pct:.1f}% of cleaned_data players got FPL stats")

# ── Save enriched data ─────────────────────────────────────────────────────────
df.to_csv("cleaned_data.csv", index=False)
print(f"✅  Saved enriched cleaned_data.csv ({len(df)} rows, {df.shape[1]} columns)")
