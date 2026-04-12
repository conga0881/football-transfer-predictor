import pandas as pd
import numpy as np

print("Loading data...")

players = pd.read_csv("players.csv")
valuations = pd.read_csv("player_valuations.csv")
appearances = pd.read_csv("appearances.csv")
games = pd.read_csv("games.csv")

print("Files loaded. Processing...")

# ── 1. Get each player's LATEST market value ──────────────────────────────────
valuations["date"] = pd.to_datetime(valuations["date"])
latest_valuations = (
    valuations.sort_values("date")
    .groupby("player_id")
    .last()
    .reset_index()[["player_id", "market_value_in_eur"]]
)

# ── 2. Filter Premier League appearances directly ─────────────────────────────
pl_appearances = appearances[appearances["competition_id"] == "GB1"].copy()

# Get season from games and merge
games_slim = games[["game_id", "season"]]
pl_appearances = pl_appearances.merge(games_slim, on="game_id", how="left")

# ── 3. Aggregate player stats per season ──────────────────────────────────────
stats = (
    pl_appearances.groupby(["player_id", "season"])
    .agg(
        appearances=("game_id", "count"),
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        minutes_played=("minutes_played", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
    )
    .reset_index()
)

# Keep most recent season per player
stats = stats.sort_values("season").groupby("player_id").last().reset_index()

# Only keep players from recent seasons (2022 onwards)
stats = stats[stats["season"] >= 2022]

# ── 4. Merge everything together ───────────────────────────────────────────────
df = stats.merge(latest_valuations, on="player_id", how="inner")
df = df.merge(
    players[["player_id", "name", "position", "sub_position", "foot",
             "height_in_cm", "current_club_name", "country_of_citizenship",
             "date_of_birth"]],
    on="player_id", how="left"
)

# ── 5. Feature engineering ─────────────────────────────────────────────────────
df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
df["age"] = ((pd.Timestamp.now() - df["date_of_birth"]).dt.days / 365.25).round(1)
df["goals_per_90"] = (df["goals"] / df["minutes_played"] * 90).replace([np.inf, -np.inf], 0)
df["assists_per_90"] = (df["assists"] / df["minutes_played"] * 90).replace([np.inf, -np.inf], 0)
df["minutes_per_game"] = (df["minutes_played"] / df["appearances"]).replace([np.inf, -np.inf], 0)

# Drop players with no market value or too few appearances
df = df.dropna(subset=["market_value_in_eur", "age"])
df = df[df["appearances"] >= 5]
df = df[df["market_value_in_eur"] > 0]

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)

print(f"✅ Done! {len(df)} Premier League players ready for modelling.")
print(df[["name", "age", "position", "goals", "assists", "market_value_in_eur"]].head(10))