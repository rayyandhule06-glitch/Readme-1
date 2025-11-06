import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# CONFIG
DATA_PATH = "accidents.csv"  # <-- change if needed
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {DATA_PATH} with {len(df)} rows")
print("Available columns:", list(df.columns))

# Inspect Time column
print("\nSample Time values:")
print(df["Time"].head(20))

# Convert Time to hour
df["hour"] = pd.NA  # create empty column

# Try format HH:MM:SS
try:
    df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour
except Exception as e:
    print("Warning: HH:MM:SS format error:", e)

# Try format HH:MM
if df["hour"].isna().all():
    try:
        df["hour"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour
    except Exception as e:
        print("Warning: HH:MM format error:", e)

# If still empty, extract first 2 digits
if df["hour"].isna().all():
    df["hour"] = df["Time"].astype(str).str[:2].str.extract("(\d{1,2})").astype(float)

print("Unique hours extracted:", df["hour"].dropna().unique())

# Day of week
df["day_of_week"] = df["Day_of_week"]

# Map Accident Severity to numeric
severity_map = {
    "Slight Injury": 1,
    "Serious Injury": 2,
    "Fatal": 3
}
df["severity_num"] = df["Accident_severity"].map(severity_map)

# Key columns
weather_col = "Weather_conditions"
road_col = "Road_surface_conditions"

# Aggregations
by_hour = df.groupby("hour").size()
weather = df.groupby(weather_col).agg({"hour": "count"})
avg_severity = df.groupby("hour")["severity_num"].mean()

by_road = df.groupby(road_col).agg({"hour": "count"})
avg_severity_road = df.groupby(road_col)["severity_num"].mean()

by_day = df.groupby("day_of_week").size()

# Visualizations
plt.figure()
by_hour.plot(kind="line", marker="o")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.title("Accidents by Hour of Day")
plt.grid()
plt.savefig(OUT_DIR / "accidents_by_hour.png")

plt.figure()
weather["hour"].plot(kind="bar")
plt.title("Accidents by Weather")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.savefig(OUT_DIR / "accidents_by_weather.png")

plt.figure()
avg_severity_road.plot(kind="bar")
plt.title("Avg Severity by Road Condition")
plt.ylabel("Avg Severity (1=Slight, 2=Serious, 3=Fatal)")
plt.xticks(rotation=45)
plt.savefig(OUT_DIR / "avg_severity_by_road.png")

plt.figure()
by_day.plot(kind="bar")
plt.title("Accidents by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.savefig(OUT_DIR / "accidents_by_dayofweek.png")

print("✅ Analysis complete! Charts saved in", OUT_DIR.resolve())
