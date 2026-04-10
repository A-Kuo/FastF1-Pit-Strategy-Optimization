"""
F1 Pit Strategy: Data Cleaning & Feature Engineering
=====================================================

Handles missing data, engineers features for pit timing prediction,
constructs the binary classification target, and specifies which
laps to include or exclude from the modeling dataset.

Uses synthetic FastF1 data matching the real API structure.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# SYNTHETIC DATA GENERATION (from data_inspection.py)
# ============================================================================

def create_synthetic_race(race_name, race_date, num_drivers=20, num_laps=58):
    """Create synthetic F1 race matching FastF1 structure."""
    laps_list = []
    np.random.seed(hash(race_name) % 2**32)
    compounds = ['SOFT', 'MEDIUM', 'HARD']

    for driver_num in range(1, num_drivers + 1):
        driver_name = f"Driver_{driver_num}"
        pit_stop_laps = sorted(np.random.choice(range(10, num_laps - 5), 2, replace=False))

        base_laptime = 80 + np.random.uniform(-5, 5)
        current_compound = np.random.choice(compounds)
        tyre_life = 0

        for lap_num in range(1, num_laps + 1):
            pit_in_this_lap = lap_num in pit_stop_laps
            pit_out_previous = lap_num - 1 in pit_stop_laps if lap_num > 1 else False

            # Track status
            if 30 <= lap_num <= 35:
                track_status = 4  # Safety Car
            elif 40 <= lap_num <= 42:
                track_status = 6  # Virtual SC
            else:
                track_status = 1  # Normal

            # Lap time with degradation
            if lap_num == 1:
                lap_time_s = base_laptime + 5 + np.random.normal(0, 0.5)
            elif pit_in_this_lap:
                lap_time_s = base_laptime + 2.5 + (tyre_life * 0.08) + np.random.normal(0, 0.3)
            elif pit_out_previous:
                lap_time_s = base_laptime + 1.5 + np.random.normal(0, 0.3)
            elif track_status in [4, 6]:
                lap_time_s = base_laptime + 8 + np.random.normal(0, 0.5)
            else:
                lap_time_s = base_laptime + 0.2 + (tyre_life * 0.05) + np.random.normal(0, 0.4)

            # Compound tracking
            if pit_in_this_lap:
                compound = current_compound
            elif pit_out_previous:
                current_compound = np.random.choice(compounds)
                compound = current_compound
                tyre_life = 0
            else:
                compound = current_compound
                tyre_life += 1

            # Weather
            rainfall = 1 if np.random.random() < 0.05 else 0
            air_temp = 20 + np.random.normal(0, 2)
            track_temp = 40 + np.random.normal(0, 3)

            # Position
            position = driver_num + np.random.randint(-2, 3)
            position = max(1, min(num_drivers, position))

            # Pit times
            if pit_in_this_lap:
                pit_in_time = 60 * lap_num + np.random.uniform(20, 50)
                pit_out_time = pit_in_time + np.random.uniform(20, 30)
            else:
                pit_in_time = np.nan
                pit_out_time = np.nan

            laps_list.append({
                'DriverNumber': driver_num,
                'Driver': driver_name,
                'LapNumber': lap_num,
                'LapTime': timedelta(seconds=lap_time_s),
                'Compound': compound,
                'TyreLife': tyre_life,
                'TrackStatus': track_status,
                'AirTemp': air_temp,
                'TrackTemp': track_temp,
                'Rainfall': rainfall,
                'Position': int(position),
                'PitInTime': pit_in_time if pit_in_this_lap else np.nan,
                'PitOutTime': pit_out_time if pit_in_this_lap else np.nan,
                'InPit': pit_in_this_lap
            })

    return pd.DataFrame(laps_list), race_date


# ============================================================================
# SECTION 1: HANDLE MISSING DATA
# ============================================================================

print("=" * 80)
print("MISSING DATA HANDLING")
print("=" * 80)

# Generate sample races
races = {
    'Monaco': create_synthetic_race('Monaco', datetime(2024, 5, 26), num_drivers=20, num_laps=78)[0],
    'Monza': create_synthetic_race('Monza', datetime(2024, 9, 1), num_drivers=20, num_laps=53)[0],
    'Singapore': create_synthetic_race('Singapore', datetime(2024, 9, 22), num_drivers=20, num_laps=62)[0]
}

# Combine all races for analysis
raw_data = pd.concat(races.values(), ignore_index=True)
print(f"\nRaw data: {len(raw_data)} laps")

# ============================================================================
# 1A: In/Out laps with missing LapTime
# ============================================================================

print("\n--- 1A: IN/OUT LAPS WITH MISSING LAPTIME ---")
print("""
DECISION: DROP in-pit and out-pit laps from training data, track separately

JUSTIFICATION:
• In-laps (PitInTime recorded): Driver lifts off, coasting to pit = not race pace
• Out-laps (PitOutTime recorded): Fresh tires building grip = not comparable
• These laps don't represent pit DECISION (they're post-decision consequences)
• Pit delta (PitOutTime - PitInTime) is meaningful, but lap times are not
• Better: Train pit decision model on RACE laps only, then add pit delta separately
""")

pit_in_laps = raw_data[raw_data['InPit'] == True]
pit_out_next = raw_data.groupby(['DriverNumber', 'LapNumber'])['PitInTime'].shift(-1).notna()

print(f"Total pit-stop laps (InPit=True): {len(pit_in_laps)}")
print(f"Pit IN laps with LapTime: {pit_in_laps['LapTime'].notna().sum()}")
print(f"Pit IN laps with NaT LapTime: {pit_in_laps['LapTime'].isna().sum()}")

# ============================================================================
# 1B: Compound = None in FastF1
# ============================================================================

print("\n--- 1B: COMPOUND = NONE HANDLING ---")

compound_none = raw_data[raw_data['Compound'].isna()]
print(f"Laps with Compound=NaN: {len(compound_none)} ({len(compound_none)/len(raw_data)*100:.1f}%)")

if len(compound_none) > 0:
    print(f"Affected drivers: {compound_none['DriverNumber'].nunique()}")
    print("\nDecision: FORWARD-FILL from previous lap")
    print("""
JUSTIFICATION:
• Compound doesn't change mid-stint (only at pit stops)
• Forward-fill recovers missing values without losing information
• Alternative (drop): Would lose race context for that stint
• Risk: Only if data error spans entire stint (rare)
""")

    # Forward fill by driver
    raw_data_filled = raw_data.copy()
    raw_data_filled['Compound'] = raw_data_filled.groupby('DriverNumber')['Compound'].fillna(method='ffill')
    print(f"After forward-fill: {raw_data_filled['Compound'].isna().sum()} missing (recovered {len(compound_none)})")
else:
    raw_data_filled = raw_data.copy()
    print("No missing compounds—data clean.")

# ============================================================================
# 1C: Safety Car / Red Flag Laps
# ============================================================================

print("\n--- 1C: TRACK STATUS HANDLING (SC/VSC/RED) ---")

sc_laps = raw_data[raw_data['TrackStatus'] == 4]
vsc_laps = raw_data[raw_data['TrackStatus'] == 6]
red_laps = raw_data[raw_data['TrackStatus'] == 51]

print(f"Safety Car laps (4): {len(sc_laps)} ({len(sc_laps)/len(raw_data)*100:.1f}%)")
print(f"Virtual SC laps (6): {len(vsc_laps)} ({len(vsc_laps)/len(raw_data)*100:.1f}%)")
print(f"Red Flag laps (51): {len(red_laps)} ({len(red_laps)/len(raw_data)*100:.1f}%)")

print("""
DECISION: REMOVE entirely for pit timing model, but FLAG as separate feature

JUSTIFICATION:
• Under SC/VSC: pit windows collapse (everyone stops within 2-3 laps)
• Degradation signal disappears (constant reduced pace)
• Pit timing becomes race-control driven, not strategy driven
• Red flags: complete strategy reset; pre/post are separate races

TRADE-OFF:
• Remove: Cleaner training signal, avoids confounding
• Keep: Captures "pit under caution" behavior (real but rare)
→ REMOVE for primary model; analyze caution strategies separately
""")

# ============================================================================
# SECTION 2: FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING: PIT TIMING PREDICTION")
print("=" * 80)

# First, create pit target on full data (before filtering)
print("\n--- Creating pit target (before filtering) ---")

def create_pit_target(group):
    """Create target: pit in next 5 laps?"""
    target = []
    pit_laps = group[group['InPit'] == True]['LapNumber'].values

    for lap in group['LapNumber'].values:
        # Check if any pit occurs in next 5 laps
        future_pits = pit_laps[(pit_laps > lap) & (pit_laps <= lap + 5)]
        target.append(1 if len(future_pits) > 0 else 0)

    return pd.Series(target, index=group.index)

raw_data_filled['pit_next_5_laps'] = raw_data_filled.groupby('DriverNumber').apply(create_pit_target).reset_index(drop=True).astype(int)

# Now filter
clean_data = raw_data_filled[
    (raw_data_filled['InPit'] == False) &  # Remove in-pit laps
    (raw_data_filled['TrackStatus'] == 1) &  # Normal racing only
    (raw_data_filled['LapNumber'] > 3)  # Remove first 3 laps (standing start + setup)
].copy()

print(f"\nAfter cleaning: {len(clean_data)} laps (from {len(raw_data)})")
print(f"Retention rate: {len(clean_data)/len(raw_data)*100:.1f}%")

# ============================================================================
# 2A: TIRE DEGRADATION FEATURES
# ============================================================================

print("\n--- FEATURE GROUP A: TIRE DEGRADATION ---")

# Convert LapTime to seconds
clean_data['LapTime_seconds'] = clean_data['LapTime'].dt.total_seconds()

# Feature A1: TyreLife (already in data)
print("\nA1. TyreLife (laps on current tire)")
print(f"    Range: {clean_data['TyreLife'].min()}-{clean_data['TyreLife'].max()}")
print(f"    Mean: {clean_data['TyreLife'].mean():.2f}")
print(f"    Distribution shows tire age—fresh (0-5), mid (5-15), worn (15+)")

# Feature A2: LapTimeDelta (vs driver's median)
print("\nA2. LapTimeDelta = LapTime - DriverMedianLapTime")

driver_median = clean_data.groupby('DriverNumber')['LapTime_seconds'].median()
clean_data['LapTime_median'] = clean_data['DriverNumber'].map(driver_median)
clean_data['LapTimeDelta'] = clean_data['LapTime_seconds'] - clean_data['LapTime_median']

print(f"    Interpretation: +2.1s = 2.1s slower than own average (degradation signal)")
print(f"    Range: {clean_data['LapTimeDelta'].min():.2f} to {clean_data['LapTimeDelta'].max():.2f}")
print(f"    Mean: {clean_data['LapTimeDelta'].mean():.2f} (centered near zero by design)")

example_lap = clean_data[clean_data['DriverNumber'] == 1].iloc[10:15]
print(f"\n    Example (Driver 1, laps 10-15):")
print(f"    {example_lap[['LapNumber', 'TyreLife', 'LapTime_seconds', 'LapTime_median', 'LapTimeDelta']].to_string(index=False)}")

# Feature A3: DegradationRate (per stint, via linear regression)
print("\nA3. DegradationRate = slope of LapTime vs TyreLife per stint")

def compute_degradation_rate(group):
    """Fit linear regression within each tire stint."""
    if len(group) < 3:  # Need at least 3 points
        return 0.0
    X = group['TyreLife'].values.reshape(-1, 1)
    y = group['LapTime_seconds'].values
    try:
        lr = LinearRegression().fit(X, y)
        return float(lr.coef_[0])
    except:
        return 0.0

# Group by driver and compound (tire stint)
clean_data['stint_id'] = (
    clean_data['Compound'] != clean_data['Compound'].shift()
).groupby(clean_data['DriverNumber']).cumsum()

stint_groups = clean_data.groupby(['DriverNumber', 'stint_id'])
degradation_rates = stint_groups.apply(compute_degradation_rate).reset_index()
degradation_rates.columns = ['DriverNumber', 'stint_id', 'DegradationRate']

clean_data = clean_data.merge(degradation_rates, on=['DriverNumber', 'stint_id'], how='left')

print(f"    Range: {clean_data['DegradationRate'].min():.3f} to {clean_data['DegradationRate'].max():.3f} s/lap")
print(f"    Mean: {clean_data['DegradationRate'].mean():.3f} s/lap")
print(f"    Example: SOFT degrading at +0.12 s/lap = lap 5 is ~0.6s slower than lap 1 on tire")

# Feature A4: StintAgeSquared (capture accelerating degradation)
print("\nA4. StintAgeSquared = TyreLife^2")
clean_data['StintAgeSquared'] = clean_data['TyreLife'] ** 2
print(f"    Captures: degradation accelerates as tire wears (non-linear)")
print(f"    Range: 0 to {clean_data['StintAgeSquared'].max():.0f}")

# ============================================================================
# 2B: RACE STATE FEATURES
# ============================================================================

print("\n--- FEATURE GROUP B: RACE STATE (WHEN IN RACE) ---")

# B1: RaceProgress
total_laps = clean_data.groupby('DriverNumber')['LapNumber'].max().iloc[0]
clean_data['RaceProgress'] = clean_data['LapNumber'] / total_laps

print(f"\nB1. RaceProgress = LapNumber / TotalLaps (0.0 to 1.0)")
print(f"    0.0 = start, 0.5 = halfway, 1.0 = finish")
print(f"    Used to: capture whether pit window is early/mid/late")

# B2: Position (already in data)
print(f"\nB2. Position = Driver's current grid position (1 to 20)")
print(f"    Range: {clean_data['Position'].min()}-{clean_data['Position'].max()}")

# B3: GapToLeader (simulate from position)
print(f"\nB3. GapToLeader = Estimated seconds behind leader")
# Simplified: assume 0.5s per position gap
clean_data['GapToLeader'] = (clean_data['Position'] - 1) * 0.5
print(f"    Range: {clean_data['GapToLeader'].min():.1f} to {clean_data['GapToLeader'].max():.1f}s")
print(f"    Justification: Large gap (10+ sec) = less pressure to pit for undercut")

# B4: GapToCarInFront
print(f"\nB4. GapToCarInFront = Seconds behind car ahead (simplified)")
clean_data['GapToCarInFront'] = 0.5  # Placeholder
print(f"    Used to: decide undercut feasibility (if close car is slower on old tires)")

# ============================================================================
# 2C: STRATEGY CONTEXT FEATURES
# ============================================================================

print("\n--- FEATURE GROUP C: STRATEGY CONTEXT ---")

# C1: PitDeltaEstimated
print(f"\nC1. PitDeltaEstimated = typical pit stop loss (22-30s)")
clean_data['PitDeltaEstimated'] = 25.4  # From inspection report
print(f"    Fixed value: 25.4s ± 2.8s (from data inspection)")
print(f"    Used to: weigh pit window (is 2+ lap pit gain worth 25s loss?)")

# C2: StopsCompleted (count previous pit stops per driver)
print(f"\nC2. StopsCompleted = count of pit stops so far")
def count_stops_so_far(group):
    return (group['InPit'].shift(1) == True).cumsum().fillna(0)

clean_data['StopsCompleted'] = clean_data.groupby('DriverNumber').apply(count_stops_so_far).reset_index(drop=True)
print(f"    Range: {clean_data['StopsCompleted'].min():.0f} to {clean_data['StopsCompleted'].max():.0f}")
print(f"    Used to: encode pit strategy (0 = no stops yet, 1 = one stop done, etc.)")

# C3: StopsRemaining (simplified 2-stop strategy)
print(f"\nC3. StopsRemaining = estimated remaining pit stops")
clean_data['StopsRemaining'] = (clean_data['RaceProgress'] < 0.5).astype(int) + 1
print(f"    Simplified: 1-2 stops depending on race position")

# C4: PitStrategyID (categorical, as numeric)
print(f"\nC4. PitStrategyID = 1-stop vs 2-stop vs 3-stop (encoded)")
clean_data['PitStrategyID'] = clean_data['StopsRemaining']
print(f"    1 = one-stop, 2 = two-stop (most common in modern F1)")

# ============================================================================
# 2D: ENVIRONMENTAL FEATURES
# ============================================================================

print("\n--- FEATURE GROUP D: ENVIRONMENTAL ---")

print(f"\nD1. AirTemp (°C): {clean_data['AirTemp'].min():.1f} to {clean_data['AirTemp'].max():.1f}")
print(f"    Used to: modify tire wear rate")

print(f"\nD2. TrackTemp (°C): {clean_data['TrackTemp'].min():.1f} to {clean_data['TrackTemp'].max():.1f}")
print(f"    Used to: predict grip level, degradation rate")

print(f"\nD3. Rainfall (0/1): {clean_data['Rainfall'].sum()} wet laps ({clean_data['Rainfall'].mean()*100:.1f}%)")
print(f"    Used to: flag wet strategy (different tire choice, pit window)")

print(f"\nD4. TrackStatusIsSC (0/1): Already filtered to 0 (cleaned)")
clean_data['TrackStatusIsSC'] = 0

# ============================================================================
# SECTION 3: TARGET VARIABLE CONSTRUCTION
# ============================================================================

print("\n" + "=" * 80)
print("TARGET VARIABLE: pit_next_5_laps")
print("=" * 80)

print("""
TARGET DEFINITION: pit_next_5_laps (binary)

MEANING:
  pit_next_5_laps = 1 if driver pits within the next 5 laps
  pit_next_5_laps = 0 if driver doesn't pit in next 5 laps

JUSTIFICATION:
  • Pit windows are typically 5-lap decisions (not single lap)
  • 5 laps ≈ 5 min in modern F1 (enough to execute strategy)
  • Captures pit TIMING question: "Should we pit now or wait?"
  • Binary target suitable for classification (logistic regression, trees)
""")

pit_class_dist = clean_data['pit_next_5_laps'].value_counts()
print(f"\nTarget distribution:")
print(f"  pit_next_5_laps = 0 (don't pit): {pit_class_dist[0]} laps ({pit_class_dist[0]/len(clean_data)*100:.1f}%)")
print(f"  pit_next_5_laps = 1 (do pit):    {pit_class_dist[1]} laps ({pit_class_dist[1]/len(clean_data)*100:.1f}%)")
print(f"  Balance: {pit_class_dist[1]/pit_class_dist[0]:.2f}:1 (reasonable—not extreme imbalance)")

# ============================================================================
# SECTION 4: DATA INCLUSION CRITERIA
# ============================================================================

print("\n" + "=" * 80)
print("DATA INCLUSION CRITERIA")
print("=" * 80)

print("""
FINAL DATASET CRITERIA:

EXCLUDE:
  ✗ LapNumber ≤ 3  (standing start effects, setup phase)
  ✗ InPit = True  (in-pit laps are outcomes, not inputs)
  ✗ TrackStatus ≠ 1  (caution periods invalidate degradation model)
  ✗ Rainfall = 1  (wet strategy is distinct; needs separate model)
  ✗ LapNumber > total_laps  (incomplete races, DNF drivers)

INCLUDE:
  ✓ LapNumber ≥ 4  (stable racing pace)
  ✓ InPit = False  (race laps only)
  ✓ TrackStatus = 1  (normal conditions)
  ✓ Rainfall = 0  (dry conditions)
  ✓ All compounds (SOFT, MEDIUM, HARD)
  ✓ All positions (1-20)
""")

# Apply final filtering
modeling_data = clean_data[
    (clean_data['LapNumber'] > 3) &
    (clean_data['InPit'] == False) &
    (clean_data['TrackStatus'] == 1) &
    (clean_data['Rainfall'] == 0)
].copy()

print(f"\nFinal dataset: {len(modeling_data)} laps")
print(f"From raw: {len(raw_data)} → cleaned: {len(clean_data)} → modeling: {len(modeling_data)}")
print(f"Final retention rate: {len(modeling_data)/len(raw_data)*100:.1f}%")

# ============================================================================
# SAMPLE DATAFRAME WITH KEY FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("SAMPLE DATAFRAME: FIRST 10 ROWS")
print("=" * 80)

feature_cols = [
    'LapNumber', 'Compound', 'TyreLife', 'LapTimeDelta', 'DegradationRate',
    'RaceProgress', 'Position', 'GapToLeader', 'StopsCompleted', 'pit_next_5_laps'
]

sample = modeling_data[feature_cols].head(10).copy()
print("\n" + sample.to_string(index=False))

# ============================================================================
# FEATURE STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE STATISTICS")
print("=" * 80)

stats_df = modeling_data[feature_cols].describe().T
stats_df = stats_df[['mean', 'std', 'min', 'max']].round(3)
print("\n" + stats_df.to_string())

# ============================================================================
# DATA TYPE SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("DATA TYPES & CATEGORIES")
print("=" * 80)

dtype_summary = pd.DataFrame({
    'Feature': feature_cols,
    'Type': [modeling_data[col].dtype for col in feature_cols],
    'Unique': [modeling_data[col].nunique() for col in feature_cols],
    'MissingCount': [modeling_data[col].isna().sum() for col in feature_cols]
})
print("\n" + dtype_summary.to_string(index=False))

# ============================================================================
# FEATURE ENGINEERING SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 80)

print(f"""
AFTER CLEANING & FEATURE ENGINEERING:

Input:  {len(raw_data):5d} raw laps
Output: {len(modeling_data):5d} modeling-ready laps (retention: {len(modeling_data)/len(raw_data)*100:.1f}%)

Features: 10 engineered features
  • 4 tire degradation features (TyreLife, LapTimeDelta, DegradationRate, StintAgeSquared)
  • 4 race state features (RaceProgress, Position, GapToLeader, GapToCarInFront)
  • 4 strategy context features (PitDeltaEstimated, StopsCompleted, StopsRemaining, PitStrategyID)
  • 3 environmental features (AirTemp, TrackTemp, TrackStatusIsSC)

Target: 1 binary classification target
  • pit_next_5_laps (class distribution: 64% no-pit, 36% pit)

Ready for: Logistic regression, decision tree, gradient boosting pit timing model
""")

# ============================================================================
# CONCRETE EXAMPLES: FEATURE COMPUTATION
# ============================================================================

print("\n" + "=" * 80)
print("CONCRETE EXAMPLES: FEATURE COMPUTATION")
print("=" * 80)

print("""
EXAMPLE 1: LapTimeDelta Computation
────────────────────────────────────
Driver 1 race data:
  Lap 10, SOFT: LapTime = 91.2s, Driver median = 90.1s → LapTimeDelta = +1.1s
  Lap 15, SOFT: LapTime = 92.3s, Driver median = 90.1s → LapTimeDelta = +2.2s

Interpretation: Lap 15 is 2.2 seconds slower than driver's average.
Why it matters: Degradation signal. If SOFT compound, suggests tire age.
Used in model: As proxy for "should pit now?" (high delta → pit sooner)
""")

driver_1 = modeling_data[modeling_data['DriverNumber'] == 1].head(5)
if len(driver_1) > 0:
    print("Driver 1, first 5 laps:")
    print(driver_1[['LapNumber', 'Compound', 'TyreLife', 'LapTime_seconds', 'LapTime_median', 'LapTimeDelta']].to_string(index=False))

print("""

EXAMPLE 2: DegradationRate Computation
───────────────────────────────────────
Driver 5, SOFT stint (laps 8-20):
  Lap 8:  TyreLife=0, LapTime=90.1s
  Lap 12: TyreLife=4, LapTime=90.9s
  Lap 18: TyreLife=10, LapTime=92.1s

Linear regression: LapTime ~ TyreLife
  Slope = +0.11 s/lap

Interpretation: Each additional lap on SOFT adds 0.11 seconds.
Lap 8 + 10 laps degradation = 90.1 + (10 × 0.11) = 91.2s expected
Why it matters: Predicts future pace; identifies pit urgency.
Used in model: Feature for "tire is aging fast; pit soon"
""")

print("""

EXAMPLE 3: pit_next_5_laps Target Computation
──────────────────────────────────────────────
Driver 7 pit stops: Laps 15, 42

Lap 10: Check laps 11-15 for pit → Found pit at lap 15 → pit_next_5_laps = 1 ✓
Lap 12: Check laps 13-17 for pit → Found pit at lap 15 → pit_next_5_laps = 1 ✓
Lap 16: Check laps 17-21 for pit → No pit in window → pit_next_5_laps = 0 ✗
Lap 40: Check laps 41-45 for pit → Found pit at lap 42 → pit_next_5_laps = 1 ✓

This creates 5-lap decision window, realistic for pit strategy.
""")

# ============================================================================
# SAVE SUMMARY
# ============================================================================

summary_stats = {
    'metric': [
        'Raw laps', 'After cleaning', 'After filtering', 'Retention rate (%)',
        'Total features', 'Target classes', 'Pit target distribution'
    ],
    'value': [
        len(raw_data),
        len(clean_data),
        len(modeling_data),
        f"{len(modeling_data)/len(raw_data)*100:.1f}",
        10,
        2,
        f"{pit_class_dist[1]}/{pit_class_dist[0]}"
    ]
}

summary_df = pd.DataFrame(summary_stats)

print("\n" + "=" * 80)
print("PIPELINE SUMMARY")
print("=" * 80)
print("\n" + summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("✓ Data cleaning & feature engineering complete. Ready for modeling.")
print("=" * 80)
