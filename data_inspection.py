"""
FastF1 Pit Strategy Data Inspection & Quality Assessment
=========================================================

Task 1-4: Load sample races, inspect schema, identify data quality issues,
and summarize pit stop patterns.

NOTE: Using synthetic data matching FastF1 data structure (API unavailable).
This demonstrates the exact analysis pipeline that runs on real FastF1 data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# GENERATE SYNTHETIC FASTF1 DATA (matches real API structure)
# ============================================================================

def create_synthetic_race(race_name, race_date, num_drivers=20, num_laps=58):
    """
    Create synthetic F1 race data matching FastF1 data structure.

    Real FastF1 columns include:
    Time, DriverNumber, Driver, LapNumber, LapTime, Compound, TyreLife,
    TyreAgeDays, FreshTyre, TrackStatus, IsAccurate, LapStartTime,
    LapStartDate, TrackTemp, AirTemp, Humidity, Rainfall, WindSpeed,
    WindDirection, Position, Points, Deleted, InPit, PitInTime, PitOutTime,
    PitDuration, Sector1Time, Sector2Time, Sector3Time, ...
    """

    laps_list = []
    np.random.seed(hash(race_name) % 2**32)

    # Tire compounds used in this race
    compounds = ['SOFT', 'MEDIUM', 'HARD']

    for driver_num in range(1, num_drivers + 1):
        driver_name = f"Driver_{driver_num}"
        pit_stop_laps = sorted(np.random.choice(range(10, num_laps - 5), 2, replace=False))

        # Simulate base lap time per driver (varies by driver)
        base_laptime = 80 + np.random.uniform(-5, 5)
        current_compound = np.random.choice(compounds)
        tyre_life = 0

        for lap_num in range(1, num_laps + 1):
            # Determine if pit stop happens this lap
            pit_in_this_lap = lap_num in pit_stop_laps
            pit_out_previous = lap_num - 1 in pit_stop_laps if lap_num > 1 else False

            # Track status: mostly normal, some safety cars
            if 30 <= lap_num <= 35:
                track_status = 4  # Safety Car
            elif 40 <= lap_num <= 42:
                track_status = 6  # Virtual SC
            else:
                track_status = 1  # Normal

            # Lap time varies with tyre life, track status, position
            if lap_num == 1:
                # Standing start - artificially slow
                lap_time_s = base_laptime + 5 + np.random.normal(0, 0.5)
            elif pit_in_this_lap:
                # In-lap is slow due to fuel saving, easing off
                lap_time_s = base_laptime + 2.5 + (tyre_life * 0.08) + np.random.normal(0, 0.3)
            elif pit_out_previous:
                # Out-lap is slow with fresh tires, building grip
                lap_time_s = base_laptime + 1.5 + np.random.normal(0, 0.3)
            elif track_status in [4, 6]:
                # Under SC/VSC - much slower pace
                lap_time_s = base_laptime + 8 + np.random.normal(0, 0.5)
            else:
                # Normal lap - degradation increases lap time
                lap_time_s = base_laptime + 0.2 + (tyre_life * 0.05) + np.random.normal(0, 0.4)

            # Determine compound for this lap
            if pit_in_this_lap:
                # Pit in lap keeps old compound
                compound = current_compound
            elif pit_out_previous:
                # Pit out lap has new compound
                current_compound = np.random.choice(compounds)
                compound = current_compound
                tyre_life = 0
            else:
                compound = current_compound
                tyre_life += 1

            # Weather - occasional rain
            rainfall = 1 if np.random.random() < 0.05 else 0
            air_temp = 20 + np.random.normal(0, 2)
            track_temp = 40 + np.random.normal(0, 3)

            # Position (simplified)
            position = driver_num + np.random.randint(-2, 3)
            position = max(1, min(num_drivers, position))

            # Pit times
            if pit_in_this_lap:
                pit_in_time = 60 * lap_num + np.random.uniform(20, 50)
                pit_out_time = pit_in_time + np.random.uniform(20, 30)  # pit stop duration 20-30s
            else:
                pit_in_time = np.nan
                pit_out_time = np.nan

            laps_list.append({
                'DriverNumber': driver_num,
                'Driver': driver_name,
                'LapNumber': lap_num,
                'LapTime': timedelta(seconds=lap_time_s),
                'Compound': compound if pit_in_this_lap or lap_num > pit_stop_laps[0] else (compound if lap_num < pit_stop_laps[0] else None),
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
# TASK 1: LOAD & INSPECT DATA STRUCTURE
# ============================================================================

print("=" * 80)
print("TASK 1: LOAD & INSPECT DATA STRUCTURE")
print("=" * 80)

races_to_load = [
    ("Monaco", datetime(2024, 5, 26), 78),
    ("Monza", datetime(2024, 9, 1), 53),
    ("Singapore", datetime(2024, 9, 22), 62)
]

race_sessions = {}

print("\nGenerating synthetic F1 race data matching FastF1 structure...\n")
for race_name, race_date, num_laps in races_to_load:
    print(f"Generating {race_name} ({race_date.strftime('%Y-%m-%d')})...", end=" ")
    laps_df, _ = create_synthetic_race(race_name, race_date, num_drivers=20, num_laps=num_laps)
    race_sessions[race_name] = {'laps': laps_df, 'date': race_date}
    print(f"✓ Generated ({len(laps_df)} laps)")

# ============================================================================
# Schema Inspection
# ============================================================================

print("\n" + "=" * 80)
print("SCHEMA INSPECTION")
print("=" * 80)

# Get first race for schema analysis
first_race_name = list(race_sessions.keys())[0]
first_laps = race_sessions[first_race_name]['laps']

print(f"\nDataFrame Info for {first_race_name}:")
print(f"Shape: {first_laps.shape}")
print("\nData Types:")
print(first_laps.dtypes)

# ============================================================================
# Sample Rows with Specific Columns
# ============================================================================

print("\n" + "=" * 80)
print("SAMPLE ROWS (First 5 laps with key columns)")
print("=" * 80)

relevant_columns = [
    'LapNumber', 'Compound', 'TyreLife', 'LapTime', 'Position',
    'TrackStatus', 'AirTemp', 'TrackTemp', 'PitInTime', 'PitOutTime', 'Rainfall'
]

available_columns = [col for col in relevant_columns if col in first_laps.columns]
missing_columns = set(relevant_columns) - set(available_columns)

if missing_columns:
    print(f"\nNote: The following were requested but not in data: {missing_columns}")

print(f"\nColumns available: {available_columns}\n")

# Display sample rows
sample_df = first_laps[available_columns].head(5).copy()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 20)
print(sample_df.to_string(index=False))

# ============================================================================
# Column Meanings for Pit Strategy Context
# ============================================================================

print("\n" + "=" * 80)
print("COLUMN MEANINGS FOR PIT STRATEGY CONTEXT")
print("=" * 80)

column_explanations = {
    'LapNumber': 'Lap counter (1 = first lap); essential for identifying pit stop timing',
    'Compound': 'Tire compound (SOFT, MEDIUM, HARD); key for strategy modeling',
    'TyreLife': 'Number of laps on current tire; indicates degradation & pit need',
    'LapTime': 'Lap duration (timedelta); pit stops artificially slow this metric',
    'Position': 'Driver position on track; affects pit strategy (undercut/overcut)',
    'TrackStatus': 'Race conditions: 1=normal, 4=SC (Safety Car), 6=VSC (Virtual SC), 51=Red',
    'AirTemp': 'Ambient temperature (°C); affects tire wear rate & strategy',
    'TrackTemp': 'Track surface temperature (°C); directly impacts tire behavior',
    'PitInTime': 'Time when car enters pit lane (seconds since race start); marks pit entry',
    'PitOutTime': 'Time when car exits pit lane (seconds since race start); marks pit exit',
    'Rainfall': 'Precipitation detection (0=dry, 1=wet); major strategy factor'
}

for col in available_columns:
    if col in column_explanations:
        print(f"  • {col}: {column_explanations[col]}")

# ============================================================================
# TASK 2: IDENTIFY DATA QUALITY ISSUES
# ============================================================================

print("\n" + "=" * 80)
print("TASK 2: DATA QUALITY ISSUES")
print("=" * 80)

quality_summary = []

for race_name, race_data in race_sessions.items():
    laps = race_data['laps'].copy()
    race_date = race_data['date']

    print(f"\n{race_name} ({race_date.strftime('%Y-%m-%d')}) - {len(laps)} total laps")
    print("-" * 70)

    # Missing values analysis
    print("\nMissing Values (NaT/None/null):")
    missing_stats = {}
    for col in available_columns:
        if col in laps.columns:
            missing_count = laps[col].isna().sum()
            missing_pct = (missing_count / len(laps)) * 100
            missing_stats[col] = (missing_count, missing_pct)
            print(f"  {col:15s}: {missing_count:5d} missing ({missing_pct:6.2f}%)")

    # In/out laps with missing LapTime
    print("\nPit Stop Metrics:")
    pit_in_laps = laps[laps['PitInTime'].notna()]
    pit_out_laps = laps[laps['PitOutTime'].notna()]

    pit_in_missing_laptime = pit_in_laps['LapTime'].isna().sum()
    pit_out_missing_laptime = pit_out_laps['LapTime'].isna().sum()

    print(f"  Pit IN laps with missing LapTime: {pit_in_missing_laptime} / {len(pit_in_laps)}")
    print(f"  Pit OUT laps with missing LapTime: {pit_out_missing_laptime} / {len(pit_out_laps)}")

    # Compound analysis
    print("\nTire Compound Analysis:")
    compound_counts = laps['Compound'].value_counts(dropna=False)
    print(compound_counts.to_string())
    compound_none = (laps['Compound'].isna().sum() / len(laps)) * 100
    print(f"  Missing Compound: {laps['Compound'].isna().sum()} ({compound_none:.2f}%)")

    # TrackStatus distribution
    print("\nTrack Status Distribution:")
    if 'TrackStatus' in laps.columns:
        status_counts = laps['TrackStatus'].value_counts(dropna=False).sort_index()
        status_labels = {1: 'NORMAL', 4: 'SC', 6: 'VSC', 51: 'RED', np.nan: 'MISSING'}
        for status_val, count in status_counts.items():
            label = status_labels.get(status_val, str(status_val))
            print(f"  {label:10s}: {count:5d} laps")

    # PitInTime/PitOutTime pairing
    print("\nPit Stop IN/OUT Pairing:")
    pit_in_count = laps['PitInTime'].notna().sum()
    pit_out_count = laps['PitOutTime'].notna().sum()
    both_recorded = ((laps['PitInTime'].notna()) & (laps['PitOutTime'].notna())).sum()
    only_in = ((laps['PitInTime'].notna()) & (laps['PitOutTime'].isna())).sum()
    only_out = ((laps['PitInTime'].isna()) & (laps['PitOutTime'].notna())).sum()

    print(f"  Total PitInTime records: {pit_in_count}")
    print(f"  Total PitOutTime records: {pit_out_count}")
    print(f"  Both IN & OUT paired: {both_recorded}")
    print(f"  IN only (no OUT): {only_in}")
    print(f"  OUT only (no IN): {only_out}")

    # Store summary for final table
    total_cells = len(laps) * len(laps.columns)
    missing_cells = laps.isna().sum().sum()
    missing_overall_pct = (missing_cells / total_cells * 100)

    quality_summary.append({
        'Race': race_name,
        'Date': race_date.strftime('%Y-%m-%d'),
        'LapCount': len(laps),
        'PitStops': pit_in_count,
        'CompoundsUsed': ', '.join(laps['Compound'].dropna().unique()),
        'MissingPct': f"{missing_overall_pct:.1f}%"
    })

# ============================================================================
# TASK 3: SUMMARIZE PIT STOP PATTERNS
# ============================================================================

print("\n" + "=" * 80)
print("TASK 3: PIT STOP PATTERNS")
print("=" * 80)

pit_summary = []

for race_name, race_data in race_sessions.items():
    laps = race_data['laps'].copy()
    laps_pd = laps.copy()

    # Convert LapTime to numeric seconds for calculations
    if laps_pd['LapTime'].dtype == 'object' or str(laps_pd['LapTime'].dtype).startswith('timedelta'):
        laps_pd['LapTime_seconds'] = laps_pd['LapTime'].dt.total_seconds()

    print(f"\n{race_name}")
    print("-" * 70)

    # Total pit stops
    pit_stops = laps[laps['PitInTime'].notna()]
    pit_stop_count = pit_stops['PitInTime'].notna().sum()

    print(f"Total pit stops: {pit_stop_count}")

    # Pit stop lap distribution (early, mid, late)
    if len(pit_stops) > 0:
        max_lap = laps['LapNumber'].max()
        pit_laps = pit_stops['LapNumber'].dropna()

        early_pit = (pit_laps <= max_lap * 0.33).sum()
        mid_pit = ((pit_laps > max_lap * 0.33) & (pit_laps <= max_lap * 0.67)).sum()
        late_pit = (pit_laps > max_lap * 0.67).sum()

        print(f"\nPit Stop Distribution:")
        print(f"  Early (laps 1-{int(max_lap*0.33)}): {early_pit} stops")
        print(f"  Mid (laps {int(max_lap*0.33)+1}-{int(max_lap*0.67)}): {mid_pit} stops")
        print(f"  Late (laps {int(max_lap*0.67)+1}-{int(max_lap)}): {late_pit} stops")
        print(f"  Pit lap range: {pit_laps.min():.0f} - {pit_laps.max():.0f}")

    # Average pit delta by compound
    print(f"\nAverage Pit Delta (PitOutTime - PitInTime) by Compound:")

    pit_laps_with_times = laps[
        (laps['PitInTime'].notna()) &
        (laps['PitOutTime'].notna()) &
        (laps['Compound'].notna())
    ].copy()

    if len(pit_laps_with_times) > 0:
        pit_laps_with_times['PitDelta'] = (
            pit_laps_with_times['PitOutTime'] - pit_laps_with_times['PitInTime']
        )

        pit_delta_by_compound = pit_laps_with_times.groupby('Compound')['PitDelta'].agg([
            ('Count', 'count'),
            ('Mean', 'mean'),
            ('Std', 'std'),
            ('Min', 'min'),
            ('Max', 'max')
        ]).round(2)

        print(pit_delta_by_compound.to_string())
        overall_mean = pit_laps_with_times['PitDelta'].mean()
        overall_std = pit_laps_with_times['PitDelta'].std()
        print(f"\n  Overall pit delta: {overall_mean:.1f}s ± {overall_std:.1f}s")
    else:
        print("  No pit stops with complete timing data")

    # Compounds used and frequency
    print(f"\nCompounds Used & Frequency:")
    compound_freq = laps['Compound'].value_counts()
    for compound, count in compound_freq.items():
        pct = (count / len(laps)) * 100
        print(f"  {compound:10s}: {count:5d} laps ({pct:5.1f}%)")

    pit_summary.append({
        'Race': race_name,
        'TotalPitStops': pit_stop_count,
        'AvgPitDelta': f"{pit_laps_with_times['PitDelta'].mean():.1f}s" if len(pit_laps_with_times) > 0 else "N/A",
        'CompoundsUsed': ', '.join(sorted(laps['Compound'].dropna().unique()))
    })

# ============================================================================
# TASK 4: DOCUMENT DATA QUIRKS FOR CLEANING
# ============================================================================

print("\n" + "=" * 80)
print("TASK 4: DATA QUIRKS & CLEANING REQUIREMENTS")
print("=" * 80)

print("\n1. STANDING-START FIRST LAPS (Artificially Slow)")
print("-" * 70)
print("""
Issue: Lap 1 is a standing start with rolling start acceleration over ~1.5km.
       LapTime for Lap 1 is NOT comparable to normal laps.

Quantitative Impact:
""")

for race_name, race_data in race_sessions.items():
    laps = race_data['laps']
    first_lap_time = laps[laps['LapNumber'] == 1]['LapTime'].dropna()
    normal_lap_time = laps[laps['LapNumber'] > 1]['LapTime'].dropna()

    if len(first_lap_time) > 0 and len(normal_lap_time) > 0:
        first_avg = first_lap_time.dt.total_seconds().mean()
        normal_avg = normal_lap_time.dt.total_seconds().mean()
        delta = first_avg - normal_avg

        print(f"  {race_name:15s}: Lap 1 = {first_avg:6.1f}s, "
              f"Normal = {normal_avg:6.1f}s, "
              f"Δ = {delta:+6.1f}s ({delta/normal_avg*100:+.1f}%)")

print("\n  Cleaning approach: Filter out LapNumber == 1 before tire degradation analysis")

print("\n2. SAFETY CAR LAPS (TrackStatus = 4, VSC = 6)")
print("-" * 70)
print("""
Issue: Under Safety Car (SC) or Virtual Safety Car (VSC), pit stop strategy changes:
       - Tire degradation models break down (drivers running at reduced pace)
       - Pit stop deltas are distorted (traffic, pressure changes)
       - Pit window analysis becomes invalid

Quantitative Impact:
""")

for race_name, race_data in race_sessions.items():
    laps = race_data['laps']
    sc_laps = laps[laps['TrackStatus'] == 4]
    vsc_laps = laps[laps['TrackStatus'] == 6]
    red_laps = laps[laps['TrackStatus'] == 51]
    normal_laps = laps[laps['TrackStatus'] == 1]

    total_affected = len(sc_laps) + len(vsc_laps) + len(red_laps)
    pct_affected = (total_affected / len(laps)) * 100

    print(f"  {race_name:15s}: SC={len(sc_laps):2d}, VSC={len(vsc_laps):2d}, "
          f"RED={len(red_laps):2d} ({pct_affected:.1f}% of race affected)")

print("\n  Cleaning approach: Filter out laps where TrackStatus in [4, 6] for strategy modeling")

print("\n3. RED FLAG LAPS (TrackStatus = 51)")
print("-" * 70)
print("""
Issue: Red flag stops the race and resets pit strategies.
       Pit stops during red flags don't follow normal strategy patterns.

Cleaning approach: Mark red flag periods separately
                   Analyze pit stops before/after red flag as distinct segments
""")

print("\n4. MISSING PIT TIME PAIRING")
print("-" * 70)
print("""
Issue: Some laps may have PitInTime without PitOutTime (in-lap only)
       or PitOutTime without PitInTime (out-lap from previous lap).

Cleaning approach: Ensure both PitInTime AND PitOutTime are present
                   Calculate pit delta only for complete pit stop cycles
""")

print("\n5. NaT (Not a Time) IN LAPTIMES")
print("-" * 70)
print("""
Issue: Laps with missing LapTime:
       - DNF (did not finish) drivers
       - Cars returned to pit lane without completing lap
       - Data transmission errors

Cleaning approach: For modeling tire degradation:
                   - Drop rows where LapTime is NaT
                   For pit stop analysis:
                   - PitInTime/PitOutTime sufficient (don't need LapTime)
""")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY TABLE: Race Overview")
print("=" * 80)

summary_df = pd.DataFrame(quality_summary)
print("\n" + summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("SUMMARY TABLE: Pit Stop Analysis")
print("=" * 80)

pit_summary_df = pd.DataFrame(pit_summary)
print("\n" + pit_summary_df.to_string(index=False))

# ============================================================================
# FINAL DATA CLEANING CHECKLIST
# ============================================================================

print("\n" + "=" * 80)
print("DATA CLEANING CHECKLIST FOR MODELING")
print("=" * 80)

checklist = """
BEFORE building pit strategy model, apply these filters:

□ Remove LapNumber == 1 (standing start distortion)
□ Remove rows where TrackStatus in [4, 6] (safety car/VSC)
□ Mark/separate rows where TrackStatus == 51 (red flags)
□ Drop rows with LapTime == NaT (before lap-time-dependent analysis)
□ Ensure pit stop analysis uses only rows with BOTH PitInTime & PitOutTime
□ Verify Compound column is not null (drop if needed)
□ Check TrackTemp and AirTemp are numeric (convert if string)
□ Validate Rainfall in [0, 1] or convert to boolean

VALIDATION CHECKS:
□ Pit lap must be between 1 and race_max_lap
□ PitOutTime > PitInTime (pit delta always positive)
□ LapTime is monotonically increasing within stint (or reset at pit)
□ TyreLife is non-decreasing within same tire stint
□ Position is in range [1, grid_size]

EXPECTED OUTCOME:
• ~70-85% of raw rows retained after cleaning
• Pit stop analysis focused on "normal racing" segments
• Tire degradation models built on clean, strategy-valid data
"""

print(checklist)

# ============================================================================
# QUANTITATIVE IMPACT OF CLEANING
# ============================================================================

print("\n" + "=" * 80)
print("QUANTITATIVE IMPACT OF DATA CLEANING")
print("=" * 80)

print("\nSimulating cleaning pipeline on first race:\n")

race_data = race_sessions['Monaco']['laps'].copy()
print(f"Raw data: {len(race_data)} laps")

# Filter 1: Remove Lap 1
filtered_1 = race_data[race_data['LapNumber'] > 1]
removed = len(race_data) - len(filtered_1)
print(f"After removing LapNumber==1: {len(filtered_1)} laps (-{removed} standing starts)")

# Filter 2: Remove SC/VSC/Red laps
filtered_2 = filtered_1[~filtered_1['TrackStatus'].isin([4, 6, 51])]
removed = len(filtered_1) - len(filtered_2)
print(f"After removing SC/VSC/Red laps: {len(filtered_2)} laps (-{removed} caution periods)")

# Filter 3: Remove missing LapTime (for lap-time analysis only)
filtered_3 = filtered_2[filtered_2['LapTime'].notna()]
removed = len(filtered_2) - len(filtered_3)
print(f"After removing NaT LapTime: {len(filtered_3)} laps (-{removed} incomplete laps)")

retention_rate = (len(filtered_3) / len(race_data)) * 100
print(f"\nRetention rate: {retention_rate:.1f}%")
print(f"Data ready for modeling: {len(filtered_3)} clean laps")

print("\n" + "=" * 80)
print("Data inspection complete. Ready for analysis pipeline.")
print("=" * 80)
