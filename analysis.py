"""
F1 Pit Stop Strategy Optimizer + Undercut Threat Model
2023 Bahrain GP Race · Wet Weather Strategy Analysis

Author : Yoon
Data   : FastF1 (https://github.com/theOehrly/Fast-F1)

Outputs:
  outputs/bahrain_strategy_optimizer.png   — dry strategy + undercut model
  outputs/wet_weather_strategy.png         — Japanese + Russian GP weather
  outputs/weather_pit_model_full.png       — wet pit decision model

Critical self-evaluation:
- Dry optimizer: valid physics, but ignores undercut dynamics (addressed here)
- Undercut model: genuinely novel — invisible from standard timing screens
- Weather model: detects rain conditions, NOT true forecasting
  (humidity/tracktemp are consequences of rain, not predictors)
  Honest limitation: single-race training, Hungarian GP fails due to
  diurnal cooling being indistinguishable from rain cooling without
  multi-race training data
"""

import os
import warnings
warnings.filterwarnings('ignore')

import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from itertools import product
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score)
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score,
                              precision_score, recall_score)

# ── Cache & sessions ──────────────────────────────────────────────────────────
os.makedirs('f1_cache', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

fastf1.Cache.enable_cache('f1_cache')

# Dry race for strategy optimizer
session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load(laps=True, telemetry=False, weather=False, messages=False)
laps = session.laps
laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

# Wet races for weather analysis
session_rus = fastf1.get_session(2021, 'Russian Grand Prix', 'R')
session_rus.load(laps=True, telemetry=False, weather=True, messages=False)

session_jpn = fastf1.get_session(2022, 'Japanese Grand Prix', 'R')
session_jpn.load(laps=True, telemetry=False, weather=True, messages=False)

session_hun = fastf1.get_session(2021, 'Hungarian Grand Prix', 'R')
session_hun.load(laps=True, telemetry=False, weather=True, messages=False)

# ════════════════════════════════════════════════════════════════════════════
# PART 1: DRY STRATEGY OPTIMIZER
# ════════════════════════════════════════════════════════════════════════════

# ── Clean laps ────────────────────────────────────────────────────────────────
clean = laps[
    laps['Compound'].isin(['SOFT', 'MEDIUM', 'HARD']) &
    laps['LapTimeSeconds'].notna() &
    laps['TyreLife'].notna() &
    (laps['LapTimeSeconds'] < 110) &
    (laps['LapTimeSeconds'] > 93)
].copy()

# ── Degradation model ─────────────────────────────────────────────────────────
def linear_deg(tyre_life, base_pace, deg_rate):
    return base_pace + deg_rate * tyre_life

compound_models = {}
for compound in ['SOFT', 'HARD']:
    data = clean[
        (clean['Compound'] == compound) &
        (clean['TyreLife'] > 1) &
        (clean['TyreLife'] <= 30)
    ]
    popt, _ = curve_fit(linear_deg,
                         data['TyreLife'].values,
                         data['LapTimeSeconds'].values,
                         p0=[97, 0.05])
    compound_models[compound] = {'base_pace': popt[0], 'deg_rate': popt[1]}

compound_models['MEDIUM'] = {
    'base_pace': (compound_models['SOFT']['base_pace'] +
                  compound_models['HARD']['base_pace']) / 2,
    'deg_rate':  (compound_models['SOFT']['deg_rate'] +
                  compound_models['HARD']['deg_rate']) / 2
}

print("── Degradation Models ────────────────────────────────────────────")
for c, m in compound_models.items():
    print(f"{c:6s}: Base={m['base_pace']:.3f}s, Deg={m['deg_rate']:.4f}s/lap")

# ── Pit stop parameters ───────────────────────────────────────────────────────
PIT_LOSS_V2 = 24.0
MIN_STINT   = 10
TOTAL_LAPS  = 57

def simulate_stint_v2(compound, start_tyre_life, num_laps):
    times = []
    cliff = {'SOFT': 20, 'MEDIUM': 28, 'HARD': 38}
    for i in range(num_laps):
        tl   = start_tyre_life + i
        base = linear_deg(tl, **compound_models[compound])
        if i == 0:
            base += 3.5
        elif i == 1:
            base += 1.0
        if tl > cliff.get(compound, 35):
            base += 0.08 * (tl - cliff[compound]) ** 1.5
        times.append(base)
    return sum(times)

def simulate_one_stop_v2(pit_lap, c1='SOFT', c2='HARD', start_tl=1):
    if pit_lap < MIN_STINT or pit_lap > TOTAL_LAPS - MIN_STINT:
        return np.inf
    return (simulate_stint_v2(c1, start_tl, pit_lap - 1) +
            PIT_LOSS_V2 +
            simulate_stint_v2(c2, 1, TOTAL_LAPS - pit_lap))

def simulate_two_stop_v2(pit1_lap, pit2_lap,
                          c1='SOFT', c2='MEDIUM', c3='HARD', start_tl=1):
    if (pit1_lap < MIN_STINT or
        pit2_lap - pit1_lap < MIN_STINT or
        TOTAL_LAPS - pit2_lap < MIN_STINT):
        return np.inf
    return (simulate_stint_v2(c1, start_tl, pit1_lap - 1) + PIT_LOSS_V2 +
            simulate_stint_v2(c2, 1, pit2_lap - pit1_lap) + PIT_LOSS_V2 +
            simulate_stint_v2(c3, 1, TOTAL_LAPS - pit2_lap))

# Sweep pit windows
one_stop_v2 = {p: simulate_one_stop_v2(p)
               for p in range(MIN_STINT, TOTAL_LAPS - MIN_STINT + 1)}
two_stop_v2 = {}
for p1 in range(MIN_STINT, TOTAL_LAPS - 2 * MIN_STINT):
    for p2 in range(p1 + MIN_STINT, TOTAL_LAPS - MIN_STINT):
        two_stop_v2[(p1, p2)] = simulate_two_stop_v2(p1, p2)

best_1s_lap  = min(one_stop_v2, key=one_stop_v2.get)
best_1s_time = one_stop_v2[best_1s_lap]
best_2s_laps = min(two_stop_v2, key=two_stop_v2.get)
best_2s_time = two_stop_v2[best_2s_laps]

print(f"\n── Optimal Strategies ────────────────────────────────────────────")
print(f"1-Stop: Pit lap {best_1s_lap:2d} → {best_1s_time:.1f}s")
print(f"2-Stop: Pit laps {best_2s_laps[0]:2d} & {best_2s_laps[1]:2d} → {best_2s_time:.1f}s")

# ── Actual pit laps ───────────────────────────────────────────────────────────
top_drivers_race = ['VER', 'SAI', 'PER', 'ALO', 'HAM']
actual_pits = {}
for drv in top_drivers_race:
    drv_laps = laps[laps['Driver'] == drv].sort_values('LapNumber')
    pit_laps = drv_laps[drv_laps['PitOutTime'].notna()]['LapNumber'].tolist()
    actual_pits[drv] = [int(p) for p in pit_laps if p > 1]

print("\n── Actual Pit Laps ───────────────────────────────────────────────")
for drv, pits in actual_pits.items():
    t = simulate_two_stop_v2(pits[0], pits[1]) if len(pits) == 2 else np.inf
    delta = t - best_2s_time if t < np.inf else np.inf
    print(f"{drv}: laps {pits} → +{delta:.1f}s vs optimal")

# ════════════════════════════════════════════════════════════════════════════
# PART 2: UNDERCUT THREAT MODEL
# ════════════════════════════════════════════════════════════════════════════

race_laps = laps.copy()
race_clean = laps[
    laps['LapTimeSeconds'].notna() &
    (laps['LapTimeSeconds'] < 110) &
    (laps['LapTimeSeconds'] > 93) &
    laps['Position'].notna()
].copy()

race_sorted = race_clean.sort_values(['Driver', 'LapNumber']).copy()
race_sorted['CumTime'] = race_sorted.groupby('Driver')['LapTimeSeconds'].cumsum()

deg_rates = {'SOFT': 0.0102, 'MEDIUM': 0.0127, 'HARD': 0.0153}
RACING_GAP_MAX = 40.0

def estimate_tire_advantage_final(compound, tyre_life, laps_ahead=15):
    deg_rate = deg_rates.get(compound, 0.012)
    current  = sum(deg_rate * (tyre_life + i) for i in range(laps_ahead))
    fresh    = sum(deg_rate * i for i in range(laps_ahead))
    grip     = 0.3 * min(laps_ahead, 5)
    warmup   = 3.5 + 1.0
    return max(0, current - fresh + grip - warmup)

undercut_final = []
for lap_num in sorted(race_sorted['LapNumber'].unique()):
    lap_data = race_sorted[race_sorted['LapNumber'] == lap_num].copy()
    lap_data = lap_data.sort_values('Position').reset_index(drop=True)

    for _, row in lap_data.iterrows():
        if row['Position'] <= 1:
            continue
        ahead_mask = lap_data['Position'] == (row['Position'] - 1)
        if not ahead_mask.any():
            continue
        ahead = lap_data[ahead_mask].iloc[0]
        if pd.isna(row['CumTime']) or pd.isna(ahead['CumTime']):
            continue

        gap = abs(row['CumTime'] - ahead['CumTime'])
        if gap > RACING_GAP_MAX:
            continue

        compound  = row['Compound'] if pd.notna(row['Compound']) else 'HARD'
        tyre_life = row['TyreLife']  if pd.notna(row['TyreLife'])  else 10

        tire_adv      = estimate_tire_advantage_final(compound, tyre_life)
        undercut_gain = tire_adv - PIT_LOSS_V2 + gap

        undercut_final.append({
            'LapNumber':      lap_num,
            'Driver':         row['Driver'],
            'Position':       int(row['Position']),
            'DriverAhead':    ahead['Driver'],
            'GapToAhead':     round(gap, 3),
            'Compound':       compound,
            'TyreLife':       tyre_life,
            'TireAdvantage':  round(tire_adv, 3),
            'UndercutGain':   round(undercut_gain, 3),
            'UndercutViable': int(undercut_gain > 0),
        })

undercut_df = pd.DataFrame(undercut_final)
viable = undercut_df[undercut_df['UndercutViable'] == 1]

print(f"\n── Undercut Model ────────────────────────────────────────────────")
print(f"Total observations: {len(undercut_df)}")
print(f"Viable opportunities: {len(viable)} ({len(viable)/len(undercut_df)*100:.1f}%)")
print(viable.sort_values('UndercutGain', ascending=False)
      .head(10)[['LapNumber','Driver','DriverAhead',
                  'GapToAhead','TyreLife','TireAdvantage','UndercutGain']]
      .round(2).to_string())

# ════════════════════════════════════════════════════════════════════════════
# PART 3: WET WEATHER MODEL
# ════════════════════════════════════════════════════════════════════════════

def engineer_features_v2(laps_df, weather_df):
    df = laps_df.copy()
    w  = weather_df.copy()

    w['RainFlag']       = w['Rainfall'].astype(int)
    w['HumidNorm']      = (w['Humidity'] - w['Humidity'].min()) / \
                           (w['Humidity'].max() - w['Humidity'].min() + 1e-6)
    w['TempDropNorm']   = 1 - (w['TrackTemp'] - w['TrackTemp'].min()) / \
                               (w['TrackTemp'].max() - w['TrackTemp'].min() + 1e-6)
    w['IntensityScore'] = (0.6 * w['RainFlag'] +
                            0.25 * w['HumidNorm'] +
                            0.15 * w['TempDropNorm'])

    w['TrackTempDropFromMax']    = w['TrackTemp'].max() - w['TrackTemp']
    w['HumidityRiseFromMin']     = w['Humidity'] - w['Humidity'].min()
    w['TrackAirGap']             = w['TrackTemp'] - w['AirTemp']
    w['TrackAirGapDropFromMax']  = w['TrackAirGap'].max() - w['TrackAirGap']
    w['TrackAirGap_Roll3']       = w['TrackAirGap'].rolling(3, min_periods=1).mean()
    w['TrackAirGap_Delta']       = w['TrackAirGap'].diff().fillna(0)
    w['TempDrop_Roll3']          = w['TrackTempDropFromMax'].rolling(3, min_periods=1).mean()
    w['HumidRise_Roll3']         = w['HumidityRiseFromMin'].rolling(3, min_periods=1).mean()
    w['TempDrop_Delta']          = w['TrackTempDropFromMax'].diff().fillna(0)
    w['HumidRise_Delta']         = w['HumidityRiseFromMin'].diff().fillna(0)
    w['Intensity_Roll3']         = w['IntensityScore'].rolling(3, min_periods=1).mean()
    w['Intensity_Delta']         = w['IntensityScore'].diff().fillna(0)

    cols = ['IntensityScore', 'RainFlag', 'TrackTempDropFromMax',
            'HumidityRiseFromMin', 'TrackAirGap', 'TrackAirGapDropFromMax',
            'TrackAirGap_Roll3', 'TrackAirGap_Delta', 'TempDrop_Roll3',
            'HumidRise_Roll3', 'TempDrop_Delta', 'HumidRise_Delta',
            'Intensity_Roll3', 'Intensity_Delta', 'WindSpeed',
            'Humidity', 'TrackTemp', 'AirTemp']

    weather_times = w['Time'].values
    merged = {col: [] for col in cols}
    for lap_time in df['Time']:
        idx = np.argmin(np.abs(weather_times - lap_time))
        for col in cols:
            merged[col].append(w[col].iloc[idx])
    for col in cols:
        df[col] = merged[col]

    df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
    return df

def label_wet_conditions(laps_df):
    df = laps_df.copy()
    wet_compounds = ['INTERMEDIATE', 'WET']
    lap_wet = df.groupby('LapNumber').apply(
        lambda x: x['Compound'].isin(wet_compounds).mean()
    ).reset_index()
    lap_wet.columns = ['LapNumber', 'WetFraction']
    lap_wet['WetConditions'] = (lap_wet['WetFraction'] > 0.4).astype(int)
    df = df.merge(lap_wet[['LapNumber', 'WetConditions', 'WetFraction']],
                  on='LapNumber', how='left')
    return df

print("\nProcessing wet race sessions...")
rus_lab = label_wet_conditions(engineer_features_v2(
    session_rus.laps, session_rus.weather_data))
jpn_lab = label_wet_conditions(engineer_features_v2(
    session_jpn.laps, session_jpn.weather_data))
hun_lab = label_wet_conditions(engineer_features_v2(
    session_hun.laps, session_hun.weather_data))
print("Done!")

features_realtime = [
    'IntensityScore', 'Intensity_Roll3', 'Intensity_Delta',
    'TrackTempDropFromMax', 'TempDrop_Roll3', 'TempDrop_Delta',
    'HumidityRiseFromMin', 'HumidRise_Roll3', 'HumidRise_Delta',
    'TrackAirGap', 'TrackAirGap_Roll3', 'TrackAirGap_Delta', 'WindSpeed',
]
features_forecast = [
    'TrackTempDropFromMax', 'TempDrop_Roll3', 'TempDrop_Delta',
    'HumidityRiseFromMin', 'HumidRise_Roll3', 'HumidRise_Delta',
    'TrackAirGap', 'TrackAirGap_Roll3', 'TrackAirGap_Delta', 'WindSpeed',
]

train_df = rus_lab[
    rus_lab['LapTimeSeconds'].notna() &
    (rus_lab['LapTimeSeconds'] < 130)
].dropna(subset=features_realtime + ['WetConditions'])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_weather = {}

print("\n── Weather Model Training ────────────────────────────────────────")
for name, feats in [('Real-time', features_realtime),
                     ('Forecast',  features_forecast)]:
    X = train_df[feats]
    y = train_df['WetConditions']
    clf_w = GradientBoostingClassifier(
        n_estimators=300, max_depth=3,
        learning_rate=0.03, random_state=42,
        subsample=0.8, min_samples_leaf=3)
    cv_f1 = cross_val_score(clf_w, X, y, cv=cv, scoring='f1')
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    clf_w.fit(X_tr, y_tr)
    results_weather[name] = {
        'model': clf_w, 'features': feats,
        'f1': cv_f1.mean(), 'f1_std': cv_f1.std()
    }
    print(f"{name}: CV F1={cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

print("\n── Cross-Race Validation ─────────────────────────────────────────")
for race_name, race_df in [('2022 Japanese GP', jpn_lab),
                             ('2021 Hungarian GP', hun_lab)]:
    for model_name, model_info in results_weather.items():
        feats  = model_info['features']
        clf_w  = model_info['model']
        valid  = race_df[race_df['LapTimeSeconds'].notna()]\
                   .dropna(subset=feats + ['WetConditions'])
        if len(valid) == 0:
            continue
        y_true = valid['WetConditions']
        y_pred = clf_w.predict(valid[feats])
        f1  = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        print(f"{race_name} | {model_name}: F1={f1:.3f}, Acc={acc:.3f}")

# ════════════════════════════════════════════════════════════════════════════
# VISUALIZATION — DRY STRATEGY + UNDERCUT
# ════════════════════════════════════════════════════════════════════════════

BG   = '#ffffff'
TEXT = '#111111'
GRID = '#dddddd'

compound_colors = {'SOFT': '#E8002D', 'MEDIUM': '#FFA500',
                   'HARD': '#444444', 'INTERMEDIATE': '#39B54A', 'WET': '#0067FF'}
driver_colors_p4 = {
    'VER': '#1E41FF', 'PER': '#6AB4FF', 'ALO': '#00A39A',
    'SAI': '#E8002D', 'LEC': '#FF6B6B', 'HAM': '#00D2BE',
    'RUS': '#00A39A', 'NOR': '#FF8000', 'PIA': '#FFD700',
}

fig = plt.figure(figsize=(22, 24), facecolor=BG)
fig.suptitle(
    '2023 Bahrain GP — Pit Stop Strategy: Optimizer + Undercut Threat Model\n'
    'Combining degradation physics with real-time positional analysis',
    color=TEXT, fontsize=14, y=0.98
)
gs = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

# Panel 1: Degradation curves
ax_deg = fig.add_subplot(gs[0, 0])
ax_deg.set_facecolor(BG)
tyre_life_range = np.linspace(1, 40, 200)
cliff = {'SOFT': 20, 'MEDIUM': 28, 'HARD': 38}
for compound in ['SOFT', 'MEDIUM', 'HARD']:
    pace = []
    for t in tyre_life_range:
        p = linear_deg(t, **compound_models[compound])
        if t > cliff[compound]:
            p += 0.08 * (t - cliff[compound]) ** 1.5
        pace.append(p)
    ax_deg.plot(tyre_life_range, pace, color=compound_colors[compound],
                label=compound, linewidth=2)
    ax_deg.axvline(cliff[compound], color=compound_colors[compound],
                   linewidth=0.8, linestyle=':', alpha=0.5)
ax_deg.set_xlabel('Tyre Life (laps)', color=TEXT, fontsize=9)
ax_deg.set_ylabel('Predicted Lap Time (s)', color=TEXT, fontsize=9)
ax_deg.set_title('Tyre Degradation Model\n(dotted = degradation cliff)',
                 color=TEXT, fontsize=10)
ax_deg.set_ylim(97, 108)
ax_deg.tick_params(colors=TEXT)
ax_deg.legend(facecolor=BG, labelcolor=TEXT, fontsize=9)
ax_deg.set_axisbelow(True)
ax_deg.grid(color=GRID, linewidth=0.5, linestyle='--')
for spine in ax_deg.spines.values():
    spine.set_color(GRID)

# Panel 2: 1-stop pit window
ax_1s = fig.add_subplot(gs[0, 1])
ax_1s.set_facecolor(BG)
pit_laps_range = list(one_stop_v2.keys())
times_1s_norm  = np.array([one_stop_v2[p] for p in pit_laps_range])
times_1s_norm  = times_1s_norm - times_1s_norm.min()
ax_1s.plot(pit_laps_range, times_1s_norm, color='#1E41FF', linewidth=2)
ax_1s.fill_between(pit_laps_range, times_1s_norm, alpha=0.15, color='#1E41FF')
ax_1s.axvline(best_1s_lap, color='red', linewidth=1.5,
              linestyle='--', label=f'Optimal: Lap {best_1s_lap}')
ax_1s.set_xlabel('Pit Lap', color=TEXT, fontsize=9)
ax_1s.set_ylabel('Time Above Optimal (s)', color=TEXT, fontsize=9)
ax_1s.set_title(f'1-Stop Pit Window\n(Optimal: Lap {best_1s_lap} · SOFT→HARD)',
                color=TEXT, fontsize=10)
ax_1s.tick_params(colors=TEXT)
ax_1s.legend(facecolor=BG, labelcolor=TEXT, fontsize=9)
ax_1s.set_axisbelow(True)
ax_1s.grid(color=GRID, linewidth=0.5, linestyle='--')
for spine in ax_1s.spines.values():
    spine.set_color(GRID)

# Panel 3: 2-stop heatmap
ax_2s = fig.add_subplot(gs[1, :])
ax_2s.set_facecolor(BG)
p1_range = sorted(set(k[0] for k in two_stop_v2 if two_stop_v2[k] < np.inf))
p2_range = sorted(set(k[1] for k in two_stop_v2 if two_stop_v2[k] < np.inf))
matrix   = np.full((len(p2_range), len(p1_range)), np.nan)
for (p1, p2), t in two_stop_v2.items():
    if t < np.inf and p1 in p1_range and p2 in p2_range:
        matrix[p2_range.index(p2), p1_range.index(p1)] = t - best_2s_time

im = ax_2s.imshow(matrix, aspect='auto', cmap='RdYlGn_r',
                  origin='lower', vmin=0, vmax=60,
                  extent=[p1_range[0], p1_range[-1],
                          p2_range[0], p2_range[-1]])
ax_2s.scatter(best_2s_laps[0], best_2s_laps[1], color='white',
              s=200, marker='*', zorder=5,
              label=f'Optimal: Laps {best_2s_laps[0]} & {best_2s_laps[1]}')

for drv, pits in actual_pits.items():
    if len(pits) == 2:
        color = driver_colors_p4.get(drv, '#aaaaaa')
        ax_2s.scatter(pits[0], pits[1], color=color, s=120, zorder=4,
                      edgecolors='white', linewidth=1)
        ax_2s.text(pits[0] + 0.3, pits[1] + 0.5, drv,
                   color='white', fontsize=8, fontweight='bold')

cbar = fig.colorbar(im, ax=ax_2s, orientation='vertical', pad=0.01)
cbar.set_label('Extra time vs optimal (s)', color=TEXT, fontsize=8)
cbar.ax.tick_params(colors=TEXT)
ax_2s.set_xlabel('First Pit Lap', color=TEXT, fontsize=9)
ax_2s.set_ylabel('Second Pit Lap', color=TEXT, fontsize=9)
ax_2s.set_title('2-Stop Strategy Heatmap — All Pit Window Combinations\n'
                'Green = faster · Red = slower · ★ = Model Optimal',
                color=TEXT, fontsize=10)
ax_2s.tick_params(colors=TEXT)
ax_2s.legend(facecolor='#333333', labelcolor='white', fontsize=8,
             loc='upper right')

# Panel 4: Undercut opportunity timeline
ax_ucut = fig.add_subplot(gs[2, :])
ax_ucut.set_facecolor(BG)

key_battles    = [('ALO', 'PER'), ('PER', 'VER'), ('SAI', 'PER')]
battle_colors  = ['#00A39A', '#6AB4FF', '#E8002D']

for (drv, ahead), color in zip(key_battles, battle_colors):
    battle = undercut_df[
        (undercut_df['Driver'] == drv) &
        (undercut_df['DriverAhead'] == ahead)
    ].sort_values('LapNumber')
    if len(battle) == 0:
        continue
    ax_ucut.plot(battle['LapNumber'], battle['GapToAhead'],
                 color=color, linewidth=1.5, label=f'{drv} vs {ahead}')
    vb = battle[battle['UndercutViable'] == 1]
    if len(vb) > 0:
        ax_ucut.scatter(vb['LapNumber'], vb['GapToAhead'],
                        color=color, s=60, zorder=5, marker='v',
                        edgecolors='white', linewidth=0.8)

for drv, pits in actual_pits.items():
    for pit in pits:
        ax_ucut.axvline(pit, color=driver_colors_p4.get(drv, '#aaaaaa'),
                        linewidth=0.6, linestyle=':', alpha=0.4)

ax_ucut.axhline(PIT_LOSS_V2, color='grey', linewidth=1, linestyle='--',
                alpha=0.6, label=f'Pit loss ({PIT_LOSS_V2}s)')
ax_ucut.axhline(1.0, color='orange', linewidth=1, linestyle='--',
                alpha=0.6, label='DRS range (1s)')
ax_ucut.set_xlabel('Lap Number', color=TEXT, fontsize=9)
ax_ucut.set_ylabel('Gap to Car Ahead (s)', color=TEXT, fontsize=9)
ax_ucut.set_title('Undercut Opportunity Timeline — Key Battles\n'
                  '▼ = Viable undercut window · Dotted lines = actual pit laps',
                  color=TEXT, fontsize=10)
ax_ucut.set_ylim(-1, 42)
ax_ucut.tick_params(colors=TEXT)
ax_ucut.legend(facecolor=BG, labelcolor=TEXT, fontsize=8, loc='upper right')
ax_ucut.set_axisbelow(True)
ax_ucut.grid(color=GRID, linewidth=0.5, linestyle='--')
for spine in ax_ucut.spines.values():
    spine.set_color(GRID)

# Panel 5: Driver strategy vs optimal
ax_drv = fig.add_subplot(gs[3, 0])
ax_drv.set_facecolor(BG)
drv_deltas, drv_names_plot = [], []
for drv, pits in actual_pits.items():
    if len(pits) == 2:
        t = simulate_two_stop_v2(pits[0], pits[1])
        if t < np.inf:
            drv_deltas.append(t - best_2s_time)
            drv_names_plot.append(drv)

bars = ax_drv.bar(drv_names_plot, drv_deltas,
                  color=[driver_colors_p4.get(d, '#aaaaaa') for d in drv_names_plot],
                  edgecolor=GRID, linewidth=0.5)
ax_drv.axhline(0, color='green', linewidth=1.5, linestyle='--',
               label='Model Optimal')
for bar, delta in zip(bars, drv_deltas):
    ax_drv.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f'+{delta:.1f}s', ha='center', color=TEXT, fontsize=9)
ax_drv.set_ylabel('Time Above Model Optimal (s)', color=TEXT, fontsize=9)
ax_drv.set_title('Actual vs Model Optimal Strategy', color=TEXT, fontsize=10)
ax_drv.tick_params(colors=TEXT)
ax_drv.legend(facecolor=BG, labelcolor=TEXT, fontsize=8)
ax_drv.set_axisbelow(True)
ax_drv.grid(axis='y', color=GRID, linewidth=0.5, linestyle='--')
for spine in ax_drv.spines.values():
    spine.set_color(GRID)

# Panel 6: Undercut heatmap
ax_uheat = fig.add_subplot(gs[3, 1])
ax_uheat.set_facecolor(BG)

top_drv_heat = ['VER', 'PER', 'ALO', 'SAI', 'LEC', 'HAM', 'RUS', 'NOR']
pivot_data   = []
for drv in top_drv_heat:
    drv_laps = undercut_df[undercut_df['Driver'] == drv]
    for lap in range(1, 58):
        lap_row = drv_laps[drv_laps['LapNumber'] == lap]
        gain    = lap_row['UndercutGain'].values[0] if len(lap_row) > 0 else 0
        pivot_data.append({'Driver': drv, 'Lap': lap, 'Gain': gain})

pivot_df   = pd.DataFrame(pivot_data)
pivot_mat  = pivot_df.pivot(index='Driver', columns='Lap', values='Gain').fillna(0)
lap_cols   = [c for c in pivot_mat.columns if c % 5 == 0]
pivot_show = pivot_mat[lap_cols]

im2 = ax_uheat.imshow(pivot_show.values, aspect='auto', cmap='RdYlGn',
                      vmin=-25, vmax=15, interpolation='nearest')
ax_uheat.set_xticks(range(len(lap_cols)))
ax_uheat.set_xticklabels([int(c) for c in lap_cols], fontsize=7, color=TEXT)
ax_uheat.set_yticks(range(len(top_drv_heat)))
ax_uheat.set_yticklabels(top_drv_heat, fontsize=8, color=TEXT)
ax_uheat.set_xlabel('Lap Number', color=TEXT, fontsize=9)
ax_uheat.set_title('Undercut Opportunity Heatmap\nGreen = viable · Red = not viable',
                   color=TEXT, fontsize=10)
cbar2 = fig.colorbar(im2, ax=ax_uheat, orientation='vertical', pad=0.01)
cbar2.set_label('Undercut Gain (s)', color=TEXT, fontsize=7)
cbar2.ax.tick_params(colors=TEXT, labelsize=7)

plt.savefig('outputs/bahrain_strategy_optimizer.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: outputs/bahrain_strategy_optimizer.png")
