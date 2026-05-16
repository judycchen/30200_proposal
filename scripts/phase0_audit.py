import os
import re
import glob
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR     = '/project/ycleong/datasets/CogTasks-NNW_judy'
SCRIPT_PATH = '/project/ycleong/users/judycchen/prediction-proj/scripts/b1_preproc_fnirs.m'

TASK_LABELS = {
    '001': 'Nback',
    '002': 'CPT1',
    '003': 'CPTMEM1',
    '004': 'CPT2',
    '005': 'CPTMEM2',
    '006': 'NNW',
}
TASKS = list(TASK_LABELS.keys())

MIN_DURATION_S = 200   # minimum usable seconds for 0.005 Hz bandpass
RAW_FS         = 5.086 # nominal raw sampling rate (Hz)

# ── 1. Find all subject folders ────────────────────────────────────────────────
subj_folders = sorted(glob.glob(os.path.join(RAW_DIR, 'p*_fnirs_cogTasks')))
subj_ids     = [os.path.basename(f) for f in subj_folders]
print(f"\nFound {len(subj_ids)} subject folders:")
for s in subj_ids:
    print(f"  {s}")

# ── 2. Subject × Task availability matrix ─────────────────────────────────────
print("\n" + "="*70)
print("SUBJECT × TASK AVAILABILITY")
print("="*70)

# Header
header = f"{'Subject':<20}" + "".join(f"{TASK_LABELS[t]:>10}" for t in TASKS)
print(header)
print("-" * len(header))

availability = {}   # availability[subj][task] = path or None
for subj_folder in subj_folders:
    subj = os.path.basename(subj_folder)
    availability[subj] = {}
    row = f"{subj:<20}"
    for task in TASKS:
        # Task folders are named like 2026-XX-XX_00{task}
        pattern = os.path.join(subj_folder, f'*_{task}')
        matches = glob.glob(pattern)
        if matches:
            availability[subj][task] = matches[0]
            row += f"{'YES':>10}"
        else:
            availability[subj][task] = None
            row += f"{'MISSING':>10}"
    print(row)

# Summary: how many subjects have each task
print("\nSubjects available per task:")
for task in TASKS:
    n = sum(1 for s in availability if availability[s][task] is not None)
    print(f"  Task {task} ({TASK_LABELS[task]}): {n}/{len(subj_ids)} subjects")

# ── 3. Recording duration per subject per task ─────────────────────────────────
print("\n" + "="*70)
print("RECORDING DURATION CHECK (threshold = 200s)")
print("="*70)

# Try to read duration from .snirf file using h5py (lightweight, no toolbox needed)
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("  [WARNING] h5py not installed — falling back to file size estimate only")

duration_matrix = {}   # duration_matrix[subj][task] = duration in seconds or None

header2 = f"{'Subject':<20}" + "".join(f"{TASK_LABELS[t]:>12}" for t in TASKS)
print(header2)
print("-" * len(header2))

flags = []  # collect (subj, task, duration) for tasks below threshold

for subj_folder in subj_folders:
    subj = os.path.basename(subj_folder)
    duration_matrix[subj] = {}
    row = f"{subj:<20}"

    for task in TASKS:
        task_path = availability[subj].get(task)
        if task_path is None:
            duration_matrix[subj][task] = None
            row += f"{'N/A':>12}"
            continue

        # Find the .snirf file in this task folder
        snirf_files = glob.glob(os.path.join(task_path, '*.snirf'))
        if not snirf_files:
            duration_matrix[subj][task] = None
            row += f"{'NO SNIRF':>12}"
            continue

        snirf_path = snirf_files[0]
        dur_s = None

        if HAS_H5PY:
            try:
                with h5py.File(snirf_path, 'r') as f:
                    # SNIRF standard: time vector at /nirs/data1/time
                    # or derive from dataTimeSeries shape + metaDataTags/MeasurementDate
                    if 'nirs' in f:
                        nirs_keys = list(f['nirs'].keys())
                        data_key = next((k for k in nirs_keys if k.startswith('data')), None)
                        if data_key:
                            data_grp = f['nirs'][data_key]
                            if 'time' in data_grp:
                                t = data_grp['time'][:]
                                dur_s = float(t[-1] - t[0])
                            elif 'dataTimeSeries' in data_grp:
                                n_samples = data_grp['dataTimeSeries'].shape[0]
                                dur_s = n_samples / RAW_FS
            except Exception as e:
                dur_s = None

        if dur_s is None:
            # Fallback: estimate from file size (very rough)
            size_mb = os.path.getsize(snirf_path) / 1e6
            row += f"{'~'+str(round(size_mb,1))+'MB':>12}"
            duration_matrix[subj][task] = None
            continue

        duration_matrix[subj][task] = dur_s
        flag = ' !!!' if dur_s < MIN_DURATION_S else ''
        row += f"{dur_s:>10.1f}s{flag}"
        if dur_s < MIN_DURATION_S:
            flags.append((subj, task, dur_s))

    print(row)

if flags:
    print(f"\n[WARNING] {len(flags)} task(s) below {MIN_DURATION_S}s minimum:")
    for subj, task, dur in flags:
        print(f"  {subj} — Task {task} ({TASK_LABELS[task]}): {dur:.1f}s")
else:
    print(f"\nAll available recordings are >= {MIN_DURATION_S}s. OK.")

# ── 4. Check task_num in b1_preproc_fnirs.m ────────────────────────────────────
print("\n" + "="*70)
print(f"TASK_NUM CHECK in {os.path.basename(SCRIPT_PATH)}")
print("="*70)

if not os.path.isfile(SCRIPT_PATH):
    print(f"  [ERROR] Script not found: {SCRIPT_PATH}")
else:
    with open(SCRIPT_PATH, 'r') as f:
        lines = f.readlines()

    # Search for lines that look like a task number assignment
    patterns = [
        r"task[_\s]*num\s*=",           # task_num = ...
        r"task[_\s]*id\s*=",            # task_id = ...
        r"'00[1-6]'",                    # any hardcoded '00X'
        r'"00[1-6]"',                    # double-quoted version
        r'_00[1-6]',                     # folder suffix pattern
    ]
    combined = re.compile('|'.join(patterns), re.IGNORECASE)

    hits = [(i+1, line.rstrip()) for i, line in enumerate(lines) if combined.search(line)]

    if hits:
        print(f"  Found {len(hits)} relevant line(s):\n")
        for lineno, content in hits:
            print(f"  Line {lineno:>4}: {content}")
        print(f"\n  >> To run a different task, change the value on the line(s) above.")
    else:
        print("  [WARNING] No obvious task_num variable found.")
        print("  Manually inspect b1_preproc_fnirs.m to locate where the task folder is specified.")

print("\nPhase 0 audit complete.\n")