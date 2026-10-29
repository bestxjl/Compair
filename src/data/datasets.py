"""Data loading, windowing, and subject-wise splitting for 6 HAR datasets.

CRITICAL: Windowing is done AFTER subject split. No cross-subject leakage.
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.io import loadmat
from scipy import stats

from ..config import DATA_DIR


# ============================================================
# 1. UCI-HAR
# ============================================================
def load_ucihar():
    """Load UCI-HAR from raw Inertial Signals (9 channels).

    9 channels = body_acc(3) + gravity(3) + gyro(3), where gravity = total - body.
    Each .txt file has N rows × 128 columns (128 samples per window, 50Hz × 2.56s).
    """
    data_dir = os.path.join(DATA_DIR, 'UCI HAR Dataset')

    def _load_raw(subdir, set_name):
        sp = os.path.join(data_dir, subdir, 'Inertial Signals')
        # 3 body acceleration channels
        ba_x = np.loadtxt(os.path.join(sp, f'body_acc_x_{set_name}.txt')).astype(np.float32)
        ba_y = np.loadtxt(os.path.join(sp, f'body_acc_y_{set_name}.txt')).astype(np.float32)
        ba_z = np.loadtxt(os.path.join(sp, f'body_acc_z_{set_name}.txt')).astype(np.float32)
        # 3 total acceleration channels
        ta_x = np.loadtxt(os.path.join(sp, f'total_acc_x_{set_name}.txt')).astype(np.float32)
        ta_y = np.loadtxt(os.path.join(sp, f'total_acc_y_{set_name}.txt')).astype(np.float32)
        ta_z = np.loadtxt(os.path.join(sp, f'total_acc_z_{set_name}.txt')).astype(np.float32)
        # 3 gyroscope channels
        gy_x = np.loadtxt(os.path.join(sp, f'body_gyro_x_{set_name}.txt')).astype(np.float32)
        gy_y = np.loadtxt(os.path.join(sp, f'body_gyro_y_{set_name}.txt')).astype(np.float32)
        gy_z = np.loadtxt(os.path.join(sp, f'body_gyro_z_{set_name}.txt')).astype(np.float32)

        # Gravity = total - body
        gr_x = ta_x - ba_x
        gr_y = ta_y - ba_y
        gr_z = ta_z - ba_z

        # Stack: (N, 9, 128) where N is number of windows
        X = np.stack([ba_x, ba_y, ba_z, gr_x, gr_y, gr_z, gy_x, gy_y, gy_z], axis=1)
        y = np.loadtxt(os.path.join(data_dir, subdir, f'y_{set_name}.txt')).astype(int) - 1
        s = np.loadtxt(os.path.join(data_dir, subdir, f'subject_{set_name}.txt')).astype(int)
        return X, y, s

    Xtr, ytr, str_ = _load_raw('train', 'train')
    Xte, yte, ste = _load_raw('test', 'test')
    X = np.concatenate([Xtr, Xte])
    y = np.concatenate([ytr, yte])
    subjects = np.concatenate([str_, ste])

    data = {}
    for sid in np.unique(subjects):
        m = subjects == sid
        data[int(sid)] = {'x': X[m], 'y': y[m].astype(np.int64), 'subject': int(sid)}
    return data


# ============================================================
# 2. WISDM
# ============================================================
def load_wisdm():
    """Load WISDM from raw text file. Manual parsing for pandas 1.2.4 compat."""
    data_dir = os.path.join(DATA_DIR, 'WISDM_ar_v1.1')
    raw_file = os.path.join(data_dir, 'WISDM_ar_v1.1_raw.txt')

    act_map = {'Walking': 0, 'Jogging': 1, 'Upstairs': 2,
               'Downstairs': 3, 'Sitting': 4, 'Standing': 5}

    # Manual parsing (pandas 1.2.4 has issues with this file)
    rows = []
    with open(raw_file, 'r') as f:
        for line in f:
            line = line.strip().rstrip(';')
            parts = line.split(',')
            if len(parts) == 6:
                try:
                    user = int(parts[0])
                    act = parts[1]
                    if act not in act_map:
                        continue
                    label = act_map[act]
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                    rows.append((user, label, x, y, z))
                except (ValueError, IndexError):
                    continue

    if not rows:
        raise RuntimeError(f"Failed to parse WISDM data from {raw_file}")

    # Build signals by user
    user_data = defaultdict(lambda: {'sig': [], 'lab': []})
    for user, label, x, y, z in rows:
        user_data[user]['sig'].append([x, y, z])
        user_data[user]['lab'].append(label)

    # Windowing
    W, S_step = 100, 50
    data = {}
    for uid, d in user_data.items():
        sig = np.array(d['sig'], dtype=np.float32)
        lab = np.array(d['lab'], dtype=np.int64)
        xs, ys = [], []
        for i in range(0, len(sig) - W + 1, S_step):
            wlab = lab[i:i+W]
            maj = stats.mode(wlab, keepdims=False)[0]
            if isinstance(maj, np.ndarray):
                maj = maj[0]
            xs.append(sig[i:i+W].T)
            ys.append(int(maj))
        if len(xs) >= 10:
            data[uid] = {'x': np.array(xs, dtype=np.float32), 'y': np.array(ys, dtype=np.int64), 'subject': uid}
    return data


# ============================================================
# 3. PAMAP2
# ============================================================
def load_pamap2():
    """Load PAMAP2 from E:\\pre\\data\\raw\\PAMAP2_Dataset.
    Uses chest IMU (9 channels: acc16g_xyz, acc6g_xyz, gyro_xyz).
    Activities: 1=lying, 2=sitting, 3=standing, 4=walking, 5=running, 6=cycling.
    """
    data_dir = os.path.join(DATA_DIR, 'PAMAP2_Dataset', 'Protocol')
    CHEST = list(range(21, 30))  # chest IMU cols 21-29: acc16g(3) acc6g(3) gyro(3)

    data = {}
    for sid in range(1, 10):
        fname = f'subject10{sid}.dat' if sid < 9 else 'subject109.dat'
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue

        df = pd.read_csv(fpath, sep=r'\s+', header=None).values
        act = df[:, 1]
        sig = df[:, CHEST].astype(np.float32)
        valid = (act >= 1) & (act <= 6)
        if valid.sum() < 100:
            continue
        sig, lab = sig[valid], act[valid].astype(int) - 1
        # Handle NaN/inf in raw PAMAP2 data
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        sig, lab = sig[::2], lab[::2]  # 100->50Hz

        W, S_step = 128, 64
        xs, ys = [], []
        for i in range(0, len(sig) - W + 1, S_step):
            xs.append(sig[i:i+W].T.astype(np.float32))
            maj = stats.mode(lab[i:i+W], keepdims=False)[0]
            if isinstance(maj, np.ndarray):
                maj = maj[0]
            ys.append(int(maj))
        if len(xs) >= 20:
            data[sid] = {'x': np.array(xs, dtype=np.float32), 'y': np.array(ys, dtype=np.int64), 'subject': sid}
    return data


# ============================================================
# 4. MotionSense
# ============================================================
def load_motionsense():
    """Load MotionSense from E:\\pre\\data\\raw\\MotionSense Dataset.
    24 subjects, each with 6 CSVs (one per activity).
    12 channels: attitude(roll,pitch,yaw), gravity(x,y,z),
                 rotationRate(x,y,z), userAcceleration(x,y,z).
    Activities: dws=0, jog=1, ups=2, wlk=0, sit=4, std=5  (remapped).
    """
    ms_root = os.path.join(DATA_DIR, 'MotionSense Dataset',
                           'A_DeviceMotion_data', 'A_DeviceMotion_data')
    if not os.path.exists(ms_root):
        print(f"MotionSense not found at {ms_root}")
        return {}

    # Activity mapping -> standard 6-class order
    ACT_MAP = {'wlk': 0, 'jog': 1, 'ups': 2, 'dws': 3, 'sit': 4, 'std': 5}

    # Collect files by subject
    subj_data = defaultdict(lambda: {'x': [], 'y': []})

    for dname in sorted(os.listdir(ms_root)):
        dpath = os.path.join(ms_root, dname)
        if not os.path.isdir(dpath) or dname.startswith('_'):
            continue
        act_code = dname.split('_')[0]
        if act_code not in ACT_MAP:
            continue
        label = ACT_MAP[act_code]

        for csv_file in sorted(os.listdir(dpath)):
            if not csv_file.endswith('.csv'):
                continue
            # Extract subject ID from "sub_N.csv"
            sid_str = csv_file.replace('sub_', '').replace('.csv', '')
            try:
                sid = int(sid_str)
            except:
                continue

            df = pd.read_csv(os.path.join(dpath, csv_file))
            # Columns: attitude.roll, attitude.pitch, attitude.yaw,
            #          gravity.x, gravity.y, gravity.z,
            #          rotationRate.x, rotationRate.y, rotationRate.z,
            #          userAcceleration.x, userAcceleration.y, userAcceleration.z
            sig = df.iloc[:, 1:].values.astype(np.float32)  # 12 channels

            # Windowing
            W, S_step = 128, 64
            for i in range(0, len(sig) - W + 1, S_step):
                subj_data[sid]['x'].append(sig[i:i+W].T.astype(np.float32))
                subj_data[sid]['y'].append(label)

    result = {}
    for sid in subj_data:
        if len(subj_data[sid]['x']) >= 20:
            result[sid] = {
                'x': np.array(subj_data[sid]['x'], dtype=np.float32),
                'y': np.array(subj_data[sid]['y'], dtype=np.int64),
                'subject': sid,
            }
    return result


# ============================================================
# 5. USC-HAD
# ============================================================
def load_uschad():
    """Load USC-HAD from E:\\pre\\data\\raw\\USC-HAD.
    14 subjects, 12 activities, 5 trials each. .mat files.
    sensor_readings: (N, 6) = acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
    Select 6 core activities: 1=Walking, 4=Up, 5=Down, 8=Sit, 9=Stand, 10=Lie.
    """
    usc_root = os.path.join(DATA_DIR, 'USC-HAD')
    if not os.path.exists(usc_root):
        print(f"USC-HAD not found at {usc_root}")
        return {}

    # Map original activity numbers to 0-5
    # 1=walking->0, 4=upstairs->1, 5=downstairs->2, 8=sitting->3, 9=standing->4, 10=lying->5
    ACT_MAP = {1: 0, 4: 1, 5: 2, 8: 3, 9: 4, 10: 5}

    data = defaultdict(lambda: {'x': [], 'y': []})

    for subj_dir in sorted(os.listdir(usc_root)):
        spath = os.path.join(usc_root, subj_dir)
        if not os.path.isdir(spath) or not subj_dir.startswith('Subject'):
            continue
        try:
            sid = int(subj_dir.replace('Subject', ''))
        except:
            continue

        for mat_file in sorted(os.listdir(spath)):
            if not mat_file.endswith('.mat'):
                continue
            try:
                mat = loadmat(os.path.join(spath, mat_file))
                act_num = int(mat['activity_number'].flatten()[0])
                if act_num not in ACT_MAP:
                    continue
                label = ACT_MAP[act_num]
                sig = mat['sensor_readings'].astype(np.float32)  # (N, 6)
                # Downsample to 50Hz (from 100Hz)
                sig = sig[::2]

                W, S_step = 128, 64
                for i in range(0, len(sig) - W + 1, S_step):
                    data[sid]['x'].append(sig[i:i+W].T)
                    data[sid]['y'].append(label)
            except Exception:
                continue

    result = {}
    for sid in data:
        if len(data[sid]['x']) >= 20:
            result[sid] = {
                'x': np.array(data[sid]['x'], dtype=np.float32),
                'y': np.array(data[sid]['y'], dtype=np.int64),
                'subject': sid,
            }
    return result


# ============================================================
# 6. HHAR
# ============================================================
_HHAR_USERS = list('abcdefghi')
_HHAR_DEVICES = ['nexus4_1', 'nexus4_2', 's3_1', 's3_2',
                 's3mini_1', 's3mini_2', 'samsungold_1', 'samsungold_2']


def load_hhar():
    """Load HHAR phone accelerometer + gyroscope = 6 channels.
    9 users, 6 activities.
    6 classes: bike=0, sit=1, stairsdown=2, stairsup=3, stand=4, walk=5.

    THE ACCELEROMETER AND GYROSCOPE FILES ARE NOT ROW-ALIGNED. They are two
    independent recordings per device, with different sample counts (the S3
    gyroscope was logged at roughly twice the accelerometer rate, 5.24M vs
    2.56M rows), and Creation_Time is only meaningful within a single device.
    Stacking the two files by row index (the previous behaviour) therefore pairs
    an accelerometer sample with a gyroscope sample taken from an arbitrary
    other device and time: across the pooled file only 24% of index-paired
    samples came from the same device and 71% from the same subject.

    The phone gyroscope file also does not cover the Samsung Galaxy S+ at all
    (no samsungold rows exist), a further 1.89M accelerometer rows.

    We therefore merge per device on Creation_Time with a nearest-neighbour
    lookup, and drop devices the gyroscope file does not cover.
    """
    hhar_root = os.path.join(DATA_DIR, 'Heterogeneity Activity Recognition')
    acc_file = os.path.join(hhar_root, 'Phones_accelerometer.csv')
    gyr_file = os.path.join(hhar_root, 'Phones_gyroscope.csv')

    if not os.path.exists(acc_file):
        print(f"HHAR not found at {hhar_root}")
        return {}

    ACT_MAP = {'bike': 0, 'sit': 1, 'stairsdown': 2,
               'stairsup': 3, 'stand': 4, 'walk': 5}

    acc = _read_hhar_phone_stream(acc_file, ACT_MAP)
    gyr = _read_hhar_phone_stream(gyr_file, ACT_MAP)
    if acc is None or gyr is None:
        return {}

    merged = _merge_hhar_per_device(acc, gyr)
    del acc, gyr

    missing = merged['dropped_devices']
    if missing:
        print(f"  HHAR: {len(missing)} device(s) absent from the gyroscope file, "
              f"dropped: {', '.join(missing)}")

    # Window per (subject, phone) stream (each has an independent time base, so
    # a window must never straddle two of them), then pool the windows by subject.
    W, S_step = 128, 64
    per_user = defaultdict(lambda: {'x': [], 'y': []})
    for device, seg in merged['devices'].items():
        sig, lab = seg['sig'], seg['label']
        if len(sig) < W:
            continue
        xs, ys = [], []
        for i in range(0, len(sig) - W + 1, S_step):
            wlab = lab[i:i+W]
            xs.append(sig[i:i+W].T)
            maj = stats.mode(wlab, keepdims=False)[0]
            if isinstance(maj, np.ndarray):
                maj = maj[0]
            ys.append(int(maj))
        if len(xs) >= 20:
            uid = ord(seg['user']) - ord('a') + 1
            per_user[uid]['x'].extend(xs)
            per_user[uid]['y'].extend(ys)

    data = {}
    for uid, d in per_user.items():
        data[uid] = {
            'x': np.array(d['x'], dtype=np.float32),
            'y': np.array(d['y'], dtype=np.int64),
            'subject': uid,
        }

    print(f"  HHAR: {len(merged['devices'])} (subject, phone) stream(s) merged over "
          f"{len(data)} subject(s), {sum(len(v['x']) for v in data.values())} windows")
    return data


def _read_hhar_phone_stream(path, act_map, chunksize=1_000_000):
    """Read one HHAR phone file, keeping rows with a valid activity label.

    Returns a DataFrame sorted by (Device, Creation_Time) with compact dtypes,
    or None if nothing matched.
    """
    parts = []
    for chunk in pd.read_csv(
            path, chunksize=chunksize,
            usecols=['Creation_Time', 'x', 'y', 'z', 'User', 'Model', 'Device', 'gt']):
        chunk = chunk.dropna(subset=['gt'])
        chunk = chunk[chunk['gt'].isin(act_map)]
        if chunk.empty:
            continue
        chunk = chunk.assign(label=chunk['gt'].map(act_map).astype(np.int16))
        parts.append(chunk.drop(columns=['Model', 'gt']))
    if not parts:
        return None

    df = pd.concat(parts, ignore_index=True)
    df['User'] = pd.Categorical(df['User'].astype(str), categories=_HHAR_USERS)
    df['Device'] = pd.Categorical(df['Device'].astype(str), categories=_HHAR_DEVICES)

    unknown = (df['User'].cat.codes.to_numpy() < 0) | (df['Device'].cat.codes.to_numpy() < 0)
    if unknown.any():
        print(f"  HHAR: {int(unknown.sum())} row(s) with an unrecognised User/Device, dropped")
        df = df[~unknown]

    # A recording stream is one (subject, phone) pair. The eight phones in HHAR
    # were carried simultaneously by every subject in turn, so Device on its own
    # interleaves all nine subjects.
    df['_key'] = (df['User'].cat.codes.to_numpy(np.int64) * 16
                  + df['Device'].cat.codes.to_numpy(np.int64))
    df['Creation_Time'] = df['Creation_Time'].astype('int64')
    for col in ('x', 'y', 'z'):
        df[col] = df[col].astype('float32')
    return df.sort_values(['_key', 'Creation_Time'], kind='stable')


# An accelerometer sample whose nearest same-device gyroscope sample is further
# away than this is not trustworthy; drop it.
_HHAR_MAX_GAP_NS = 100_000_000  # 100 ms


def _contiguous_runs(sorted_keys):
    """Yield (key, start, stop) for each run of equal keys in a sorted array."""
    n = len(sorted_keys)
    if n == 0:
        return
    starts = np.flatnonzero(np.concatenate([[True], sorted_keys[1:] != sorted_keys[:-1]]))
    stops = np.concatenate([starts[1:], [n]])
    for s, e in zip(starts, stops):
        yield sorted_keys[s], int(s), int(e)


def _merge_hhar_per_device(acc, gyr, max_gap_ns=_HHAR_MAX_GAP_NS):
    """Pair every accelerometer sample with the nearest-in-time gyroscope sample
    from the *same (subject, phone)* stream.

    Returns {'devices': {key: {'sig', 'label', 'user', 'device', 't'}},
             'dropped_devices': [device, ...]}
    """
    acc_key = acc['_key'].to_numpy()
    acc_t = acc['Creation_Time'].to_numpy()
    acc_sig = acc[['x', 'y', 'z']].to_numpy()
    acc_lab = acc['label'].to_numpy()

    gyr_key = gyr['_key'].to_numpy()
    gyr_t = gyr['Creation_Time'].to_numpy()
    gyr_sig = gyr[['x', 'y', 'z']].to_numpy()

    gyr_by_key = {int(k): (gyr_t[i:j], gyr_sig[i:j])
                  for k, i, j in _contiguous_runs(gyr_key)}

    devices, dropped = {}, set()
    merged_samples = dropped_samples = 0

    for key, i, j in _contiguous_runs(acc_key):
        key = int(key)
        user_code, device_code = divmod(key, 16)
        user, device = _HHAR_USERS[user_code], _HHAR_DEVICES[device_code]

        g = gyr_by_key.get(key)
        at = acc_t[i:j]
        if g is None or len(g[0]) < 2:
            dropped.add(device)
            continue
        gt, gsig = g

        # Nearest-neighbour lookup of gt for every at (both already sorted).
        pos = np.clip(np.searchsorted(gt, at), 1, len(gt) - 1)
        left, right = gt[pos - 1], gt[pos]
        idx = np.where((at - left) <= (right - at), pos - 1, pos)
        keep = np.abs(at - gt[idx]) <= max_gap_ns
        if keep.sum() < 128:
            dropped.add(device)
            continue

        merged_samples += len(at)
        dropped_samples += int((~keep).sum())

        sig = np.column_stack([
            acc_sig[i:j][keep],
            gsig[idx[keep]],
        ]).astype(np.float32)

        devices[key] = {
            't': at[keep],
            'sig': sig,
            'label': acc_lab[i:j][keep],
            'user': user,
            'device': device,
        }

    print(f"  HHAR merge: {merged_samples:,} acc samples paired "
          f"({dropped_samples:,} dropped, {100.0*dropped_samples/max(merged_samples,1):.2f}%)")
    return {'devices': devices, 'dropped_devices': sorted(dropped)}


# ============================================================
# 7. RealWorld HAR
# ============================================================
def load_realworld():
    """Load RealWorld HAR dataset.

    Dataset structure (inside realworld2016_dataset.zip or extracted folder):
      proband1/data/
        acc_{activity}_csv.zip   -> contains acc_{activity}_{position}.csv (7 body positions)
        gyr_{activity}_csv.zip   -> contains Gyroscope_{activity}_{position}.csv
        ... (gps, lig, mag, mic zips, ignored)

    ACC CSV columns: id, attr_time, attr_x, attr_y, attr_z
    GYR CSV columns: id, attr_time, attr_x, attr_y, attr_z

    15 subjects, 50 Hz. Uses accelerometer + gyroscope = 6 channels.
    Selects 6 activities: walking, running, sitting, standing, climbingup, climbingdown.
    Merges all 7 body positions per subject -> LOSO CV (15 folds).
    """
    import zipfile
    import re

    _RW_MERGE_STATS.update({'segments': 0, 'segments_rejected': 0,
                            'acc_samples': 0, 'acc_samples_dropped': 0})

    # 1. Locate data source
    rw_dir = os.path.join(DATA_DIR, 'realworld2016_dataset')
    rw_zip = os.path.join(DATA_DIR, 'realworld2016_dataset.zip')
    # Legacy path
    rw_legacy = os.path.join(DATA_DIR, 'realworld2016')

    if os.path.isdir(rw_dir):
        rw_root = rw_dir
    elif os.path.isdir(rw_legacy):
        rw_root = rw_legacy
    elif os.path.exists(rw_zip):
        print("  RealWorld: extracting realworld2016_dataset.zip (one-time, ~3.7GB)...")
        with zipfile.ZipFile(rw_zip, 'r') as zf:
            zf.extractall(DATA_DIR)
        rw_root = rw_dir
        if not os.path.isdir(rw_root):
            print(f"  RealWorld: extraction failed, expected {rw_root}")
            return {}
        print("  RealWorld: extraction complete.")
    else:
        print(f"RealWorld not found. Tried:\n  {rw_dir}\n  {rw_zip}\n  {rw_legacy}")
        return {}

    # 2. Activity mapping (dataset name -> label 0-5)
    ACT_MAP = {
        'walking': 0, 'running': 1, 'sitting': 2,
        'standing': 3, 'climbingup': 4, 'climbingdown': 5,
        # Aliases
        'stairsup': 4, 'stairsdown': 5,
        'upstairs': 4, 'downstairs': 5,
    }

    # Body positions present in the dataset
    BODY_POSITIONS = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']

    data = defaultdict(lambda: {'x': [], 'y': []})

    # 3. Iterate over proband directories
    proband_dirs = sorted([
        d for d in os.listdir(rw_root)
        if os.path.isdir(os.path.join(rw_root, d)) and 'proband' in d.lower()
    ])

    if not proband_dirs:
        print(f"  RealWorld: no proband directories found in {rw_root}")
        return {}

    for proband_dir in proband_dirs:
        # Extract subject ID from "probandN"
        try:
            sid = int(re.search(r'(\d+)', proband_dir).group(1))
        except (ValueError, AttributeError):
            print(f"  SKIP: cannot parse subject ID from {proband_dir}")
            continue

        data_subdir = os.path.join(rw_root, proband_dir, 'data')
        if not os.path.isdir(data_subdir):
            print(f"  SKIP: {proband_dir}/data not found")
            continue

        # List all inner zip files OR extracted directories
        inner_files = os.listdir(data_subdir)
        # Support both .zip format (acc_walking_csv.zip) and extracted dirs (acc_walking_csv/)
        csv_sources = []
        for f in inner_files:
            fpath = os.path.join(data_subdir, f)
            if f.endswith('_csv.zip') and os.path.isfile(fpath):
                csv_sources.append(f)
            elif f.endswith('_csv') and os.path.isdir(fpath):
                csv_sources.append(f)

        # Group by activity: {activity: {'acc': path, 'gyr': path}}
        activity_zips = defaultdict(dict)
        for fname in csv_sources:
            # e.g. "acc_walking_csv.zip" or "acc_walking_csv" -> sensor="acc", activity="walking"
            base = fname[:-4] if fname.endswith('.zip') else fname  # strip .zip suffix
            sensor, rest = base.split('_', 1)
            activity = rest.rsplit('_csv', 1)[0]
            if sensor in ('acc', 'gyr'):
                activity_zips[activity][sensor] = os.path.join(data_subdir, fname)

        # 4. For each activity that has both acc+gyr
        for activity, sensor_files in activity_zips.items():
            if activity not in ACT_MAP:
                continue  # skip jumping, lying
            if 'acc' not in sensor_files or 'gyr' not in sensor_files:
                continue

            label = ACT_MAP[activity]

            try:
                # Read acc CSVs from inner zip or extracted directory
                acc_path = sensor_files['acc']
                gyr_path = sensor_files['gyr']
                if os.path.isfile(acc_path):
                    acc_data = _read_realworld_sensor_zip(acc_path, BODY_POSITIONS, 'acc')
                else:
                    acc_data = _read_realworld_sensor_dir(acc_path, BODY_POSITIONS, 'acc')
                if os.path.isfile(gyr_path):
                    gyr_data = _read_realworld_sensor_zip(gyr_path, BODY_POSITIONS, 'gyr')
                else:
                    gyr_data = _read_realworld_sensor_dir(gyr_path, BODY_POSITIONS, 'gyr')

                if acc_data is None or gyr_data is None:
                    continue

                # Merge acc+gyr by position and time -> (N, 6)
                sig_combined = _merge_acc_gyr(acc_data, gyr_data)
                if sig_combined is None or len(sig_combined) == 0:
                    continue

                data[sid]['x'].append(sig_combined)
                data[sid]['y'].append(np.full(len(sig_combined), label, dtype=np.int64))

            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
                print(f"  WARN: {proband_dir}/{activity}: {e}")
                continue

    # 5. Per-subject windowing (windows per activity, then concatenate
    #    to avoid windows spanning independently-recorded activity boundaries)
    result = {}
    for sid in sorted(data.keys()):
        if not data[sid]['x']:
            continue

        W, S_step = 128, 64
        all_xs, all_ys = [], []

        for seg, seg_label in zip(data[sid]['x'], data[sid]['y']):
            seg = seg.astype(np.float32)
            seg_len = len(seg)
            if seg_len < W:
                continue
            for i in range(0, seg_len - W + 1, S_step):
                wlab = seg_label[i:i+W]
                all_xs.append(seg[i:i+W].T)
                maj = stats.mode(wlab, keepdims=False)[0]
                if isinstance(maj, np.ndarray):
                    maj = maj[0]
                all_ys.append(int(maj))

        if len(all_xs) >= 20:
            result[sid] = {
                'x': np.array(all_xs, dtype=np.float32),
                'y': np.array(all_ys, dtype=np.int64),
                'subject': sid,
            }

    print(f"  RealWorld: {len(result)}/{len(proband_dirs)} subjects loaded "
          f"({sum(len(v['x']) for v in result.values())} windows)")
    print(f"  RealWorld merge: {_RW_MERGE_STATS['segments']} segments, "
          f"{_RW_MERGE_STATS['segments_rejected']} rejected, "
          f"{_RW_MERGE_STATS['acc_samples']} acc samples "
          f"({_RW_MERGE_STATS['acc_samples_dropped']} dropped)")
    return result


def _realworld_signal_from_df(df):
    """Extract (time, xyz) arrays from a RealWorld sensor CSV (already read)."""
    x_col = [c for c in df.columns if c in ('attr_x', 'x')]
    y_col = [c for c in df.columns if c in ('attr_y', 'y')]
    z_col = [c for c in df.columns if c in ('attr_z', 'z')]
    time_col = [c for c in df.columns if 'time' in c.lower()]
    if not (x_col and y_col and z_col):
        return None, None
    if time_col:
        t = df[time_col[0]].values
        sort_idx = np.argsort(t)
        t_sorted = t[sort_idx]
    else:
        sort_idx = slice(None)
        t_sorted = np.arange(len(df))
    x = df[x_col[0]].values.astype(np.float32)[sort_idx]
    y = df[y_col[0]].values.astype(np.float32)[sort_idx]
    z = df[z_col[0]].values.astype(np.float32)[sort_idx]
    sig = np.column_stack([x, y, z])
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    return t_sorted, sig


def _realworld_csv_position(filename):
    """Body position encoded in a RealWorld CSV filename.

    'acc_climbingup_chest.csv' -> 'chest'; 'acc_climbingup_2_chest.csv' -> 'chest';
    'Gyroscope_climbingup_2_chest.csv' -> 'chest'.
    """
    return filename.lower()[:-4].rsplit('_', 1)[-1]


def _realworld_matches_sensor(filename, sensor_type):
    """True if `filename` is the requested acc/gyr sensor CSV."""
    nl = filename.lower()
    if sensor_type == 'acc':
        return nl.startswith('acc_')
    return 'gyroscope' in nl


def _realworld_concat_frames(frames):
    """Concatenate (time, sig) frames chronologically by first timestamp."""
    frames.sort(key=lambda ts: ts[0][0] if len(ts[0]) else 0.0)
    return (np.concatenate([t for t, _ in frames]),
            np.concatenate([s for _, s in frames]))


def _read_realworld_sensor_zip(zip_path, body_positions, sensor_type):
    """Read one inner sensor zip (e.g. acc_walking_csv.zip).

    Handles flat (CSVs directly inside) and nested (session sub-zips such as
    acc_climbingup_1_csv.zip) layouts; multi-session CSVs for the same body
    position are concatenated in chronological order.
    """
    import zipfile
    import io

    result = {}
    with zipfile.ZipFile(zip_path, 'r') as zf:
        inner_names = zf.namelist()
        csv_names = [n for n in inner_names if n.lower().endswith('.csv')]
        nested_zips = [n for n in inner_names if n.lower().endswith('.zip')]

        per_position = {p: [] for p in body_positions}

        def _collect(name, raw):
            if not _realworld_matches_sensor(name, sensor_type):
                return
            pos = _realworld_csv_position(name)
            if pos in per_position:
                per_position[pos].append(raw)

        if csv_names:
            for name in csv_names:
                _collect(name, zf.read(name))
        elif nested_zips:
            for nz in nested_zips:
                with zipfile.ZipFile(io.BytesIO(zf.read(nz))) as nzf:
                    for inner in nzf.namelist():
                        if inner.lower().endswith('.csv'):
                            _collect(inner, nzf.read(inner))

        for position in body_positions:
            frames = []
            for raw in per_position[position]:
                try:
                    df = pd.read_csv(io.BytesIO(raw))
                except Exception:
                    continue
                t, sig = _realworld_signal_from_df(df)
                if t is None:
                    continue
                frames.append((t, sig))
            if not frames:
                continue
            t_all, sig_all = _realworld_concat_frames(frames)
            result[position] = {'time': t_all, 'sig': sig_all}

    return result if result else None


def _read_realworld_sensor_dir(dir_path, body_positions, sensor_type):
    """Read one extracted sensor directory (e.g. acc_walking_csv/).

    Handles flat (CSVs directly inside) and nested (session subdirectories such
    as acc_climbingup_1_csv/) layouts; multi-session CSVs for the same body
    position are concatenated in chronological order.
    """
    per_position = {p: [] for p in body_positions}
    for root, _dirs, files in os.walk(dir_path):
        for fname in files:
            if not fname.lower().endswith('.csv'):
                continue
            if not _realworld_matches_sensor(fname, sensor_type):
                continue
            pos = _realworld_csv_position(fname)
            if pos in per_position:
                per_position[pos].append(os.path.join(root, fname))

    result = {}
    for position in body_positions:
        frames = []
        for csv_path in per_position[position]:
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            t, sig = _realworld_signal_from_df(df)
            if t is None:
                continue
            frames.append((t, sig))
        if not frames:
            continue
        t_all, sig_all = _realworld_concat_frames(frames)
        result[position] = {'time': t_all, 'sig': sig_all}

    return result if result else None


_RW_MERGE_STATS = {'segments': 0, 'segments_rejected': 0,
                   'acc_samples': 0, 'acc_samples_dropped': 0}


def _merge_acc_gyr(acc_data, gyr_data):
    """Merge accelerometer and gyroscope data for matching body positions.

    acc_data: {position: {'time': array, 'sig': (N, 3)}}
    gyr_data: {position: {'time': array, 'sig': (N, 3)}}

    Returns: np.array (M, 6) where 6 = acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z

    The two sensors are written to separate files per (activity, position) and
    are *usually* synchronised at 50 Hz, but not always: for ~11% of
    (proband, activity, position) combinations the gyroscope file carries
    several concatenated sessions and is 1.2-4.0x longer than the accelerometer
    file (chest is the worst offender; proband7/walking/chest is 30,905 acc vs
    113,368 gyr rows). Pairing the two files by row index over a common prefix
    (the previous behaviour) therefore splices the middle of the recording onto
    the beginning of a different session, up to ~2/3 of the "gyroscope" channel
    is then the wrong part of the exercise.

    We instead resample the gyroscope onto the accelerometer timestamps by
    nearest-neighbour lookup, restricted to the overlapping time span, and drop
    accelerometer samples whose nearest gyroscope sample is more than two
    sampling intervals away. If the time columns are absent (both readers fall
    back to a sample index in that case) the tolerance degenerates to two
    samples, which is exactly the old index alignment.
    """
    all_segments = []

    for position in acc_data:
        if position not in gyr_data:
            continue

        acc_t = np.asarray(acc_data[position]['time'])
        acc_sig = acc_data[position]['sig']
        gyr_t = np.asarray(gyr_data[position]['time'])
        gyr_sig = gyr_data[position]['sig']

        if len(acc_t) < 10 or len(gyr_t) < 2 or len(acc_sig) != len(acc_t):
            continue

        # Sort defensively and drop non-finite timestamps.
        a_ord = np.argsort(acc_t, kind='stable')
        acc_t, acc_sig = acc_t[a_ord], acc_sig[a_ord]
        g_ord = np.argsort(gyr_t, kind='stable')
        gyr_t, gyr_sig = gyr_t[g_ord], gyr_sig[g_ord]

        # Tolerance = two sampling intervals, inferred from the accel stream
        # so it is unit-agnostic (ms for RealWorld, samples if time is absent).
        dt = np.median(np.diff(acc_t)) if len(acc_t) > 1 else 1.0
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
        max_gap = 2.0 * dt

        # Restrict to the overlapping time span.
        t0, t1 = max(acc_t[0], gyr_t[0]), min(acc_t[-1], gyr_t[-1])
        if not t1 > t0:
            _RW_MERGE_STATS['segments_rejected'] += 1
            continue
        in_span = (acc_t >= t0) & (acc_t <= t1)
        if in_span.sum() < 10:
            _RW_MERGE_STATS['segments_rejected'] += 1
            continue

        a_t, a_sig = acc_t[in_span], acc_sig[in_span]

        # Nearest-neighbour lookup of gyr_t for every accel timestamp.
        pos = np.searchsorted(gyr_t, a_t)
        pos = np.clip(pos, 1, len(gyr_t) - 1)
        left, right = gyr_t[pos - 1], gyr_t[pos]
        idx = np.where((a_t - left) <= (right - a_t), pos - 1, pos)
        gap = np.abs(a_t - gyr_t[idx])
        keep = gap <= max_gap
        if keep.sum() < 10:
            _RW_MERGE_STATS['segments_rejected'] += 1
            continue

        _RW_MERGE_STATS['segments'] += 1
        _RW_MERGE_STATS['acc_samples'] += len(a_t)
        _RW_MERGE_STATS['acc_samples_dropped'] += int((~keep).sum())

        combined = np.column_stack([
            a_sig[keep],
            gyr_sig[idx[keep]],
        ]).astype(np.float32)
        all_segments.append(combined)

    if not all_segments:
        return None

    return np.concatenate(all_segments, axis=0)


# ============================================================
# Registry
# ============================================================
LOADERS = {
    'ucihar': load_ucihar,
    'wisdm': load_wisdm,
    'pamap2': load_pamap2,
    'motionsense': load_motionsense,
    'uschad': load_uschad,
    'hhar': load_hhar,
    'realworld': load_realworld,
}


def load_dataset(dataset_key):
    """Load a dataset by key. Returns {subject_id: {x, y, subject}}."""
    if dataset_key not in LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Options: {list(LOADERS.keys())}")
    return LOADERS[dataset_key]()


# ============================================================
# Subject-Wise Cross-Validation
# ============================================================
def subject_wise_split(data, test_subjects, normalize=True):
    """Split data into train/test by subject IDs. Z-score from train only."""
    train_x, train_y = [], []
    test_x, test_y = [], []

    for sid, d in data.items():
        if sid in test_subjects:
            test_x.append(d['x']); test_y.append(d['y'])
        else:
            train_x.append(d['x']); train_y.append(d['y'])

    if not train_x or not test_x:
        raise ValueError(f"Empty split: train={len(train_x)}, test={len(test_x)} subjects")

    X_train = np.concatenate(train_x); y_train = np.concatenate(train_y)
    X_test = np.concatenate(test_x);   y_test = np.concatenate(test_y)

    if normalize:
        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    train_subjs = [s for s in data if s not in test_subjects]
    return X_train, y_train, X_test, y_test, train_subjs, list(test_subjects)


def get_cv_splits(data, dataset_key, n_folds=5, seed=42):
    """Generate subject-wise CV splits (k-fold or LOSO)."""
    from ..config import DATASETS
    cfg = DATASETS.get(dataset_key, {})
    cv_method = cfg.get('cv_method', 'kfold')
    subject_ids = sorted(data.keys())

    if cv_method == 'loso':
        for test_sid in subject_ids:
            yield [test_sid]
    else:
        np.random.seed(seed)
        shuffled = np.random.permutation(subject_ids)
        fold_size = max(1, len(shuffled) // n_folds)
        for fold in range(min(n_folds, len(shuffled))):
            start = fold * fold_size
            if fold == n_folds - 1:
                yield shuffled[start:].tolist()
            else:
                yield shuffled[start:start + fold_size].tolist()


# ============================================================
# PyTorch Dataset wrapper
# ============================================================
import torch
from torch.utils.data import Dataset

class HARWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NoisyHARWindowDataset(Dataset):
    """Wrapper that adds Gaussian noise at specified SNR to each sample.

    Noise is generated fresh on each __getitem__ call, so the same underlying
    sample can appear with different noise realizations across epochs.
    """
    def __init__(self, X, y, snr_db):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.snr_db = snr_db

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.snr_db is None:
            return x, y
        # Add Gaussian noise at specified SNR
        signal_power = (x ** 2).mean()
        if signal_power > 0:
            snr_linear = 10 ** (self.snr_db / 10.0)
            noise_power = signal_power / snr_linear
            noise = torch.randn_like(x) * torch.sqrt(noise_power)
            x = x + noise
        return x, y
