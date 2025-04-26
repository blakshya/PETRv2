import json
import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def ensure_backups(root: Path, filenames: list, backup_suffix: str = '.backup.json'):
    """
    Ensure that each JSON metadata file has a backup. If `<stem>.backup.json` does not exist,
    copy `<stem>.json` to `<stem>.backup.json`.
    """
    root = Path(root)
    for fname in filenames:
        orig = root / fname
        backup = root / f"{orig.stem}{backup_suffix}"
        if not backup.exists():
            print(f"Backing up {orig.name} -> {backup.name}")
            shutil.copy2(orig, backup)
        else:
            print(f"Backup already exists: {backup.name}")


def restore_backups(root: Path, filenames: list, backup_suffix: str = '.backup.json'):
    """
    Restore original JSON metadata files from their backups.
    """
    root = Path(root)
    for fname in filenames:
        orig = root / fname
        backup = root / f"{orig.stem}{backup_suffix}"
        if backup.exists():
            print(f"Restoring {orig.name} from {backup.name}")
            shutil.copy2(backup, orig)
        else:
            print(f"Backup not found for {orig.name}, skipping.")


def load_json(path: Path):
    """
    Load a JSON file (list of records or dict) into Python.
    """
    with open(path, 'r') as f:
        return json.load(f)


def write_json(path: Path, data):
    """
    Write Python list/dict back to JSON with indentation.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def filter_records(records: list, keep_fn) -> list:
    """
    Return the full record dicts for which keep_fn(record) is True,
    preserving all metadata fields.
    """
    return [r for r in records if keep_fn(r)]


def load_from_backups(data_root: Path, filenames: list, backup_suffix: str = '.backup.json') -> dict:
    """
    Concurrently load metadata JSONs from their backup files into a dict.
    """
    data = {}
    with ThreadPoolExecutor(max_workers=min(len(filenames), 8)) as executor:
        futures = {}
        for fname in filenames:
            backup_path = data_root / f"{Path(fname).stem}{backup_suffix}"
            futures[executor.submit(load_json, backup_path)] = fname
        for future in as_completed(futures):
            fname = futures[future]
            data[fname] = future.result()
    return data


def filter_metadata(data_root: Path, subset_scenes_file: str):
    """
    Filter nuScenes metadata in-place to include only a given set of scene tokens.
    """
    backup_suffix = '.backup.json'
    metadata_files = [
        'scene.json', 'sample.json', 'sample_data.json',
        'sample_annotation.json', 'calibrated_sensor.json',
        'ego_pose.json', 'log.json', 'map.json'
    ]
    # Ensure backups
    ensure_backups(data_root, metadata_files, backup_suffix)
    # Load metadata
    data = load_from_backups(data_root, metadata_files, backup_suffix)
    # Scene tokens
    keep_scenes = set(load_json(Path(subset_scenes_file))['scene_tokens'])
    # Build keep sets
    
    # Samples
    keep_samples        = set(r['token'] for r in data['sample.json']
                            if r.get('scene_token') in keep_scenes)
    # Sample data
    keep_sample_data    = set(r['token'] for r in data['sample_data.json']
                            if r.get('sample_token') in keep_samples)
    # Annotations
    keep_annots         = set(r['token'] for r in data['sample_annotation.json']
                            if r.get('sample_token') in keep_samples)
    # Calibration & poses
    sd_map              = {r['token']: r for r in data['sample_data.json']}
    keep_calib          = set(sd_map[t]['calibrated_sensor_token'] for t in keep_sample_data)
    keep_pose           = set(sd_map[t]['ego_pose_token'] for t in keep_sample_data)
    # Logs
    keep_logs           = set(r['log_token'] for r in data['scene.json'] if r['token'] in keep_scenes)
    # Maps: keep map tokens whose log_tokens intersect keep_logs
    keep_map_tokens     = set(r['token'] for r in data['map.json']
                            if set(r.get('log_tokens', [])).intersection(keep_logs))
    # Filter and write
    filters = {
        'scene.json':             lambda r: r['token'] in keep_scenes,
        'sample.json':            lambda r: r['token'] in keep_samples,
        'sample_data.json':       lambda r: r['token'] in keep_sample_data,
        'sample_annotation.json': lambda r: r['token'] in keep_annots,
        'calibrated_sensor.json': lambda r: r['token'] in keep_calib,
        'ego_pose.json':          lambda r: r['token'] in keep_pose,
        'log.json':               lambda r: r['token'] in keep_logs,
        'map.json':               lambda r: r['token'] in keep_map_tokens,
    }
    for fname, keep_fn in filters.items():
        orig = data[fname]
        filtered = filter_records(orig, keep_fn)
        write_json(data_root / fname, filtered)
        print(f"Filtered {fname}: {len(filtered)}/{len(orig)} records kept ({len(filtered)/len(orig)*100:.2f}%)")
        
    print("Filtering complete.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="nuScenes subset filter/recovery tool.")
    parser.add_argument('data_root', nargs='?', default='/scratch1/lbhatnag/nuscenes_extracted/v1.0-trainval',
                        help="Path to v1.0-trainval folder.")
    parser.add_argument('--subset', dest='subset_scenes', default='/project2/ywang234_1595/petr_v2/nuscenes_subset/subset_scenes.json',
                        help="JSON file with {'scene_tokens': [...]}.")
    parser.add_argument('--restore', action='store_true',
                        help="Restore metadata files from backups instead of filtering.")
    args = parser.parse_args()
    root = Path(args.data_root)
    metadata_files = [
        'scene.json', 'sample.json', 'sample_data.json',
        'sample_annotation.json', 'calibrated_sensor.json',
        'ego_pose.json', 'log.json', 'map.json'
    ]
    if args.restore:
        restore_backups(root, metadata_files)
    else:
        filter_metadata(root, args.subset_scenes)
