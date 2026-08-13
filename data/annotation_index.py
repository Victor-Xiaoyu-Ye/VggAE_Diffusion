import argparse
import csv
import json
import os


def _derive_group(video_path):
    """'videos/group_0001/<id>.mp4' -> 'group_0001'."""
    for part in str(video_path).split("/"):
        if part.startswith("group"):
            return part
    return ""


def _caption_entry(cap):
    return {
        "caption": cap.get("SceneDescription", ""),
        "camera_motion": cap.get("CameraMotion", ""),
        "scene_type": cap.get("CategoryTags", {}).get("sceneType", {}).get("first", ""),
    }


def _read_csv_ids(csv_paths):
    """Ordered unique (video_id, group) pairs from one or more metadata CSVs."""
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    pairs, seen = [], set()
    for csv_path in csv_paths:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                vid = row["id"]
                if vid in seen:
                    continue
                seen.add(vid)
                pairs.append((vid, _derive_group(row.get("video path", ""))))
    return pairs


def _save_index(index, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Annotation index: {len(index)} entries saved to {out_path}")


def build_annotation_index(csv_path, anno_dir, out_path):
    """Walk a LOCAL annotations tree, read caption.json for each video.
    Save as single JSON: {video_id: {"caption": str, "camera_motion": str, "scene_type": str}}
    """
    available_ids = {vid for vid, _ in _read_csv_ids(csv_path)}
    index = {}
    for group_dir in sorted(os.listdir(anno_dir)):
        group_path = os.path.join(anno_dir, group_dir)
        if not os.path.isdir(group_path):
            continue
        for vid_id in os.listdir(group_path):
            if vid_id not in available_ids:
                continue
            caption_path = os.path.join(group_path, vid_id, "caption.json")
            if not os.path.exists(caption_path):
                continue
            with open(caption_path) as f:
                index[vid_id] = _caption_entry(json.load(f))
    _save_index(index, out_path)
    return index


def build_annotation_index_remote(csv_paths, anno_root, out_path, workers=16):
    """Read ``<anno_root>/<group>/<video_id>/caption.json`` keys directly from
    OBS for exactly the CSV video ids. Avoids copying the annotations tree:
    ``mox.file.copy_parallel`` of tens of thousands of small objects proved
    unreliable, while direct keyed reads only depend on the object keys.
    """
    from concurrent.futures import ThreadPoolExecutor

    from utils.moxing_io import read_bytes

    root = anno_root.rstrip("/")
    jobs = []
    skipped_group = 0
    for vid, group in _read_csv_ids(csv_paths):
        if not group:
            skipped_group += 1
            continue
        jobs.append((vid, f"{root}/{group}/{vid}/caption.json"))
    if skipped_group:
        print(f"[WARN] {skipped_group} rows without a group_* video path")

    def fetch(job):
        vid, url = job
        try:
            return vid, _caption_entry(json.loads(read_bytes(url).decode("utf-8")))
        except Exception:
            return vid, None

    index, missing = {}, 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for done, (vid, entry) in enumerate(pool.map(fetch, jobs), 1):
            if entry is None:
                missing += 1
            else:
                index[vid] = entry
            if done % 2000 == 0:
                print(f"annotation index: {done}/{len(jobs)} fetched "
                      f"({missing} missing)")
    print(f"annotation index: {len(index)} captions, {missing} missing "
          f"of {len(jobs)} ids under {root}")
    if not index:
        raise SystemExit(
            f"no caption.json readable under {root}; checked e.g. "
            f"{jobs[0][1] if jobs else '<no ids>'}")
    _save_index(index, out_path)
    return index


def load_annotation_index(path):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build annotation index from caption.json files")
    parser.add_argument("--csv_path", type=str, required=True, action="append",
                        help="SpatialVid metadata CSV (repeatable)")
    parser.add_argument("--anno_dir", type=str, required=True,
                        help="Annotations root: local dir, or obs://... for direct keyed reads")
    parser.add_argument("--out_path", type=str, required=True, help="Output JSON path")
    parser.add_argument("--workers", type=int, default=16,
                        help="parallel OBS reads in remote mode")
    args = parser.parse_args()

    if args.anno_dir.startswith(("obs://", "s3://")):
        build_annotation_index_remote(
            args.csv_path, args.anno_dir, args.out_path, args.workers)
    else:
        build_annotation_index(args.csv_path, args.anno_dir, args.out_path)
