import argparse
import json
import os


def build_annotation_index(csv_path, anno_dir, out_path):
    """Walk annotation directories, read caption.json for each video.
    Save as single JSON: {video_id: {"caption": str, "camera_motion": str, "scene_type": str}}
    """
    # Read CSV to get video IDs that are actually available
    available_ids = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            available_ids.add(row["id"])

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
                cap = json.load(f)
            index[vid_id] = {
                "caption": cap.get("SceneDescription", ""),
                "camera_motion": cap.get("CameraMotion", ""),
                "scene_type": cap.get("CategoryTags", {}).get("sceneType", {}).get("first", ""),
            }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Annotation index: {len(index)} entries saved to {out_path}")
    return index


def load_annotation_index(path):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import csv

    parser = argparse.ArgumentParser(description="Build annotation index from caption.json files")
    parser.add_argument("--csv_path", type=str, required=True, help="SpatialVid metadata CSV")
    parser.add_argument("--anno_dir", type=str, required=True, help="Root dir of annotation subfolders")
    parser.add_argument("--out_path", type=str, required=True, help="Output JSON path")
    args = parser.parse_args()

    build_annotation_index(args.csv_path, args.anno_dir, args.out_path)
