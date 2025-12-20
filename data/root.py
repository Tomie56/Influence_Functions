import json
from pathlib import Path

input_path = "/mnt/afs/jingjinhao/project/influence_function/data/train_data.jsonl"
output_path = "/mnt/afs/jingjinhao/project/influence_function/data/train_data_with_root.jsonl"

IMAGE_ROOT = "/mnt/afs/jingjinhao/project/latex_render" 

image_root = Path(IMAGE_ROOT)

cnt, changed = 0, 0
with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)

        img = obj.get("image")
        if isinstance(img, str) and img:
            img_path = Path(img)
            if not img_path.is_absolute():
                new_img = str(image_root / img_path)
                if new_img != img:
                    obj["image"] = new_img
                    changed += 1

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        cnt += 1
