import os

input_path = "/mnt/afs/jingjinhao/project/influence_function/data/test_data.jsonl"
output_path = "/mnt/afs/jingjinhao/project/influence_function/data/test_data_new.jsonl"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

count = 0
with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        if count >= 25:
            break
        fout.write(line)
        count += 1
        
print("Done")