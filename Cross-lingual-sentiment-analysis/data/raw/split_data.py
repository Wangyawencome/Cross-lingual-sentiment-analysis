
from tqdm import tqdm
import json
import os
import random

# file_path="/root/new_work/multiemo/data/raw/alldata"
# for file in tqdm(os.listdir(file_path)):
#     if "text" in str(file):
#         with open(os.path.join(file_path,file), 'r', encoding='utf-8') as f:
#             transformed_data = []
#             original_data = json.load(f)
#             print("original_data",len(original_data))
#             for item in tqdm(original_data):
#                 transformed_item = {
#                     "content": item["input"],
#                     "label": item["label"]
#                 }
#                 transformed_data.append(transformed_item)

    
#         random.shuffle(transformed_data)

#         print("transformed_data",len(transformed_data))
#         with open(os.path.join('./embedding_data',"train_"+file), 'w') as json_file:
#             json.dump(transformed_data[:int(len(transformed_data)*0.9)], json_file, indent=2)

#         with open(os.path.join('./embedding_data',"test_"+file), 'w') as json_file:
#             json.dump(transformed_data[int(len(transformed_data)*0.9):], json_file, indent=2)


#         print("The data has been successfully transformed and saved.")

file_path="/root/new_work/multiemo/data/raw/alldata/all_text_zh_sampled.json"

with open(file_path, 'r') as f:
    transformed_data = []
    original_data = json.load(f)
    print("original_data",len(original_data))
    for item in tqdm(original_data):
        transformed_item = {
            "content": item["input"],
            "label": item["label"]
        }
        transformed_data.append(transformed_item)


random.shuffle(transformed_data)

file="all_text_zh_sampled.json"
print("transformed_data",len(transformed_data))
with open(os.path.join('./single_embedding_data',"train_"+file), 'w') as json_file:
    json.dump(transformed_data[:int(len(transformed_data)*0.9)], json_file, indent=2,ensure_ascii=False)

with open(os.path.join('./single_embedding_data',"test_"+file), 'w') as json_file:
    json.dump(transformed_data[int(len(transformed_data)*0.9):], json_file, indent=2,ensure_ascii=False)


print("The data has been successfully transformed and saved.")



