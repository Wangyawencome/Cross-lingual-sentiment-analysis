
from tqdm import tqdm
import json
from semantic_similarity_search import retrieved_content




prompt="You are a sentiment classification model. Determine whether given Question is positive, negative, neutral or ambivalent in terms of its sentiment:\n"




def one_high(content,labels_set,contents_set):
    if content in contents_set:
        index = contents_set.index(content)
        contents_set.pop(index)
        labels_set.pop(index)

    transformed_data=[]
    negative_index=[index for index, value in enumerate(labels_set) if value == "negative"]
    positive_index=[index for index, value in enumerate(labels_set) if value == "positive"]
    neutral_index=[index for index, value in enumerate(labels_set) if value == "neutral"]
    ambivalent_index=[index for index, value in enumerate(labels_set) if value == "ambivalent"]

    instruction="Question:"+contents_set[negative_index[0]]+"\n"+"Answer:"+"negative"+"\n"+"Question:"+contents_set[positive_index[0]]+"\n"+"Answer:"+"positive"+"\n"+"Question:"+contents_set[neutral_index[0]]+"\n"+"Answer:"+"neutral"+"\n"+"Question:"+contents_set[ambivalent_index[0]]+"\n"+"Answer:"+"ambivalent"

    return instruction

import os
transformed_data = []
file="test_all_text_zh_sampled.json"

with open(os.path.join("/root/new_work/multiemo/data/raw/single_embedding_data/",file), 'r') as f:
    original_data = json.load(f)
    print("original_data",len(original_data))
    for item in tqdm(original_data):
        try:
            labels_set,contents_set=retrieved_content(item["content"][:512].strip())
        except Exception as e:
            print('str(e):', str(e))

        instruction=one_high(item["content"],labels_set,contents_set)
        transformed_item = {
            "instruction": prompt+"\n"+instruction,
            "input": item["content"],
            "output": item["label"]
        }
        transformed_data.append(transformed_item)

import random
random.shuffle(transformed_data)

print("transformed_data",len(transformed_data))

with open(os.path.join('./sft/',file), 'w') as json_file:
    json.dump(transformed_data[:int(len(transformed_data))], json_file, indent=2)

# with open('./sft/all_text_ru_task_test.json', 'w') as json_file:
#     json.dump(transformed_data[int(len(transformed_data)*0.9):], json_file, indent=2)


print("The data has been successfully transformed and saved.")
