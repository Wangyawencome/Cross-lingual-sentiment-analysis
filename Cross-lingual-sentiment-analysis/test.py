import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from sklearn.metrics import f1_score, accuracy_score,classification_report
from tqdm import tqdm
import pandas as pd

model_id = "/root/new_work/LLaMA-Factory/models/llama3_lora_sft-pl-pl-epoch5"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


file_path="/root/new_work/multiemo/data/sft/"
with pd.ExcelWriter('multiemo_text_result.xlsx', engine='openpyxl') as writer:
    f1_macro_list=[]
    f1_micro_list=[]
    f1_weighted_list=[]
    accuracy_list=[]
    classification_accuracy_list=[]
    name=[]
    for file in tqdm(os.listdir(file_path)):
        if "test" in str(file) :
            all_excemple=[]
            bad_excemple=[]
            for i in tqdm(range(0,1)):
                true_label=[]
                pre_label=[]
                num=0
                with open(os.path.join(file_path,file), 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    for line_ in tqdm(content):
                        
                        # try:
                            messages = [
                                {"role": "system", "content": "You are an all-purpose assistant."},
                                {"role": "user", "content": line_['instruction'].strip()+"\n"+line_['input'].strip()},       
                            ]

                            input_ids = tokenizer.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                return_tensors="pt"
                            ).to(model.device)

                            terminators = [
                                tokenizer.eos_token_id,
                                tokenizer.convert_tokens_to_ids("<|eot_id|>")
                            ]
                            outputs = model.generate(
                                input_ids,
                                max_new_tokens=4000,
                                eos_token_id=terminators,
                                do_sample=True,
                                temperature=0.01,
                                top_p=0.9,
                            )
                            response = outputs[0][input_ids.shape[-1]:]
                            response_=tokenizer.decode(response, skip_special_tokens=True)
                            all_excemple.append({"content":line_['input'].strip(),"true_label":line_['output'].strip(),"pre_label":response_})
                            response__=response_.replace("Positive","positive").replace("Negative","negative").replace("Neutral","neutral").replace("Ambivalent","ambivalent").split("Answer")[-1]


                            # aa=input()
                            if "positive" in response__.strip() :
                                pre_label.append("positive")
                                true_label.append(str(line_['output']).strip())
                            elif "negative" in response__.strip() :
                                pre_label.append("negative")
                                true_label.append(str(line_['output']).strip())
                            elif "neutral" in response__.strip() :
                                pre_label.append("neutral")
                                true_label.append(str(line_['output']).strip())    
                            elif "ambivalent" in response__.strip() :
                                pre_label.append("ambivalent")
                                true_label.append(str(line_['output']).strip())                   
                            else:
                                bad_excemple.append({"content":line_['input'].strip(),"true_label":str(line_['output']).strip(),"pre_label":response_})

                        # except Exception as ex:
                        #     print(ex)


                bad_excemple.append(f"***************第{i}轮*******************")
                f1_macro = "{:.4f}".format(f1_score(true_label, pre_label, average='macro'))
                f1_micro = "{:.4f}".format(f1_score(true_label, pre_label, average='micro'))
                f1_weighted = "{:.4f}".format(f1_score(true_label, pre_label, average='weighted'))
                accuracy = "{:.4f}".format(accuracy_score(true_label, pre_label))
                classification_accuracy=classification_report(true_label, pre_label, digits=4)


                name.append(str(file).split("_")[3])
                f1_macro_list.append(f1_macro)
                f1_micro_list.append(f1_micro)
                f1_weighted_list.append(f1_weighted)
                accuracy_list.append(accuracy)
                classification_accuracy_list.append(classification_accuracy)


            with open(os.path.join("/root/new_work/multiemo/multiemo_lora_predict/t0.01_alldata_new",file), 'w', encoding='utf-8') as file_:
                for item in all_excemple:
                    # 写入单个 JSON 对象，不加逗号，每个对象之间用换行符分隔
                    json_str = json.dumps(item, ensure_ascii=False,indent=4)
                    file_.write(json_str + "\n")  # 在每个 JSON 对象后加两个换行符以分隔

    data_ = {
                "name":name,
                "f1_macro":f1_macro_list,
                "f1_micro": f1_micro_list,
                "f1_weighted": f1_weighted_list,
                "accuracy": accuracy_list,
                "classification_accuracy_list": classification_accuracy_list,
            }

    df_ = pd.DataFrame(data_)
    df_.to_excel(writer, index=False)


