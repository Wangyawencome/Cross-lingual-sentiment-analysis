# coding=utf-8
import requests
import json
from requests.auth import HTTPBasicAuth
import pandas as pd 
import csv
import time
from sklearn.metrics import f1_score

from transformers import AutoModel, AutoTokenizer, BertModel
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
import faiss
import json 
import torch 
class SentRetriever(object):
    def __init__(self,input_path,output_path):
        self.input_sents_file = input_path
        self.output_sents_file = output_path
        self.batch_size = 64

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_bert="/root/new_work/models/bert-base-multilingual-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_bert)
        self.model = AutoModel.from_pretrained(model_bert).to(self.device)
        self.model.eval()

        self.query_model = AutoModel.from_pretrained(model_bert).to(self.device)
        self.query_tokenizer = AutoTokenizer.from_pretrained(model_bert)
        self.query_model.eval()

        if os.path.isfile(self.output_sents_file):
            self.sent_features, self.sent_set, self.target_set = self.load_sent_features(self.output_sents_file)
            print(f"{len(self.sent_set)} sents loaded from {self.output_sents_file}")
        else:
            self.sent_set, self.target_set= self.load_sents()
            self.sent_features = self.build_sent_features()
            self.save_sent_features(self.sent_features, self.sent_set, self.target_set , self.output_sents_file)

    def load_sents(self):
        i_set = []
        # predict_set=[]
        target_set=[]
        test_file = open(self.input_sents_file, 'r', encoding='utf-8')
        test_data = json.load(test_file)
        for element in test_data:

            i_set.append(element['content'])
            # predict_set.append(element['Predict'])
            target_set.append(element['label'])

        print(f"Loading {len(i_set)} sents in total.")
        return i_set,target_set
    
    def build_sent_features(self):
        print(f"Build features for {len(self.sent_set)} sents...")
        batch_size, counter = self.batch_size, 0
        batch_text = []
        all_i_features = []
        # Prepare the inputs
        for i_n in tqdm(self.sent_set):
            counter += 1
            batch_text.append(i_n)
            if counter % batch_size == 0 or counter >= len(self.sent_set):
                with torch.no_grad():
                    i_input = self.tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
                    i_feature = self.model(**i_input, output_hidden_states=True, return_dict=True).pooler_output
                i_feature /= i_feature.norm(dim=-1, keepdim=True)
                all_i_features.append(i_feature.squeeze().to('cpu'))
                batch_text = []
        returned_text_features = torch.cat(all_i_features)
        return returned_text_features

    def save_sent_features(self, sent_feats, sent_names, sent_targets, path_to_save):
        assert len(sent_feats) == len(sent_names)
        print(f"Save {len(sent_names)} sent features at {path_to_save}...")
        torch.save({'sent_feats':sent_feats, 'sent_names':sent_names, 'sent_targets':sent_targets}, path_to_save)
        print(f"Done.")
    
    def load_sent_features(self, path_to_save):
        print(f"Load sent features from {path_to_save}...")
        checkpoint = torch.load(path_to_save)
        return checkpoint['sent_feats'], checkpoint['sent_names'] , checkpoint['sent_targets'] 


    def get_text_features(self, text):
        self.query_model.eval()
        with torch.no_grad():
            i_input = self.query_tokenizer(text,return_tensors="pt").to(self.device)
            text_features = self.query_model(**i_input, output_hidden_states=True, return_dict=True).pooler_output
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def setup_faiss(self):
        self.index = faiss.IndexFlatIP(768)   # build the index
        self.index.add(self.sent_features.numpy())    # add vectors to the index

    def faiss_retrieve(self, text, topk=5):
        text_f = self.get_text_features(text)
        D, I = self.index.search(text_f.cpu().numpy(), topk)     # actual search
        return D, I

input_path="/root/new_work/multiemo/data/raw/single_embedding_data/train_all_text_zh_sampled.json"
output_path="/root/new_work/multiemo/data/raw/single_embedding_data/train_all_text_zh_sampled.pt"


sentRetriever = SentRetriever(input_path, output_path)
sentRetriever.setup_faiss()

# test_content="była m po raz pierwszy u pana doktora , nie jestem z białegostoku , ale bez porównania jacy są ginekolodzy w mojej miejscowości a pan Mikołajewski to duża różnica , zna się na swoim zawodzie jest miły , uprzejmy umie rozmawiac z pacjentką po prostu zna sie na tym co robi , pozdrawiam go serdecznie",

# D,I = sentRetriever.faiss_retrieve(test_content, 10) 
# for idx_ in I[0]:
#     print(sentRetriever.sent_set[idx_])
#     print(sentRetriever.target_set[idx_])
#     aa=input()




def retrieved_content(sent):
    contents=[]
    labels=[]
    D,I = sentRetriever.faiss_retrieve(sent, 5000) 
    for idx_ in I[0]:
        contents.append(sentRetriever.sent_set[idx_])
        labels.append(sentRetriever.target_set[idx_])
    return labels,contents


# test_content="We were having a couple of friends over for some memorial weekend bbq so we wanted to check out the butcher shop for some steaks and charcuterie.  We frequent the restaurant often and always wonder what's going on with all the meat displayed in huge refrigerators near the entrance of the restaurant.\n\nThe butcher (Aaron) was EXTREMELY helpful.  He suggested a nice mix of charcuterie based on our adventurousness as well as provided some great tips on how to cook the steak.  With our skirt steak, he suggested that the best way to get an even temperature is to flip the steak every minute (vs flipping it once).  \n\nHe also suggested wines and cheeses that would go well with the charcuterie.  Even wrote it all down on butcher paper for us! He really knows his stuff!\n\nIs it a little on the pricey side? Sure, but not more than Whole Foods.  You are getting premium service and products so you def get what you pay for here."
# test_re=retrieved_content(test_content)
# print(test_re)




