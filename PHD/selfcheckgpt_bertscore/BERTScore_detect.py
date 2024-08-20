from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
import json
import torch
import time
from tqdm import tqdm
from collections import Counter

torch.manual_seed(28)

data_path = ''
save_score_path = ''


label_mapping = {'factual':0, 'non-factual':1}

def read_data(path):
    with open(path,"r") as f:
        data = json.load(f)
    return data

data = read_data(data_path)
selfcheck_bertscore = SelfCheckBERTScore()

all_score = []
for entry in tqdm(data):
    while True:
        try:
            sen_scores_bertscore = selfcheck_bertscore.predict(
                sentences= entry['sentences'],
                sampled_passages= entry['samples_text']
            )
            break
        except Exception as e:
            print("try again")
            time.sleep(3)
            
    passage_score = sum(sen_scores_bertscore)/len(sen_scores_bertscore)    
    print(float(passage_score))
    all_score.append(float(passage_score))
    
    
min_score = min(all_score)
print(min_score)

with open(save_score_path,"w") as w:
    json.dump(all_score,w)

    