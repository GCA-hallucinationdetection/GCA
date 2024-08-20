from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
import json
import torch
import time
from tqdm import tqdm

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
    single_sample ={
        'entity': entry['entity'],
        'label': entry['label'],
    }
    while True:
        try:
            sen_scores_bertscore = selfcheck_bertscore.predict(
                sentences= entry['gpt3_sentences'],
                sampled_passages= entry['gpt3_text_samples']
            )
            break
        except Exception as e:
            print("try again")
            time.sleep(3)
            
    passage_score = sum(sen_scores_bertscore)/len(sen_scores_bertscore)
    single_sample['selfcheck_bertscore'] = passage_score
    print(float(passage_score))
    all_score.append(single_sample)
    


with open(save_score_path,"w") as w:
    json.dump(all_score,w)

    