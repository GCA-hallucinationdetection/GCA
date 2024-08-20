import json
import torch
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

data_path = ''
save_path = ''

with open(data_path, 'r') as f:
    content = f.read()
dataset = json.loads(content)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selfcheck_nli = SelfCheckNLI(device=device,nli_model='./potsawee/deberta-v3-large-mnli') # set device to 'cuda' if GPU is available


def get_selfcheckgpt_nli_score_on_phd(dataset):
    for entry in dataset:
        sent_scores_nli = selfcheck_nli.predict(
            sentences = entry["sentences"],                          # list of sentences
            sampled_passages = entry["samples_text"], # list of sampled passages
        )
        entry['sent_scores_nli'] = sent_scores_nli
    return dataset

if __name__ == '__main__':
    new_dataset = get_selfcheckgpt_nli_score_on_phd(dataset)
    with open(save_path, 'w') as f:
        json.dump(new_dataset,f)
