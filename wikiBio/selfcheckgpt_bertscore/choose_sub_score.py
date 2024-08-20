import json


data_path = ''

score_path = ''

base_data = ''

with open(data_path,'r',encoding='utf-8') as f:
    content = f.read()
dataset = json.loads(content)

with open(score_path,'r',encoding='utf-8') as f:
    content = f.read()
score = json.loads(content)

with open(base_data,'r',encoding='utf-8') as f:
    content = f.read()
base_data = json.loads(content)

sub_score_result = []
for entry in base_data:
    singe_sample = {}
    for index,data in enumerate(dataset):
        if entry['entity'] == data['entity']:
            singe_sample['entity'] = data['entity']
            singe_sample['label'] = data['label']
            singe_sample['text_score'] = score[index]
            sub_score_result.append(singe_sample)

with open('','w',encoding='utf-8') as f:
    json.dump(sub_score_result,f)


