import json

data_path=''

with open(data_path,'r',encoding='utf-8') as f:
    content=f.read()
dataset=json.loads(content)

for entry in dataset:
    fact_triples=[]
    for triple in entry['triples']:
        if triple['classification'] == 'support':
            fact_triples.append(triple['triple'])
    entry['fact_triples']=fact_triples

with open(data_path,'w',encoding='utf-8') as f:
    json.dump(dataset,f)