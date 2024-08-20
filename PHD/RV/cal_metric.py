import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


phd_dataset = '../Extract triples/dataset/PHD_benchmark.json'  # PHD benchmark
wikibio_dataset = '../Extract triples/dataset/WikiBio_dataset/wikibio.json'

wikibio_qg_record_path = 'wikibio-dataset/all_wikibio_qg_record.json'


qg_record_path = 'our_dataset_qg_record.json'

label_mapping = {'factual': 0, 'non-factual': 1}


def read_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def get_label(data):
    all_label = []
    for key, value in data.items():
        for entry in value:
            all_label.append(entry['label'])

    return all_label



def get_qg_predict(data):  # RV-QG version
    all_predict = []
    for record in data:
        entity = record['entity'].split('(')[0].strip()
        answer = record['answer']
        if entity.lower() in answer.lower():
            label = 'factual'
        else:
            label = 'non-factual'
        all_predict.append(label)

    return all_predict


def get_wikibio_init_label(wiki_data,record):
    label = []
    for entry in record:
        for entry2 in wiki_data:
            if entry2['entity'] == entry['entity']:
                label.append(entry2['label'])
                print(entry['entity'],entry2['entity'])
                print(len(label))
    return label


'''use the following codes to calculate metrics on wikibio dataset'''

data = read_data(wikibio_dataset)
record = read_data(wikibio_qg_record_path)
predict_list = get_qg_predict(record)
label_list = get_wikibio_init_label(data,record)


map_label = [label_mapping[i] for i in label_list]
map_predict = [label_mapping[i] for i in predict_list]

f1 = f1_score(map_label, map_predict)
p = precision_score(map_label, map_predict)
r = recall_score(map_label, map_predict)
a=accuracy_score(map_label, map_predict)

print("all_metric:")
print(f1, p, r, a)

