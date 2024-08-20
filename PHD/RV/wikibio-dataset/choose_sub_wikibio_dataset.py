import json


with open('all_wikibio_qg_record.json', 'r', encoding='utf-8') as file:
  all_wikibio_qg_data = json.load(file)

with open('./sub_wikibio_dataset.json','r',encoding='utf-8') as file:
    base_dataset = json.load(file)


def get_sub_dataset(base_dataset, dataset):
    sub_dataset = []
    for entry in base_dataset:
        for data in dataset:
            if data['entity'] == entry['entity']:
                sub_dataset.append(data)
    return sub_dataset

if __name__ == '__main__':
    scemantic_entropy_sub_dataset = get_sub_dataset(base_dataset, all_wikibio_qg_data)
    with open('./sub_wikibio_qg_record.json', 'w', encoding='utf-8') as f:
        json.dump(scemantic_entropy_sub_dataset,f)




