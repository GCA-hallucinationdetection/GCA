import json

wikibio_dataset = 'wikibio.json'

with open("./dataset.json", "r") as f:
    content = f.read()
dataset = json.loads(content)



def extract_entity(sentence):
    piece = sentence.split('(')
    # name = piece[0].split(' ')
    # if len(name) > 4:
    #     return None
    # else:
    return piece[0]

def get_passage_label(annotation):
    score_map = {'major_inaccurate':1,'minor_inaccurate':1,'accurate':0}
    score_list = [int(score_map[label]) for label in annotation]
    sum_score = sum(score_list)
    hal_ratio = sum_score/len(score_list)
    return sum_score,hal_ratio
  
def reconstruct_dataset(dataset):
    new_dataset = []
    print(len(dataset))
    for entry in dataset:
        entity = extract_entity(entry['gpt3_sentences'][0])
        # if entity is not None:
        entry['entity'] = entity
        score,ratio = get_passage_label(entry['annotation'])
        if score == 0:
            entry['label'] = 'factual'
            entry['ratio'] = 0.0
        else:
            entry['label'] = 'non-factual'
            entry['ratio'] = round(ratio,1)
        new_dataset.append(entry)
    print(len(new_dataset))
    return new_dataset


new_dataset = reconstruct_dataset(dataset)

with open(wikibio_dataset, "w") as w:
    json.dump(new_dataset,w)
    
 
