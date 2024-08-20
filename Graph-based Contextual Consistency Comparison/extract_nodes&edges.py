import json
import re

data_path = ''
save_path = ''

with open(data_path,'r',encoding='utf-8') as f:
    content = f.read()
dataset = json.loads(content)

def extract_sample_nodes_edges(triples):
    """
    """
    nodes=set()
    realtions=set()
    graph = []
    triple = triples.split('\n')
    for single_triple in triple:
        try:
            if "Triple:" in single_triple:
                single_triple=single_triple.replace("Triple:","")
            single_triple=re.sub(r'^[^\w]+|[^\w]+$', '', single_triple)
            parts = single_triple.split(',')
            tail_entity=(','.join(parts[2:]))
            nodes.add(parts[0])
            nodes.add(tail_entity)
            realtions.add(parts[1])
            graph_triple = (parts[0], parts[1], tail_entity)
            graph.append(graph_triple)
        except Exception as e:
            print(f'{e}，{single_triple}')
    return nodes, realtions, graph


def extract_nodes_edges(dataset):
    new_dataset = []
    for key in dataset:
        for entry in dataset[key]:
            new_entry={
                'entity':entry['entity'],
                'sample0':{
                }
            }
            nodes=set()
            relations=set()
            graph = []
            for triple in entry['triples']:
                try:
                    if "Triple:" in triple['triple']:
                        triple['triple']=triple['triple'].replace("Triple:","")
                    new_triple = re.sub(r'^[^\w]+|[^\w]+$', '', triple['triple'])
                    parts = new_triple.split(',')
                    tail_entity=(','.join(parts[2:]))
                    if '(' in parts[0]:
                        parts[0] = parts[0].split('(')[0]
                    nodes.add(parts[0])
                    nodes.add(tail_entity)
                    relations.add(parts[1])
                    graph_triple = (parts[0], parts[1], tail_entity)
                    graph.append(graph_triple)
                except Exception as e:
                    print(f'{e}，{triple}')
            new_entry['sample0']['nodes'] = list(nodes)
            new_entry['sample0']['relations'] = list(relations)
            new_entry['sample0']['graph'] = graph
            for index,sample_triples in enumerate(entry['sample_triples']):
                single_sample_nodes,single_sample_edges,single_sample_graph=extract_sample_nodes_edges(sample_triples)
                new_entry[f'sample{index+1}'] = {}
                new_entry[f'sample{index+1}']['nodes']=list(single_sample_nodes)
                new_entry[f'sample{index+1}']['relations']=list(single_sample_edges)
                new_entry[f'sample{index+1}']['graph'] = single_sample_graph
            new_entry['label'] = entry['label']
            new_dataset.append(new_entry)
    return new_dataset


if __name__ == '__main__':
    new_dataset = extract_nodes_edges(dataset)
    with open(save_path,'w',encoding='utf-8') as f:
        json.dump(new_dataset,f)




