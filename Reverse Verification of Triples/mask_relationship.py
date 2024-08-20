import json
from Extract triples.extract_triples import save_data,load_progress
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import openai
import time
import re

gpt3_client = OpenAI()

gpt4_client = OpenAI()

def gpt3_request_api(prompt,model,temperature):
    flag = True
    while flag:
        try:
            message = [{'role': 'user', 'content': prompt}]
            response = gpt3_client.chat.completions.create(model=model, messages=message, max_tokens=500, n=1,temperature=temperature)
            text_response = response.choices[0].message.content.strip()
            flag = False
            return text_response
        except openai.RateLimitError as e:
            print("speed limit exceeded")
            time.sleep(0.01)
        except Exception as e:
            print('gpt3:',e)
            time.sleep(0.005)

def gpt4_request_api(prompt,model,temperature):
    flag = True
    while flag:
        try:
            message = [{'role': 'user', 'content': prompt}]
            response = gpt4_client.chat.completions.create(model=model, messages=message, max_tokens=500, n=1,temperature=temperature)
            text_response = response.choices[0].message.content.strip()
            flag = False
            return text_response
        except openai.RateLimitError as e:
            print("speed limit exceeded")
            time.sleep(0.01)
        except Exception as e:
            print('gpt4:',e)
            time.sleep(0.005)


data_path=''

save_path=''

with open(data_path, "r", encoding='utf-8') as f:
    content = f.read()
dataset = json.loads(content)

MODEL_PREDICTION_RELATIONSHIP='gpt-3.5-turbo'

MODEL= 'gpt-4-0125-preview'

GPT_PREDICTION_RELATION_PROMPT = \
"""In the knowledge graph, knowledge triples are a basic data structure used to represent and store information. A knowledge triplet usually consists of three parts: head entity, relationship, and tail entity. This structure helps represent the relationships between entities in a structured way. \
    Now given a knowledge triplet, in which the relationship part is replaced by “mask”, you need to predict the relation that the "mask" represents based on the provided information and your knowledge. 
    <Input>Provide information:{fact_triples}.\nMasked triple:{masked_triple}
    Please output the complete triple without any additional words.
"""\

GPT_COMPARE_CONSISTENCY_PROMPT = \
"""Please judge whether the following two knowledge triples are semantically consistent and describe the same fact.If so, please answer "yes" directly, otherwise answer "no".\n{triple1}\n{triple2}
"""


def mask_rela(init_triple):
    single_quote_positions = [m.start() for m in re.finditer("'", init_triple)]
    if len(single_quote_positions) < 4:
        first_comma = init_triple.find(',')
        second_comma = init_triple.find(',', first_comma + 1)
        masked_triple = init_triple[:first_comma + 1] + ' mask' + init_triple[second_comma:]
    else:
        head_entity = init_triple[single_quote_positions[0]:single_quote_positions[1] + 1]
        tail_entity = init_triple[single_quote_positions[2]:single_quote_positions[3] + 1]
        masked_triple = f"{head_entity}, mask, {tail_entity}"
    return masked_triple


# def get_response(filter_fact_triples,masked_triple):
#     prediction_triples=[]
#     for i in range(0,10):
#         prediction_triple=gpt3_request_api(prompt=GPT_PREDICTION_RELATION_PROMPT.format(fact_triples=filter_fact_triples,masked_triple=masked_triple),model=MODEL_PREDICTION_RELATIONSHIP,temperature=0.0)
#         prediction_triples.append(prediction_triple)
#     return prediction_triples

def get_response(filter_fact_triples, masked_triple):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(gpt3_request_api, GPT_PREDICTION_RELATION_PROMPT.format(fact_triples=filter_fact_triples, masked_triple=masked_triple), MODEL_PREDICTION_RELATIONSHIP, 0.0) for _ in range(10)]
        return [future.result() for future in as_completed(futures)]

def filter(fact_triples,verified_triple):
    if verified_triple in fact_triples:
        return [triple for triple in fact_triples if triple != verified_triple]
    else:
        return fact_triples

def compare_init_triples(fact_triples, init_triple):
    result={}
    masked_triple=mask_rela(init_triple)
    print(masked_triple)
    filter_fact_triples = filter(fact_triples, init_triple)
    prediction_triples=get_response(filter_fact_triples,masked_triple)
    compare_result=[]
    compare_score=0
    for prediction_triple in prediction_triples:
        print('init_triple', init_triple)
        print('prediction_triple', prediction_triple)
        compare_answer= gpt4_request_api(prompt=GPT_COMPARE_CONSISTENCY_PROMPT.format(triple1=prediction_triple, triple2=init_triple),model=MODEL,temperature=0.0)
        compare_result.append(compare_answer)
        if compare_answer.lower() == 'yes':
            compare_score+=1
    if compare_score>=9:
        prediction_label="fact"
    else:
        prediction_label= "hallucination"
    result['filter_fact_triples']=filter_fact_triples
    result['prediction_triples']=prediction_triples
    result['compare_result']=compare_result
    result['compare_score']=compare_score
    result['prediction_label']=prediction_label
    print('result:',result)
    return result

def compare_enrich_data(fact_triples,enrich_triple):
    result={}
    masked_triple = mask_rela(enrich_triple)
    print(masked_triple)
    prediction_triples = get_response(fact_triples, masked_triple)
    compare_result = []
    compare_score = 0
    for prediction_triple in prediction_triples:
        print('enrich_triple',enrich_triple)
        print('prediction_triple',prediction_triple)
        compare_answer = gpt4_request_api(prompt=GPT_COMPARE_CONSISTENCY_PROMPT.format(triple1=prediction_triple, triple2=enrich_triple), model=MODEL,temperature=0.0)
        compare_result.append(compare_answer)
        if compare_answer.lower() == 'yes':
            compare_score += 1
    if compare_score >= 9:
        prediction_label = "fact"
    else:
        prediction_label = "hallucination"
    result['prediction_triples'] = prediction_triples
    result['compare_result']=compare_result
    result['compare_score']=compare_score
    result['prediction_label']=prediction_label
    print(result)
    return result


# def mask_relationship_pipeline(dataset):
#     processed_data = load_progress(save_path)
#     for key in dataset:
#         current_data_length = len(processed_data.get(key, []))
#         print(f"Current data length for {key}: {current_data_length}")
#
#         for entry in dataset[key][current_data_length:]:
#             new_entry = {
#                 'entity': entry['entity'],
#                 'AI': entry['AI'],
#                 'label': entry['label'],
#                 'triples': [],
#                 'enrich_triples': []
#             }
#             with ThreadPoolExecutor(max_workers=5) as executor:
#                 triple_futures = {executor.submit(compare_init_triples, entry['fact_triples'], triple['triple']): triple
#                                   for triple in entry['triples']}
#                 for future in triple_futures:
#                     mask_triple_result = future.result()
#                     triple = triple_futures[future]
#                     new_triple = {
#                         'mask_triple_result': mask_triple_result,
#                         'label': triple['triple_label'],
#                         'init_triple': triple['triple']
#                     }
#                     new_entry['triples'].append(new_triple)
#
#                 enrich_triple_futures = {
#                     executor.submit(compare_enrich_data, entry['fact_triples'], enrich_triple): enrich_triple for
#                     enrich_triple in entry['enrich_data']}
#                 for future in enrich_triple_futures:
#                     mask_enrich_triple_result = future.result()
#                     enrich_triple = enrich_triple_futures[future]
#                     new_enrich_triple = {
#                         'enrich_triple': enrich_triple,
#                         'mask_enrich_triple_result': mask_enrich_triple_result
#                     }
#                     new_entry['enrich_triples'].append(new_enrich_triple)
#
#             if key not in processed_data:
#                 processed_data[key] = []
#             processed_data[key].append(new_entry)
#             save_data(processed_data, save_path)
def mask_relationship_pipeline(dataset):
    processed_data = load_progress(save_path)
    current_data_length = len(processed_data)
    print(f"Current data length: {current_data_length}")

    for entry in dataset[current_data_length:]:
        new_entry = {
            'entity': entry['entity'],
            'AI': entry['gpt3_text'],
            'label': entry['label'],
            'triples': [],
        }
        with ThreadPoolExecutor(max_workers=5) as executor:
            triple_futures = {executor.submit(compare_init_triples, entry['fact_triples'], triple['triple']): triple
                              for triple in entry['triples']}
            for future in triple_futures:
                mask_triple_result = future.result()
                triple = triple_futures[future]
                new_triple = {
                    'mask_triple_result': mask_triple_result,
                    'init_triple': triple['triple']
                }
                new_entry['triples'].append(new_triple)
        processed_data.append(new_entry)
        save_data(processed_data, save_path)

if __name__ == '__main__':
    mask_relationship_pipeline(dataset)
