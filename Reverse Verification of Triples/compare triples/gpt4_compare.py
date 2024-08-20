import json
from Extract triples.extract_triples import request_api,load_progress,save_data
from concurrent.futures import ThreadPoolExecutor,as_completed

data_path = ''
save_path = ''

MODEL = "gpt-4-0125-preview"

def generate_prompt(sample_triples, unverified_triple):
    return f"""
context: {sample_triples}
unverified triple: {unverified_triple}
There is a set of triples in the context. Please determine whether the triples in the context support the unverified triple. If so, please answer "yes". If not, please answer "no".
    """

# def process_data(dataset):
#     processed_data = load_progress(save_path)
#     for key in ["PHD-Medium", "PHD-Low", "PHD-High"]:
#         current_processed_data=len(processed_data[key])
#         for index, entry in enumerate(dataset[key][current_processed_data:]):
#             new_triples=[]
#             for triple in entry["triples"]:
#                 classification_triples = {}
#                 single_triple_score = 0
#                 for sample_triple in entry["sample_triples"]:
#                     prompt = generate_prompt(sample_triple, triple["triple"])
#                     result = request_api(prompt,model=MODEL,temperature=0.0)
#                     if result.lower() == "yes":
#                         single_triple_score += 1
#                     #print(single_triple_score)
#                 classification_triples["classification"] = "support" if single_triple_score==10 else "NEI"
#                 #print(classification_triples["classification"])
#                 classification_triples["triple"] =triple["triple"]
#                 classification_triples["triple_label"] = triple["triple_label"]
#                 print(classification_triples)
#                 new_triples.append(classification_triples)
#             entry["triples"] = new_triples
#             print(entry["triples"])
#             processed_data[key].append(entry)
#             save_data(processed_data,save_path)

def classification_triple(sample_triples, unverified_triple):
    classification_triples = {}
    single_triple_score = 0
    for sample_triple in sample_triples:
        prompt = generate_prompt(sample_triple, unverified_triple["triple"])
        result = request_api(prompt, model=MODEL, temperature=0.0)
        print(result)
        if result.lower() == "yes":
            single_triple_score += 1
        print(single_triple_score)
    classification_triples["classification"] = "support" if single_triple_score == 10 else "NEI"
    print(classification_triples["classification"])
    classification_triples["triple"] = unverified_triple["triple"]
    return classification_triples

def process_data(dataset):
    processed_data = load_progress(save_path)
    current_processed_data=len(processed_data)
    for entry in dataset[current_processed_data:]:
        new_triples=[]
        with ThreadPoolExecutor(max_workers=40) as executor:
            classificationed_triple_futures={executor.submit(classification_triple,entry['sample_triples'],triple):triple for triple in entry['triples']}
            for future in as_completed(classificationed_triple_futures):
                new_triples.append(future.result())
        entry["triples"] = new_triples
        print(entry["triples"])
        processed_data.append(entry)
        save_data(processed_data,save_path)

if __name__ == "__main__":
    with open(data_path, "r",encoding='utf-8') as f:
        dataset = json.load(f)
    process_data(dataset)
