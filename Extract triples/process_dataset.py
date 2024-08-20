from extract_triples import *
from concurrent.futures import ThreadPoolExecutor,as_completed

data_path = ''

save_path = ''


with open(data_path, "r", encoding='utf-8') as f:
    content = f.read()
dataset = json.loads(content)

# def storage_triples(dataset):
#     new_dataset=dataset
#     keys_to_search = ["PHD-Medium", "PHD-Low", "PHD-High"]
#     for key in keys_to_search:
#         for entry in new_dataset.get(key, []):
#             triples=[]
#             if 'AI' in entry:
#                 response=init_triples_response(entry["entity"],entry["AI"])
#                 revise_response=update_triples_response(entry["entity"], entry["AI"], response)
#                 if entry["label"]=="factual":
#                     triples=process_triples_response(revise_response, "factual")
#                 else:
#                     triples=process_triples_response(revise_response, "")
#             entry["triples"]=triples
#     return new_dataset


def storage_triples(dataset):
    process_data=load_progress(save_path)
    current_data_length=len(process_data)
    new_dataset=[]
    with ThreadPoolExecutor(max_workers=20) as executor:
        entry_futures = {executor.submit(init_triples_response,entry["AI"]):entry for entry in dataset[current_data_length:]}
        for future in as_completed(entry_futures):
            response = future.result()
            entry = entry_futures[future]
            revise_response = update_triples_response(entry["gpt3_text"], response)
            print(revise_response)
            triples=process_triples_response(revise_response)
            new_entry={
                "entity":entry["entity"],
                "gpt3_text": entry["gpt3_text"],
                "label":entry["label"],
                "triples":triples
            }
            print(len(new_entry['triples']))
            print(new_entry)
            new_dataset.append(new_entry)
            save_data(new_dataset,save_path)
    return new_dataset

if __name__ == "__main__":
    init_dataset=dataset
    new_dataset=storage_triples(init_dataset)

