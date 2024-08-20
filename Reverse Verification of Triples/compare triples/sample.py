from Extract triples.extract_triples import *
from concurrent.futures import ThreadPoolExecutor,as_completed

MODEL="gpt-3.5-turbo"

data_path= ''

save_path=''

GPT_SAMPLE_RESPONSE_QUERY="Please write a brief Wikipedia for {Entity}"

with open(data_path,'r',encoding='utf-8') as f:
    content=f.read()
dataset=json.loads(content)

def add_sample_responses(entry):
    sample_responses=[]
    with ThreadPoolExecutor(max_workers=20) as executor:
        sample_futures = {executor.submit(request_api,GPT_SAMPLE_RESPONSE_QUERY.format(Entity=entry["entity"]),model=MODEL,temperature=1.0):i for i in range(10)}
        for future in as_completed(sample_futures):
            response=future.result()
            sample_responses.append(response)
            entry["sample_responses"]=sample_responses
            print(sample_responses)
    return sample_responses


# def add_sample_responses_triples(dataset):
#     process_sample_dataset=load_progress(save_path)
#     add_sample_triples_dataset=dataset
#     keys_to_search = ["PHD-Medium", "PHD-Low", "PHD-High"]
#     for key in keys_to_search:
#         current_process_index = len(process_sample_dataset[key])
#         print(current_process_index)
#         for index,entry in enumerate(add_sample_triples_dataset[key][current_process_index:]):
#             sample_responses=add_sample_responses(entry)
#             entry["sample_responses"]=sample_responses
#             sample_triples = []
#             for sample in sample_responses:
#                 response = init_triples_response(sample)
#                 revise_response = update_triples_response(sample, response)
#                 print("revise_response",revise_response)
#                 index = revise_response.find("Triple")
#                 if index != -1:
#                     sample_triples.append(revise_response[index:])
#                 else:
#                     print("revise_response do not have Triple")
#             entry["sample_triples"]=sample_triples
#             process_sample_dataset[key].append(entry)
#             save_data(process_sample_dataset,save_path)
#             print("sample_triples",sample_triples)

def add_sample_responses_triples(dataset):
    process_sample_dataset=load_progress(save_path)
    add_sample_triples_dataset=dataset
    print(process_sample_dataset)
    current_process_index = len(process_sample_dataset)
    print(current_process_index)
    for entry in add_sample_triples_dataset[current_process_index:]:
        sample_responses=add_sample_responses(entry)
        entry["sample_responses"]=sample_responses
        sample_triples = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            entry_futures = {executor.submit(init_triples_response, sample): sample for sample in
                            sample_responses}
            for future in as_completed(entry_futures):
                response = future.result()
                sample = entry_futures[future]
                revise_response = update_triples_response(sample, response)
                print("revise_response", revise_response)
                index = revise_response.find("Triple")
                if index != -1:
                    sample_triple=revise_response[index:]
                else:
                    print("revise_response do not have Triple")
                sample_triples.append(sample_triple)
        entry["sample_triples"]=sample_triples
        process_sample_dataset.append(entry)
        save_data(process_sample_dataset,save_path)
        print("sample_triples",sample_triples)


if __name__ == "__main__":
    add_sample_responses_triples(dataset)




