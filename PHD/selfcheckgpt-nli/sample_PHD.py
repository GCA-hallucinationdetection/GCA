import json
import time
import openai
from tqdm import tqdm

openai.api_key = ''
data_path = ''  #PHD benchmark
save_path = ''      #save sampled result


def samples(entity):
    sample_list = []
    sys_prompt = "Answer the following question only if you know the answer or can make a well-informed guess; otherwise tell me you don't know it"
    instructs = "Please write a brief Wikipedia for {} under 100 words."      #prompt
    prompt = instructs.format(entity)
    for i in range(3):          #control the number of sample
        sample = request_api(sys_prompt,prompt)
        sample_list.append(sample)
    return sample_list

def request_api(sys_prompt,prompt):          
  flag = True
  while flag:
      try:
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=1.0           
                )
        flag = False
      except Exception as e:
          print("try again!")
          print(e)
          time.sleep(5)
  text_response = response["choices"][0]["message"]["content"]
  return text_response

def read_data(path):
    with open(path,"r") as f:
        data = json.load(f)
    return data

def add_samples_to_data(data):
    new_data = []
    for key,value in data.items():
        for entry in tqdm(value):
            sample_list = samples(entry['entity'])
            entry['samples_text'] = sample_list
            new_data.append(entry)
    return new_data
    
def save_new_data(path,new_data):
    with open(path,"w") as w:
        json.dump(new_data,w)

data = read_data(data_path)
new_data = add_samples_to_data(data)
save_new_data(save_path,new_data)