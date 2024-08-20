from openai import OpenAI
import openai
import time
import json


client = OpenAI()

MODEL="gpt-3.5-turbo"


GPT_TRIPLE_EXTRACTION_PROMPT = \
    """In the knowledge graph, knowledge triples are a basic data structure used to represent and store information, and each triple is an expression of a fact. Given a piece of text, please extract all knowledge triples contained in the text, and represent the triples in the form of ("head entity", "relationship", "tail entity").\
    Note that the extracted triples need to be as fine-grained as possible. It is necessary to ensure that the semantics of the triple are consistent with the information in the corresponding part of the text and there is not a pronoun in the triple.  All knowledge triples in the text need to be extracted\
    Here is an in-context example:
    <text>:Paris, the capital of France, is a city with a long history and full of romance. Not only is there the world-famous Eiffel Tower and Louvre Museum, but it also has a unique artistic atmosphere and rich cultural heritage.
    <response>:
    Triple: (Paris, is, the capital of France)
    Triple: (Paris, possession, long history)
    Triple: (Paris, full, romantic)
    Triple: (Paris, possession, Eiffel Tower)
    Triple: (Paris, possession, Louvre)
    Triple: (Paris, possessions, unique artistic atmosphere)
    Triple: (Paris, possessions, rich cultural heritage)
    <text>:"\"The Girl Who Loved Tom Gordon\" is a novel by Stephen King, published in 1999. The story follows a young girl named Trisha McFarland who becomes lost in the woods while on a family hike. As she struggles to survive, she turns to her favorite baseball player, Tom Gordon, for comfort and guidance. The novel explores themes of isolation, fear, and the power of imagination. It was a critical and commercial success, and has been adapted into a comic book and a stage play."
    <response>:
    Triple: ("The Girl Who Loved Tom Gordon", is, a novel by Stephen King)
    Triple: ("The Girl Who Loved Tom Gordon", published in, 1999)
    Triple: ("The Girl Who Loved Tom Gordon", follows, Trisha McFarland)
    Triple: ("The Girl Who Loved Tom Gordon" protagonist: Trisha McFarland, becomes, lost in the woods)
    Triple: ("The Girl Who Loved Tom Gordon" protagonist: Trisha McFarland, turns to, Tom Gordon for comfort and guidance)
    Triple: ("The Girl Who Loved Tom Gordon", explores themes of, "isolation, fear, and the power of imagination")
    Triple: ("The Girl Who Loved Tom Gordon", was, a critical and commercial success)
    Triple: ("The Girl Who Loved Tom Gordon", has been adapted into, a comic book)
    Triple: ("The Girl Who Loved Tom Gordon", has been adapted into, a stage play)
    <text>{init_text}
    """
GPT_TRIPLE_EXTRACTION_PROMPT_REVISE=\
    """Below are the knowledge Triples you extracted based on the text, but there are still some errors in it. For example,  the semantics of the triple are different from the semantics of the corresponding part in the original text or there is a pronoun in the triple. Please check and correct
    <initial prompt>{p}
    <triples>{t}\
    Please output all corrected triples directly ,including changed and unmodified ones. Don't output any other words. """

def load_progress(save_path):
    try:
        with open(save_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_data(data, save_path):
    with open(save_path, 'w') as file:
        json.dump(data, file, indent=4)

def request_api(prompt,model,temperature):
    flag = True
    while flag:
        try:
            message = [{'role': 'user', 'content': prompt}]
            response = client.chat.completions.create(model=model, messages=message, max_tokens=500, n=1,temperature=temperature)
            text_response = response.choices[0].message.content.strip()
            flag = False
            return text_response
        except openai.RateLimitError as e:
            print("speed limit exceeded")
            time.sleep(0.01)
        except Exception as e:
            print(e)
            time.sleep(0.005)


def init_triples_response(AI_text):
    prompts=GPT_TRIPLE_EXTRACTION_PROMPT.format(init_text=AI_text)
    init_triples =request_api(prompts,model=MODEL,temperature=0.0)
    return init_triples


def update_triples_response(AI_text, response):
    update_prompt = GPT_TRIPLE_EXTRACTION_PROMPT_REVISE.format(p=GPT_TRIPLE_EXTRACTION_PROMPT.format(init_text=AI_text), t=response)
    update_triples= request_api(update_prompt,model=MODEL,temperature=0.0)
    return update_triples

def process_triples_response(response):
    Triple_start = response.find("Triple")
    if Triple_start != -1:
        response = response[Triple_start:]
    else:
        print("No 'Triple' found in the string")
        return {}
    processed_data = []
    lines = response.split('\n')
    for ts in lines:
        if ts.strip():
            try:
                Triple = ts.split(': ')[1]
                # if label is not None:
                #     temp_data = {}
                #     temp_data['triple'] = Triple
                    # temp_data["triple_label"]=label
                temp_data={}
                temp_data['triple']=Triple
                processed_data.append(temp_data)
            except Exception as e:
                print(e)
                continue
    return processed_data