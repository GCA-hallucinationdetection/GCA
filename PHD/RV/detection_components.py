import openai
import time
from openai import OpenAI

client = OpenAI()

def question_generation(entity,content):
  instructs = "I will give you some information about the entity. You should use all this information to generate a question, and the answer to your question is the entity. Do not include the entity in your question.\
  \nThere is an example.\nentity:World War II\ninformation: World War II, also known as the Second World War, was a global war that lasted from 1939 to 1945.\nquestion: which global war lasted from 1939 to 1945?\n\
  entity: {}\ninformation: {}\nquestion:"     
  prompt = instructs.format(entity,content)
  response = request_api(prompt)
  return response
      
def reverse_modeling(question):
  prompt = "You should answer the following question as short as possible. {}"
  prompt = prompt.format(question)
  response = request_api(prompt)
  return response

def request_api(Prompts):
    flag = True
    while flag:
        try:
            message = [{'role': 'user', 'content': Prompts}]
            response = client.chat.completions.create(model='gpt-3.5-turbo', messages=message, max_tokens=500, n=1,
                                                           temperature=0)
            text_response = response.choices[0].message.content.strip()
            flag = False
            return text_response
        except openai.RateLimitError as e:
            print("speed limit exceeded")
            time.sleep(0.01)
        except Exception as e:
            print('gpt4:', e)
            time.sleep(0.005)

def question_generation_pipeline(entity, content):
  question = question_generation(entity,content)    #construct query
  answer = reverse_modeling(question)
  record = {'entity':entity,'claim':content,'question':question,'answer':answer}  
  
  return record
  

  
# if __name__ == '__main__':
#   record = question_generation_pipeline()
  
