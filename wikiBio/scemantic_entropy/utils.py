"""Utility functions."""
import json
import logging
import hashlib
import openai
import time
from openai import OpenAI


import numpy as np
from eval_utils import (
    bootstrap, compatible_bootstrap, auroc, accuracy_at_quantile,
    area_under_thresholded_accuracy)


CLIENT = OpenAI()



def oai_predict(prompt):
    """Predict with GPT-4 model."""
    flag = True
    while flag:
        try:
            if isinstance(prompt, str):
                messages = [
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = prompt

            output = CLIENT.chat.completions.create(
                # model='gpt-4-0613',
                model='gpt-3.5-turbo',
                messages=messages,
                max_tokens=200,
            )
            response = output.choices[0].message.content
            flag = False
            return response
        except openai.RateLimitError as e:
            print("speed limit exceeded")
            time.sleep(0.01)
        except Exception as e:
            print(e)
            time.sleep(0.005)




def predict_w_log(prompt):
    """Predict and log inputs and outputs."""
    print(f'Input: {prompt}')
    response = oai_predict(prompt)
    print(f'Output: {response}')
    return response


def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy



def get_yes_no(response):
    binary_response = response.lower()[:10]
    if 'yes' in binary_response:
        uncertainty = 0
    elif 'no' in binary_response:
        uncertainty = 1
    else:
        uncertainty = 1
        logging.warning('MANUAL NO!')
    return uncertainty


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