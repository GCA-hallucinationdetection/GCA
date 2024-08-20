import json
import statistics
from itertools import combinations
from scipy.stats import tmean
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

wikibio_dataset_path = ''
wikibio_dataset_score = ''

label_mapping = {'factual': 0, 'non-factual': 1}


def get_min_score(phd_benchmark_score):
    with open(phd_benchmark_score, "r") as f:
        data = json.load(f)
    min_score = min(data)
    return min_score


def read_data(path):
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_label(data):
    all_label = []
    for entry in data:
        all_label.append(entry['label'])
    return all_label


def predict(data, score):
    predict_list = []
    for value in data:
        if value > score:
            predict_list.append(1)
        else:
            predict_list.append(0)
    return predict_list


def compare(list1, list2):
    for comb in combinations(range(4), 3):
        if all(list1[i] > list2[i] for i in comb):
            return True

    return False


data = read_data(wikibio_dataset_path)
wikibio_score = read_data(wikibio_dataset_score)
max_score = max(wikibio_score)
label_list = get_label(data)
mean_score = tmean(wikibio_score)
print('mean_score: ', mean_score)
variance = statistics.variance(wikibio_score)

if __name__ == '__main__':
    for i in range(0,4):
        num_variance = i
        map_predict = predict(wikibio_score,  num_variance*variance )

        map_label = [label_mapping[i] for i in label_list]

        f1 = f1_score(map_label, map_predict)
        p = precision_score(map_label, map_predict)
        r = recall_score(map_label, map_predict)
        a = accuracy_score(map_label, map_predict)

        print(num_variance)
        print("f1:", f1, "p:", p, "r:", r, "a:", a)

