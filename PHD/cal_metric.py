import json
from scipy.stats import tmean
import statistics
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score

semantic_entropy_result = ''
selfcheckgpt_nli_result = ''
selfcheckgpt_bertscore_result = ''

with open(semantic_entropy_result,'r',encoding='utf-8') as f:
    content = f.read()
semantic_entropy_dataset = json.loads(content)

with open(selfcheckgpt_nli_result,'r',encoding='utf-8') as f:
    content = f.read()
selfcheckgpt_nli_dataset = json.loads(content)

with open(selfcheckgpt_bertscore_result,'r',encoding='utf-8') as f:
    content = f.read()
selfcheckgpt_bertscore_dataset = json.loads(content)


label_mapping = {'factual':0, 'non-factual':1}

def cal_text_score(entry):
    text_score = sum(sent_score for sent_score in entry['triples_score'])/len(entry['triples_score'])
    return text_score


def predict(data,threshold):
    predict_label = []
    init_label = []
    for entry in data:
        text_score = cal_text_score(entry)
        if text_score > threshold:
            predict_label.append(1)
        else:
            predict_label.append(0)
        init_label.append(entry['label'])
    return predict_label, init_label

def mean_score(dataset):
    all_text_score = []
    for entry in dataset:
        all_text_score.append(cal_text_score(entry))
    mean_score = tmean(all_text_score)
    variance = statistics.variance(all_text_score)
    return mean_score,variance

def cal_metric(predict_label, init_label):
    f1 = f1_score(init_label, predict_label)
    precision =precision_score(init_label, predict_label)
    recall = recall_score(init_label, predict_label)
    accuracy = accuracy_score(init_label, predict_label)
    return f1, precision, recall, accuracy



if __name__ == '__main__':
    data_mapping = {"factual": 0, "non-factual":1}
    for i in [num/20 for num in range(1,20)]:
        threshold = i
        print(threshold)
        # #***************************semantic entropy**********************************
        semantic_mean_score,variance= mean_score(semantic_entropy_dataset)
        semantic_predict_label, semantic_init_label = predict(semantic_entropy_dataset, threshold)
        semantic_init_label = [data_mapping[entry] for entry in semantic_init_label]
        semantic_f1, semantic_precision, semantic_recall, semantic_accuracy = cal_metric(semantic_predict_label, semantic_init_label)
        print('f1:', semantic_f1, 'precision:', semantic_precision, 'recall:', semantic_recall, 'a:',semantic_accuracy)

        # ***************************selfcheckgpt_nli**********************************
        # nli_mean_score,variance = mean_score(selfcheckgpt_nli_dataset)
        # nli_predict_label, nli_init_label = predict(selfcheckgpt_nli_dataset, threshold)
        # nli_init_label = [data_mapping[entry] for entry in nli_init_label]
        # nli_f1, nli_precision, nli_recall, nli_accuracy = cal_metric(nli_predict_label,nli_init_label)
        # print('f1:', nli_f1, 'precision:', nli_precision, 'recall:', nli_recall, 'a:', nli_accuracy)

        # # # ***************************selfcheckgpt_bertscore**********************************
        # bertscore_mean_score = mean_score(selfcheckgpt_bertscore_dataset)
        # bertscore_predict_label, bertscore_init_label = predict(selfcheckgpt_bertscore_dataset, bertscore_mean_score)
        # bertscore_init_label = [data_mapping[entry] for entry in bertscore_init_label]
        # bertscore_f1, bertscore_precision, bertscore_recall, bertscore_accuracy = cal_metric(bertscore_predict_label,bertscore_init_label)
        # print('f1:', bertscore_f1, 'precision:', bertscore_precision, 'recall:', bertscore_precision, 'a:', bertscore_accuracy)
