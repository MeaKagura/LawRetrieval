import json
import numpy as np


def evaluation_retrieval(predict_path, label_path, mrr_k, recall_k):
    with open(predict_path, 'r') as pf:
        with open(label_path, 'r') as lf:
            predicts, labels = json.load(pf), json.load(lf)
            mrr, recall = [], []
            for i, results in enumerate(predicts):
                assert len(results) >= mrr_k and len(results) >= recall_k
                true_law, true_num = labels[i]['true_law'], labels[i]['true_num']
                mrr.append(0)
                count = 0
                for j, result in enumerate(results):
                    predict_law, predict_num, _ = result.values()
                    if true_law == predict_law and true_num == predict_num:
                        if mrr[-1] == 0 and j < mrr_k:
                            mrr[-1] = (1 / (j + 1))
                        if j < recall_k:
                            count += 1
                recall.append(count / 1)
            return np.mean(mrr), np.mean(recall)


retrieval_predict_path = './predicts.json'
label_path = './test.json'
mrr_10, recall_5 = evaluation_retrieval(retrieval_predict_path, label_path, 10, 5)
print("mrr@10:", mrr_10)
print("recall@5:", recall_5)


