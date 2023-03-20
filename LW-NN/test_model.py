import time
import csv
import torch
from model import *
from data import *
from util import *

path = 'model/1.pth'

# 加载模型
model = MLP(20, 128)
model.load_state_dict(torch.load(path))
print(model)

test_path = 'workloads/forest/foresttest.sql'

pattern2testing, pattern2truecard, min_card_log, max_card_log = prepare_pattern_workload(test_path)
cards = []
true_cards = []
start = time.time()

for k, v in pattern2testing.items():
    cards += unnormalize(estimate(model, np.array(v)), min_card_log, max_card_log).tolist()
    true_cards += pattern2truecard[k]

end = time.time()
print("Prediction Time {}ms for each of {} queries".format((end - start) / len(cards) * 1000, len(cards)))
print_qerror(np.array(cards), np.array(true_cards))
print_mse(np.array(cards), np.array(true_cards))
print_mape(np.array(cards), np.array(true_cards))
print_pearson_correlation(np.array(cards), np.array(true_cards))

headers = ('estimate_card', 'true_card')

with open('result/forest_test_model.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(headers)
    for i in range(len(cards)):
        list = []
        list.append(cards[i])
        list.append(true_cards[i])
        write.writerow(list)
