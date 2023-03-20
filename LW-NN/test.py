import time
import csv
from data import *
from util import *
import torch
import numpy as np

SEED=10
np.random.seed(SEED)
torch.manual_seed(SEED)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed(SEED)  # 为GPU设置随机种子
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
torch.backends.cudnn.deterministic = True


def t_for_all_pattern(path, pattern2model):
    pattern2testing, pattern2truecard, min_card_log, max_card_log = prepare_pattern_workload(path)
    print('min_card_log: {}, max_card_log: {}'.format(min_card_log, max_card_log))
    cards = []
    true_cards = []
    start = time.time()
    for k, v in pattern2testing.items():
        model = pattern2model[k]
        cards += unnormalize(estimate(model, np.array(v)), min_card_log, max_card_log).tolist()
        true_cards += pattern2truecard[k]
    end = time.time()
    print("Prediction Time {}ms for each of {} queries".format((end - start) / len(cards) * 1000, len(cards)))
    qerror = print_qerror(np.array(cards), np.array(true_cards))
    print_mse(np.array(cards), np.array(true_cards))
    print_mape(np.array(cards), np.array(true_cards))
    print_pearson_correlation(np.array(cards), np.array(true_cards))
    print("qerror的长度: ", len(qerror))
    qerror = np.array(qerror)
    # print(qerror)
    np.savez("error/forest.npz", array_name=qerror)
    headers = ('estimate_card', 'true_card')

    with open('result/forest_10000.csv', 'w') as f:
        write = csv.writer(f)
        # for i in range(len(cards)):
        #     f.write(f'{cards[i]},{true_cards[i]}')
        #     f.write('\n')
        write.writerow(headers)
        for i in range(len(cards)):
            list = []
            list.append(cards[i])
            list.append(true_cards[i])
            write.writerow(list)