import pandas as pd
import math
import numpy as np
import torch

SEED=10
np.random.seed(SEED)
torch.manual_seed(SEED)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed(SEED)  # 为GPU设置随机种子
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
torch.backends.cudnn.deterministic = True

min_max_file = 'data/forest_min_max_vals.csv'

def prepare_pattern_workload(path):
    pattern2training = {}
    pattern2truecard = {}
    minmax = pd.read_csv(min_max_file)
    minmax = minmax.set_index('name')
    min_card_log = 999999999999.0
    max_card_log = 0.0
    with open(path + '.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            tables = sorted([x.split(' ')[1] for x in line.split('#')[0].split(',')])
            local_cols = []
            vecs = []
            for col_name in minmax.index:
                if col_name.split('.')[0] in tables:
                    local_cols.append(col_name)
                    vecs.append(0.0)
                    vecs.append(1.0)
            conds = [x for x in line.split('#')[2].split(',')]
            for i in range(int(len(conds) / 3)):
                attr = conds[i * 3]
                op = conds[i * 3 + 1]
                value = conds[i * 3 + 2]
                idx = local_cols.index(attr)
                maximum = float(minmax.loc[attr]['max'])
                minimum = float(minmax.loc[attr]['min'])
                distinct_num = minmax.loc[attr]['num_unique_values']
                if op == '=':
                    offset = (maximum - minimum) / distinct_num / 2.0
                    upper = ((float(value) + offset) - minimum) / (maximum - minimum)
                    lower = (float(value) - offset - minimum) / (maximum - minimum)
                elif op == '<':
                    upper = (float(value) - minimum) / (maximum - minimum)
                    lower = 0.0
                elif op == '>':
                    upper = 1.0
                    lower = (float(value) - minimum) / (maximum - minimum)
                else:
                    raise Exception(op)
                if upper < vecs[idx * 2 + 1]:
                    vecs[idx * 2 + 1] = upper
                if lower > vecs[idx * 2]:
                    vecs[idx * 2] = lower
            key = '_'.join(tables)
            card = float(line.split('#')[-1])
            if key in pattern2training:
                pattern2training[key].append(vecs)
                pattern2truecard[key].append(card)
            else:
                pattern2training[key] = [vecs]
                pattern2truecard[key] = [card]
            if math.log(card) < min_card_log:
                min_card_log = math.log(card)
            if math.log(card) > max_card_log:
                max_card_log = math.log(card)

    return pattern2training, pattern2truecard, min_card_log, max_card_log


