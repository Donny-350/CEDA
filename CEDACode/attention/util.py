import numpy as np
import pandas as pd
import csv
import torch

# Helper functions for data processing
SEED=20
np.random.seed(SEED)
torch.manual_seed(SEED)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed(SEED)  # 为GPU设置随机种子
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
torch.backends.cudnn.deterministic = True

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_all_column_names(predicates):
    column_names = set()
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                column_name = predicate[0]
                column_names.add(column_name)
    return column_names


def get_all_table_names(tables):
    table_names = set()
    for query in tables:
        for table in query:
            table_names.add(table)
    return table_names


def get_all_operators(predicates):
    operators = set()
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                operator = predicate[1]
                operators.add(operator)
    return operators


def get_all_joins(joins):
    join_set = set()
    for query in joins:
        for join in query:
            join_set.add(join)
    return join_set


def idx_to_onehot(idx, num_elements):
    onehot = np.zeros(num_elements, dtype=np.float32)
    onehot[idx] = 1.
    return onehot


def get_set_encoding(source_set, onehot=True):
    num_elements = len(source_set)
    source_list = list(source_set)
    # Sort list to avoid non-deterministic behavior
    source_list.sort()
    # Build map from s to i
    thing2idx = {s: i for i, s in enumerate(source_list)}
    # Build array (essentially a map from idx to s)
    idx2thing = [s for i, s in enumerate(source_list)]
    if onehot:
        thing2vec = {s: idx_to_onehot(i, num_elements) for i, s in enumerate(source_list)}
        return thing2vec, idx2thing
    return thing2idx, idx2thing


def get_min_max_vals(predicates, column_names):
    min_max_vals = {t: [float('inf'), float('-inf')] for t in column_names}
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                column_name = predicate[0]
                val = float(predicate[2])
                if val < min_max_vals[column_name][0]:
                    min_max_vals[column_name][0] = val
                if val > min_max_vals[column_name][1]:
                    min_max_vals[column_name][1] = val
    return min_max_vals


def normalize_data(val, column_name, column_min_max_vals):
    min_val = column_min_max_vals[column_name][0]
    max_val = column_min_max_vals[column_name][1]
    val = float(val)
    val_norm = 0.0
    if max_val > min_val:
        val_norm = (val - min_val) / (max_val - min_val)
    return np.array(val_norm, dtype=np.float32)


def normalize_labels(labels, min_val=None, max_val=None):
    labels = np.array([np.log(float(l)) for l in labels])
    if min_val is None:
        min_val = labels.min()
        print("min log(label): {}".format(min_val))
    if max_val is None:
        max_val = labels.max()
        print("max log(label): {}".format(max_val))
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val


def unnormalize_labels(labels_norm, min_val, max_val):
    labels_norm = np.array(labels_norm, dtype=np.float32)
    labels = (labels_norm * (max_val - min_val)) + min_val
    return np.array(np.round(np.exp(labels)), dtype=np.int64)


def encode_samples(tables, samples, table2vec):
    samples_enc = []
    for i, query in enumerate(tables):
        samples_enc.append(list())
        for j, table in enumerate(query):
            sample_vec = []
            # Append table one-hot vector
            sample_vec.append(table2vec[table])
            # Append bit vector
            sample_vec.append(samples[i][j])
            sample_vec = np.hstack(sample_vec)
            samples_enc[i].append(sample_vec)
    return samples_enc


def encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec):
    predicates_enc = []
    joins_enc = []
    for i, query in enumerate(predicates):
        predicates_enc.append(list())
        joins_enc.append(list())
        for predicate in query:
            if len(predicate) == 3:
                # Proper predicate
                column = predicate[0]
                operator = predicate[1]
                val = predicate[2]
                norm_val = normalize_data(val, column, column_min_max_vals)

                pred_vec = []
                pred_vec.append(column2vec[column])
                pred_vec.append(op2vec[operator])
                pred_vec.append(norm_val)
                pred_vec = np.hstack(pred_vec)
            else:
                pred_vec = np.zeros((len(column2vec) + len(op2vec) + 1))

            predicates_enc[i].append(pred_vec)

        for predicate in joins[i]:
            # Join instruction
            join_vec = join2vec[predicate]
            joins_enc[i].append(join_vec)
    return predicates_enc, joins_enc


def get_hist_file(hist_path, bin_number=50):
    hist_file = pd.read_csv(hist_path)
    for i in range(len(hist_file)):
        freq = hist_file['freq'][i]
        freq_np = np.frombuffer(bytes.fromhex(freq), dtype=np.float)
        hist_file['freq'][i] = freq_np

    table_column = []
    for i in range(len(hist_file)):
        table = hist_file['table'][i]
        col = hist_file['column'][i]
        table_alias = ''.join([tok[0] for tok in table.split('_')])
        if table == 'movie_info_idx': table_alias = 'mi_idx'
        combine = '.'.join([table_alias, col])
        table_column.append(combine)
    hist_file['table_column'] = table_column

    for rid in range(len(hist_file)):
        hist_file['bins'][rid] = \
            [int(i) for i in hist_file['bins'][rid][1:-1].split(' ') if len(i) > 0]

    if bin_number != 50:
        hist_file = re_bin(hist_file, bin_number)

    return hist_file


def re_bin(hist_file, target_number):
    for i in range(len(hist_file)):
        freq = hist_file['freq'][i]
        bins = freq2bin(freq, target_number)
        hist_file['bins'][i] = bins
    return hist_file


def freq2bin(freqs, target_number):
    freq = freqs.copy()
    maxi = len(freq) - 1

    step = 1. / target_number
    mini = 0
    while freq[mini + 1] == 0:
        mini += 1
    pointer = mini + 1
    cur_sum = 0
    res_pos = [mini]
    residue = 0
    while pointer < maxi + 1:
        cur_sum += freq[pointer]
        freq[pointer] = 0
        if cur_sum >= step:
            cur_sum -= step
            res_pos.append(pointer)
        else:
            pointer += 1

    if len(res_pos) == target_number: res_pos.append(maxi)

    return res_pos


def getPredicateHistEncode(predicate, hist_list, hist_map):
    if predicate != [''] and predicate[0] in hist_list:
        bins = hist_map[predicate[0]]
        if predicate[1] == '<':
            list = [0 for i in range(50)]
            for i in range(50):
                if (int(predicate[2]) >= int(bins[i])) and (int(predicate[2]) >= int(bins[i+1])):
                    list[i] = 1
                if int(bins[i]) < int(predicate[2]) and int(bins[i+1]) > int(predicate[2]):
                    list[i] = (int(predicate[2]) - int(bins[i]))/(int(bins[i+1]) - int(bins[i]))
            return list
        elif predicate[1] == '>':
            list = [1 for i in range(50)]
            for i in range(50):
                if (int(predicate[2]) >= int(bins[i])) and (int(predicate[2]) >= int(bins[i+1])):
                    list[i] = 0
                if (int(predicate[2]) > int(bins[i])) and (int(predicate[2]) < int(bins[i+1])):
                    list[i] = (int(bins[i+1]) - int(predicate[2])) / (int(bins[i+1]) - int(bins[i]))
            return list
        elif predicate[1] == '=':
            list = [0 for i in range(50)]
            for i in range(50):
                if (int(predicate[2]) > int(bins[i])) and (int(predicate[2]) < int(bins[i+1])):
                    list[i] = 1 / (int(bins[i+1]) - int(bins[i]))
            return list
    else:
        list = [0 for i in range(50)]
        return list