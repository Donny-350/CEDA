from train import *
from test import *

if __name__ == '__main__':
    train_file = 'workloads/forest/foresttrain'
    test_file = 'workloads/forest/foresttest'

    ##
    pattern2model = train(train_file, 300)
    t_for_all_pattern(test_file, pattern2model)