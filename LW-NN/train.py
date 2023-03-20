from torch import optim
from data import *
from model import *
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

def train(path, num_round=10):
    pattern2training, pattern2truecard, min_card_log, max_card_log = prepare_pattern_workload(path)
    print('min_card_log: {}, max_card_log: {}'.format(min_card_log, max_card_log))
    pattern2model = {}

    for k, v in pattern2training.items():
        input_dim = len(v[0])
        train_data = np.array(v)
        labels = normalize(pattern2truecard[k], min_card_log, max_card_log)

        hidden_dim = 128
        model = MLP(input_dim, hidden_dim)
        batch_size = 64
        learning_rate = 0.01
        print(train_data.shape, labels.shape)
        train_len = int(0.8 * len(train_data))
        training_data = torch.FloatTensor(train_data[:train_len])
        training_label = torch.FloatTensor(labels[:train_len]).unsqueeze(1)
        validate_data = torch.FloatTensor(train_data[train_len:])
        validate_label = torch.FloatTensor(labels[train_len:]).unsqueeze(1)
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(training_data, training_label),
                                                   batch_size=batch_size, shuffle=True)
        validate_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(validate_data, validate_label),
                                                      batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        Loss = nn.MSELoss()
        model.train()
        for epoch in range(num_round):
            for batch_idx, (data, target) in enumerate(train_loader):
                logits = model(data)
                loss = Loss(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Train Epoch: {} {} \tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), loss.item()))
        model.eval()
        test_loss = 0
        for data, target in validate_loader:
            logits = model(data)
            test_loss += Loss(logits, target).item()
        test_loss /= len(validate_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, {} \n'.format(
            test_loss, len(validate_loader.dataset)))
        pattern2model[k] = model
        torch.save(model.state_dict(), 'model/1.pth')
        return pattern2model
# if __name__ == '__main__':
#     train_path = 'workloads/forest/foresttrain.sql'
#     train(train_path, 100)
