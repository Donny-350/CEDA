from model import *
from util import *
from data import *
from torch import optim
import csv

def fune_train(path, epochs=10):
    model_path = 'model/1.pth'
    model = MLP(20, 128)
    model.load_state_dict(torch.load(model_path))
    pattern2training, pattern2truecard, min_card_log, max_card_log = prepare_pattern_workload(path)

    for k, v in pattern2training.items():
        train_data = np.array(v)
        labels = normalize(pattern2truecard[k], min_card_log, max_card_log)

        batch_size = 64
        learning_rate = 0.000005

        training_data = torch.FloatTensor(train_data)
        training_label = torch.FloatTensor(labels)
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(training_data, training_label),
                                                   batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        Loss = nn.MSELoss()
        model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                logits = model(data)
                loss = Loss(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Train Epoch: {} {} \tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), loss.item()))

    # 写入csv文件
    cards = []
    true_cards = []
    for k, v in pattern2training.items():
        cards += unnormalize(estimate(model, np.array(v)), min_card_log, max_card_log).tolist()
        true_cards += pattern2truecard[k]


    print_qerror(np.array(cards), np.array(true_cards))
    print_mse(np.array(cards), np.array(true_cards))
    print_mape(np.array(cards), np.array(true_cards))
    print_pearson_correlation(np.array(cards), np.array(true_cards))

    headers = ('estimate_card', 'true_card')

    with open('result/forest_fune_tuning_680_0.000005.csv', 'w') as f:
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



if __name__ == '__main__':
    train_path = 'workloads/forest/foresttest.sql'
    fune_train(train_path, 5)