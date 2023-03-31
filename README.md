# CEDA: Learned Cardinality Estimation with Domain Adaptation

A PyTorch Implementation of CEDA. We also provide the source code for data import, getting PostgreSQL cardinality estimation, and getting histogram information.

## Requirements

- Python 3.9.7
- PyTorch 1.10.2

## Dataset

The project works on PostgreSQL 11. If you do not have PostgreSQL installed, you first need to install [PostgreSQL](https://www.postgresql.org/download/). Our experiments are conducted on the [IMDB](https://www.imdb.com/interfaces/) and [Forest](https://archive.ics.uci.edu/ml/datasets/Covertype) datasets. You can use our forest.py script to import forest data into PostgreSQL, and similarly, we also provide a script to import Power data. 

Example usage:

```shell
cd importForestAndPower
python forest.py
```

Because the encoding part of our Attention-based Cardinality Estimator integrates histogram information. You can use a script we wrote to collect histogram information.

For example:

If you want to get the histogram information of Forest, you can run getForestHistogram.ipynb to get a histogram_forest.csv.

## Train & Test

You can train and test the model performance using the following command.

```shell
python train.py
python trainDA.py
```

In order to compare with LW-NN, we reproduced its code. You can run the train.py training model under the LW-NN folder, and test the model effect with test.py.

```shell
cd Lw-NN
python train.py
python test.py
```

If you want to compare with traditional PostgreSQL, you can run get_cardinality_estimate_actual.py in the get-postgresql-cardinality folder, and then you will get a csv result file, in which you can see the estimated cardinality, actual cardinality, q-error, etc. information.

```shell
cd get-postgresql-cardinality
python get_cardinality_estimate_actual.py
```



## Code References:

- MSCN: [https://github.com/andreaskipf/learnedcardinalities](https://github.com/andreaskipf/learnedcardinalities)

- DANN: [https://github.com/zengjichuan/DANN](https://github.com/zengjichuan/DANN)

