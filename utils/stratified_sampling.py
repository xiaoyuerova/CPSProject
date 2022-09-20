import numpy
import pandas as pd


def stratified_sampling(data_path, sample_num, outputs_path=None, typical_dict=None, random_seed=None, index=False):
    """

    :param data_path:
    :param outputs_path:
    :param sample_num:
    :param typical_dict:
    :param random_seed:
    :param index:
    :return:
    """
    df = pd.read_csv(data_path)
    sample_num_dict = {}
    if random_seed:
        numpy.random.seed(random_seed)  # 如果使用相同的seed( )值，则每次生成的随即数都相同，
    if typical_dict:
        sample_num_dict = typical_dict
    else:
        _class = df['Labels'].value_counts()
        sample_num_dict = {index: sample_num for index in _class.index}

    result = df.groupby('Labels').apply(typical_sampling, sample_num_dict)

    if outputs_path:
        result.to_csv(outputs_path, index=index)

    return result


def typical_sampling(group, sample_num_dict):
    name = group.name
    n = sample_num_dict[name]
    return group.sample(n=n, replace=True)


if __name__ == '__main__':
    data_path = '../data/single-sentence-prediction/data.csv'
    output_path = '../data/single-sentence-prediction/single-sentence-data.csv'
    stratified_sampling(data_path, 2000, output_path, random_seed=1)
