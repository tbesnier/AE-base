import numpy as np
import trimesh
import os, argparse
from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for dataset split')
    parser.add_argument('-d', '--test_label', type=int,
                        help='Please choose the expression that will be used for testing here (check read_COMA_label function (defined bellow) to see the labels corresponding to different expression')
    parser.add_argument('-v', '--target', type=int, default=0,
                        help='target or input data, target=1 -> save target data; target=0 -> save neutral data')

    args = parser.parse_args()

    test_label = args.test_label
    target = bool(args.target)
    print(target)
    train_set=[]
    test_set=[]
    data_path="../datasets/artificial_database/ref_base"

    count = 0

    for f in os.listdir(data_path):
        if count < 900:
            train_set.append(trimesh.load(os.path.join(data_path, f), process=False).vertices)
        else:
            test_set.append(trimesh.load(os.path.join(data_path, f), process=False).vertices)
        count += 1

    print(np.shape(train_set))
    print(np.shape(test_set))

    if not os.path.exists(os.path.join('../Data','ELLIPSES','preprocessed')):
        os.makedirs(os.path.join('../Data', 'ELLIPSES', 'preprocessed'))
    np.save('../Data/ELLIPSES/preprocessed/train.npy', train_set)

    if not os.path.exists(os.path.join('Data', 'ELLIPSES', 'preprocessed')):
        os.makedirs(os.path.join('Data', 'ELLIPSES', 'preprocessed'))
    np.save('../Data/ELLIPSES/preprocessed/test.npy', test_set)


