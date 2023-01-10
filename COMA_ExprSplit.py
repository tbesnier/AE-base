
import vtk
from scipy.io import savemat, loadmat
import time
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import trimesh
import open3d as o3d
import os, argparse

parser = argparse.ArgumentParser(description='Arguments for dataset split')
parser.add_argument('-d','--test_label', type=int,
            help='Please choose the expression that will be used for testing here (check read_COMA_label function (defined bellow) to see the labels corresponding to different expression')
parser.add_argument('-v', '--target', type=int,
                        help='target or input data, target=1 -> save target data; target=0 -> save neutral data')

args = parser.parse_args()

test_label = args.test_label
target = bool(args.target)

def read_COMA_label(char_label):
    if 'bareteeth' in char_label:
        label = 0
    elif 'cheeks_in' in char_label:
        label = 1
    elif 'eyebrow' in char_label:
        label = 2
    elif 'high_smile' in char_label:
        label = 3
    elif 'lips_back' in char_label:
        label = 4
    elif 'lips_up' in char_label:
        label = 5
    elif 'mouth_down' in char_label:
        label = 6
    elif 'mouth_extreme' in char_label:
        label = 7
    elif 'mouth_middle' in char_label:
        label = 8
    elif 'mouth_open' in char_label:
        label = 9
    elif 'mouth_side' in char_label:
        label = 10
    elif 'mouth_up' in char_label:
        label = 11
    else:
        print('****************label not supported***********************')
    return label

train_coma=[]
test_coma=[]
data_path="..datasets/COMA"
count=0
for subjdir in os.listdir(data_path):
   print(subjdir)
   count=count+1
   for expr_dir in os.listdir(os.path.join(data_path, subjdir)):
       expression=read_COMA_label(expr_dir)
       cc=0
       for mesh in os.listdir(os.path.join(data_path, subjdir, expr_dir)):
           # For neutral data
           if not target:
               if cc==0: ## consider only the first neutral face
                  data_loaded = trimesh.load(os.path.join(data_path, subjdir, expr_dir, mesh), process=False)
               vertices=data_loaded.vertices
               if expression == test_label:
                   test_coma.append(vertices)
               else:
                   train_coma.append(vertices)
           else:
               data_loaded = trimesh.load(os.path.join(data_path, subjdir, expr_dir, mesh), process=False)
               vertices = data_loaded.vertices
               if expression == test_label:
                   test_coma.append(vertices)
               else:
                   train_coma.append(vertices)

print(np.shape(train_coma))
print(np.shape(test_coma))

if not target:
    if not os.path.exists(os.path.join('Data', 'COMA', 'preprocessed_neutral_expr_pointnet_original')):
        os.makedirs(os.path.join('Data', 'COMA', 'preprocessed_neutral_expr_pointnet_original'))
    np.save('./Data/COMA/preprocessed_neutral_expr_pointnet_original/train.npy', train_coma)
    np.save('./Data/COMA/preprocessed_neutral_expr_pointnet_original/test.npy', test_coma)
else:
    if not os.path.exists(os.path.join('Data', 'COMA', 'preprocessed_expr_pointnet_original')):
        os.makedirs(os.path.join('Data', 'COMA', 'preprocessed'))
    np.save('./Data/COMA/preprocessed_expr_pointnet_original/train.npy', train_coma)
    np.save('./Data/COMA/preprocessed_expr_pointnet_original/test.npy', test_coma)
