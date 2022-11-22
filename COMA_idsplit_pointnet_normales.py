import numpy as np
import trimesh
import os, argparse
from tqdm import tqdm


ids = ['FaceTalk_170725_00137_TA', 'FaceTalk_170728_03272_TA', 'FaceTalk_170731_00024_TA', 'FaceTalk_170809_00138_TA',
       'FaceTalk_170811_03274_TA', 'FaceTalk_170811_03275_TA', 'FaceTalk_170904_00128_TA', 'FaceTalk_170904_03276_TA',
       'FaceTalk_170908_03277_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170913_03279_TA', 'FaceTalk_170915_00223_TA']


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Arguments for dataset split')
    parser.add_argument('-d','--test_label', type=int,
                help='Please choose the expression that will be used for testing here (check read_COMA_label function (defined bellow) to see the labels corresponding to different expression')
    parser.add_argument('-v', '--target', type=int, default=0,
                        help='target or input data, target=1 -> save target data; target=0 -> save neutral data')

    args = parser.parse_args()

    test_label = args.test_label
    target = bool(args.target)
    print(target)
    train_coma=[]
    test_coma=[]
    data_path= "C:/Users/mrtho/Desktop/Centrale/G3/Projet/DL_geometric/neural3dmm_projet_integration/COMA/"

    count=0
    subjs = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    for subjdir in subjs:
        count=count+1
        subject = ids.index(subjdir)
        print(subject, subjdir)
        for expr_dir in os.listdir(os.path.join(data_path, subjdir)):
            cc=0
            if target:
                for mesh in tqdm(os.listdir(os.path.join(data_path, subjdir, expr_dir)), "Processing folder: {0} {1}".format(subjdir, expr_dir)):
                    data_loaded = trimesh.load(os.path.join(data_path, subjdir, expr_dir, mesh), process=False)
                    vertices = data_loaded.vertices
                    normals = data_loaded.vertex_normals
                    #print(vertices.shape)
                    #print(normals.shape)
                    data = np.hstack((vertices, normals))
                    if subject == test_label:
                        test_coma.append(data)
                    else:
                        train_coma.append(data)

            else:
                # For neutral data
                cc = 0
                for mesh in os.listdir(os.path.join(data_path, subjdir, expr_dir)):
                    if cc==0: ## consider only the first neutral face
                        data_loaded = trimesh.load(os.path.join(data_path, subjdir, expr_dir, mesh), process=False)
                        cc = cc + 1
                        vertices = data_loaded.vertices
                        normals = data_loaded.vertex_normals
                        data = np.hstack((vertices, normals))
                        if subject == test_label:
                           test_coma.append(data)
                        else:
                           train_coma.append(data)


    print(np.shape(train_coma))
    print(np.shape(test_coma))

    if not target:
        if not os.path.exists(os.path.join('Data', 'COMA', 'preprocessed_neutral_pointnet_normales')):
            os.makedirs(os.path.join('Data', 'COMA', 'preprocessed_neutral_pointnet_normales'))
        np.save('./Data/COMA/preprocessed_neutral_pointnet_normales/train.npy', train_coma)
        np.save('./Data/COMA/preprocessed_neutral_pointnet_normales/test.npy', test_coma)
    else:
        if not os.path.exists(os.path.join('Data', 'COMA', 'preprocessed_identity_pointnet_normales')):
            os.makedirs(os.path.join('Data', 'COMA', 'preprocessed_identity_pointnet_normales'))
        np.save('./Data/COMA/preprocessed_identity_pointnet_normales/train.npy', train_coma)
        np.save('./Data/COMA/preprocessed_identity_pointnet_normales/test.npy', test_coma)


