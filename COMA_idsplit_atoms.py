import numpy as np
import trimesh
import os, argparse
from tqdm import tqdm
from plyfile import PlyData, PlyElement


ids = ['FaceTalk_170725_00137_TA', 'FaceTalk_170728_03272_TA']#'FaceTalk_170731_00024_TA', 'FaceTalk_170809_00138_TA',
       #'FaceTalk_170811_03274_TA', 'FaceTalk_170811_03275_TA', 'FaceTalk_170904_00128_TA', 'FaceTalk_170904_03276_TA',
      #'FaceTalk_170908_03277_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170913_03279_TA', 'FaceTalk_170915_00223_TA']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for dataset split')
    parser.add_argument('-d','--test_label', type=int,
                help='Please choose the expression that will be used for testing here (check read_COMA_label function (defined bellow) to see the labels corresponding to different expression')
    parser.add_argument('-v', '--target', type=int, default=0,
                        help='target or input data, target=1 -> save target data; target=0 -> save neutral data')

    args = parser.parse_args()

    test_label = args.test_label
    target = bool(args.target)
    train_coma=[]
    test_coma=[]
    data_path= "../datasets/COMA_2/"

    def to_measure(points, triangles):
        """Turns a triangle into a weighted point cloud."""

        # Our mesh is given as a collection of ABC triangles:
        A, B, C = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]

        # Locations and weights of our Dirac atoms:
        X = (A + B + C) / 3  # centers of the faces
        S = np.sqrt(np.sum(np.cross(B - A, C - A) ** 2, 1)) / 2  # areas of the faces

        # We return a (normalized) vector of weights + a "list" of points
        return np.array(S / np.sum(S)), np.array(X)
    def load_ply_file(fname):
        """Loads a .ply mesh to return a collection of weighted Dirac atoms: one per triangle face."""

        # Load the data, and read the connectivity information:
        plydata = PlyData.read(fname)
        triangles = np.vstack(plydata["face"].data["vertex_indices"])
        # Normalize the point cloud, as specified by the user:
        points = np.vstack([[v[0], v[1], v[2]] for v in plydata["vertex"]])

        return to_measure(points, triangles)

    count=0
    subjs = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    for subjdir in subjs:
        count=count+1
        subject = ids.index(subjdir)
        #print(subject, subjdir)
        for expr_dir in os.listdir(os.path.join(data_path, subjdir)):
            cc=0
            if target:
                for mesh in tqdm(os.listdir(os.path.join(data_path, subjdir, expr_dir)), "Processing folder: {0} {1}".format(subjdir, expr_dir)):
                    #data_loaded = trimesh.load(os.path.join(data_path, subjdir, expr_dir, mesh), process=False)
                    area, points = load_ply_file(os.path.join(data_path, subjdir, expr_dir, mesh))
                    data_loaded = np.vstack((points.T, area.T)).T
                    #vertices = data_loaded.vertices
                    if subject == test_label:
                        #print(subject, subjdir)
                        test_coma.append(data_loaded)
                    else:
                        train_coma.append(data_loaded)
            else:
                # For neutral data
                cc = 0
                for mesh in os.listdir(os.path.join(data_path, subjdir, expr_dir)):
                    if cc==0: ## consider only the first neutral face
                        #data_loaded = trimesh.load(os.path.join(data_path, subjdir, expr_dir, mesh), process=False)
                        data_loaded = load_ply_file(os.path.join(data_path, subjdir, expr_dir, mesh))
                        cc = cc + 1
                        #vertices=data_loaded.vertices
                        if subject == test_label:
                           test_coma.append(data_loaded)
                        else:
                           train_coma.append(data_loaded)

    print(np.shape(train_coma))
    print(np.shape(test_coma))

    if not target:
        if not os.path.exists(os.path.join('../', 'Data', 'COMA', 'preprocessed_neutral_pointnet_original')):
            os.makedirs(os.path.join('../', 'Data', 'COMA', 'preprocessed_neutral_pointnet_original'))
        np.save('../Data/COMA/preprocessed_neutral_pointnet_original/train.npy', train_coma)
        np.save('../Data/COMA/preprocessed_neutral_pointnet_original/test.npy', test_coma)
    else:
        if not os.path.exists(os.path.join('../', 'Data', 'COMA', 'preprocessed_identity_pointnet_original')):
            os.makedirs(os.path.join('../', 'Data', 'COMA', 'preprocessed_identity_pointnet_original'))
        np.save('../Data/COMA/preprocessed_identity_pointnet_original/train.npy', train_coma)
        np.save('../Data/COMA/preprocessed_identity_pointnet_original/test.npy', test_coma)


