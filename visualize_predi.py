import os

import numpy as np
import trimesh
from tqdm import tqdm
import os, argparse

parser = argparse.ArgumentParser(description='Arguments for results viz')
parser.add_argument('-m','--model', type=str,
            help='model')
parser.add_argument('-l','--latent_size', type=str,
            help='size of latent space')

args = parser.parse_args()

def save_viz(dest, path):
    if os.path.exists(path):
        ours = np.load(path)
        for i in tqdm(range(len(ours)), "saving vizualizations"):
            tri_mesh = trimesh.Trimesh(np.asarray(np.squeeze(ours[i, :, :3])), np.asarray(triangles), process=False)
            tri_mesh.export(os.path.join(dest, "tst{0:03}.ply".format(i)))
    else:
        print("This predictions does not exist")


def save_viz_bad(dest, path):
    if os.path.exists(path):
        ours = np.load(path)
        for i in tqdm(range(len(ours)), "saving vizualizations"):
            tri_mesh = trimesh.convex.convex_hull(np.asarray(np.squeeze(ours[i, :, :3])))
            tri_mesh.export(os.path.join(dest, "tst{0:03}.ply".format(i)))
    else:
        print("This predictions does not exist")

tri = trimesh.load('../Data/COMA/template/template.obj', process=False)
triangles=tri.faces

model = args.model
latent = args.latent_size
folder = '../Data/COMA/results_identity/{0}_autoencoder/latent_{1}'.format(model, latent)

dest = os.path.join(folder, 'viz_predictions')
os.makedirs(dest, exist_ok=True)
save_viz(dest, os.path.join(folder, 'predictions/predictions.npy'))

dest = os.path.join(folder, 'viz_targets')
os.makedirs(dest, exist_ok=True)
save_viz(dest, os.path.join(folder, 'predictions/targets.npy'))
