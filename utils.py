import os

import numpy as np
import trimesh
from tqdm import tqdm

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

#tri = trimesh.load('Data/ELLIPSES/template/template.obj', process=False)
#triangles=tri.faces

model = 'pointnet_original'
latent = 128
folder = 'Data/ELLIPSES/results/{0}_autoencoder/latent_{1}'.format(model, latent)

dest = os.path.join(folder, 'viz_predictions')
os.makedirs(dest, exist_ok=True)
save_viz(dest, os.path.join(folder, 'predictions/predictions.npy'))

