import numpy as np
import json
import os
import copy
import pickle

import mesh_sampling
import trimesh
from shape_data import ShapeData
from SimplePointNet import SimplePointNet, PointNetAutoEncoder, Decoder, Encoder, SimpleTransformer

from autoencoder_dataset import autoencoder_dataset
from torch.utils.data import DataLoader

from spiral_utils import get_adj_trigs, generate_spirals
from models import SpiralAutoencoder
from train_funcs import train_autoencoder_dataloader
from test_funcs import test_autoencoder_dataloader


import torch
torch.cuda.empty_cache()
from tensorboardX import SummaryWriter

from sklearn.metrics.pairwise import euclidean_distances
meshpackage = 'trimesh' # 'mpi-mesh', 'trimesh'
root_dir = '../Data'

dataset = 'COMA'
name = ''

GPU = True
device_idx = 0
torch.cuda.get_device_name(device_idx)

args = {}

generative_model = 'autoencoder'
downsample_method = 'COMA_downsample'  # choose'COMA_downsample' or 'meshlab_downsample'

# below are the arguments for the DFAUST run
reference_mesh_file = os.path.join(root_dir, dataset, 'template', 'template.obj')
downsample_directory = os.path.join(root_dir, dataset, 'template', downsample_method)
ds_factors = [4, 4, 4, 4]
step_sizes = [2, 2, 1, 1, 1]
filter_sizes_enc = [[64, 128], 64, [128, 64, 64]]
filter_sizes_dec = [128]
dilation_flag = True
if dilation_flag:
    dilation = [2, 2, 1, 1, 1]
else:
    dilation = None
reference_points = [[3567, 4051, 4597]]  # [[414]]  # [[3567,4051,4597]] used for COMA with 3 disconnected components

args = {'generative_model': generative_model,
        'name': name, 'data': os.path.join(root_dir, dataset, 'preprocessed_identity_pointnet_original', name),
        'results_folder': os.path.join(root_dir, dataset, 'results_identity/pointnet_original_' + generative_model),
        'reference_mesh_file': reference_mesh_file, 'downsample_directory': downsample_directory,
        'checkpoint_file': 'checkpoint',
        'seed': 2, 'loss': 'current',
        'batch_size': 2, 'num_epochs': 200, 'eval_frequency': 200, 'num_workers': 4,
        'filter_sizes_enc': filter_sizes_enc, 'filter_sizes_dec': filter_sizes_dec,
        'nz': 128,
        'ds_factors': ds_factors, 'step_sizes': step_sizes, 'dilation': dilation,
        'lr': 1e-3,
        'regularization': 5e-5,
        'scheduler': True, 'decay_rate': 0.99, 'decay_steps': 1,
        'resume': False,
        'mode': 'train', 'shuffle': True, 'nVal': 1, 'normalization': True}

args['results_folder'] = os.path.join(args['results_folder'], 'latent_' + str(args['nz']))

if not os.path.exists(os.path.join(args['results_folder'])):
    os.makedirs(os.path.join(args['results_folder']))

summary_path = os.path.join(args['results_folder'], 'summaries', args['name'])
if not os.path.exists(summary_path):
    os.makedirs(summary_path)

checkpoint_path = os.path.join(args['results_folder'], 'checkpoints', args['name'])
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

samples_path = os.path.join(args['results_folder'], 'samples', args['name'])
if not os.path.exists(samples_path):
    os.makedirs(samples_path)

prediction_path = os.path.join(args['results_folder'], 'predictions', args['name'])
if not os.path.exists(prediction_path):
    os.makedirs(prediction_path)

if not os.path.exists(downsample_directory):
    os.makedirs(downsample_directory)

np.random.seed(args['seed'])
print("Loading data .. ")
if not os.path.exists(args['data'] + '/mean.npy') or not os.path.exists(args['data'] + '/std.npy'):
    shapedata = ShapeData(nVal=args['nVal'],
                          train_file=args['data'] + '/train.npy',
                          test_file=args['data'] + '/test.npy',
                          reference_mesh_file=args['reference_mesh_file'],
                          normalization=args['normalization'],
                          meshpackage=meshpackage, load_flag=True, mean_subtraction_only=False)
    np.save(args['data'] + '/mean.npy', shapedata.mean)
    np.save(args['data'] + '/std.npy', shapedata.std)
else:
    shapedata = ShapeData(nVal=args['nVal'],
                          train_file=args['data'] + '/train.npy',
                          test_file=args['data'] + '/test.npy',
                          reference_mesh_file=args['reference_mesh_file'],
                          normalization=args['normalization'],
                          meshpackage=meshpackage, load_flag=False, mean_subtraction_only=False)
    shapedata.mean = np.load(args['data'] + '/mean.npy')
    shapedata.std = np.load(args['data'] + '/std.npy')
    shapedata.n_vertex = shapedata.mean.shape[0]
    shapedata.n_features = shapedata.mean.shape[1]

print("... Done")

torch.manual_seed(args['seed'])

if GPU:
    device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

# Building model, optimizer, and loss function

dataset_train = autoencoder_dataset(root_dir = args['data'], points_dataset = 'train',
                                           shapedata = shapedata,
                                           normalization = args['normalization'])

dataloader_train = DataLoader(dataset_train, batch_size=args['batch_size'],\
                                     shuffle = args['shuffle'], num_workers = args['num_workers'])

dataset_val = autoencoder_dataset(root_dir = args['data'], points_dataset = 'val',
                                         shapedata = shapedata,
                                         normalization = args['normalization'])

dataloader_val = DataLoader(dataset_val, batch_size=args['batch_size'],\
                                     shuffle = False, num_workers = args['num_workers'])

dataset_test = autoencoder_dataset(root_dir = args['data'], points_dataset = 'test',
                                          shapedata = shapedata,
                                          normalization = args['normalization'])

dataloader_test = DataLoader(dataset_test, batch_size=args['batch_size'],\
                                     shuffle = False, num_workers = args['num_workers'])


if 'autoencoder' in args['generative_model']:
        model = PointNetAutoEncoder(latent_size=args['nz'],
                                    filter_enc = args['filter_sizes_enc'],
                                    filter_dec = args['filter_sizes_dec'],
                                    num_points=shapedata.n_vertex+1,
                                    device=device).to(device)


optim = torch.optim.Adam(model.parameters(),lr=args['lr'],weight_decay=args['regularization'])
if args['scheduler']:
    scheduler=torch.optim.lr_scheduler.StepLR(optim, args['decay_steps'],gamma=args['decay_rate'])
else:
    scheduler = None

template_mesh = trimesh.load("../Data/COMA/template/template.obj")
faces = torch.Tensor(template_mesh.faces)
torchdtype = torch.float

if args['loss'] == 'current':
    sigma = 0.4
    sigma = torch.tensor([sigma], dtype=torchdtype, device=device)

    import lddmm_utils

    # PyKeOps counterpart
    KeOpsdeviceId = device.index  # id of Gpu device (in case Gpu is  used)
    KeOpsdtype = torchdtype.__str__().split(".")[1]  # 'float32'

    new_faces = faces.repeat(args['batch_size'], 1, 1)

    def loss_varifold(outputs, targets):
        new_faces = faces.repeat(outputs.shape[0], 1, 1)
        V1, F1 = outputs.to(dtype=torchdtype, device=device).requires_grad_(True), new_faces.to(dtype=torch.int32,
                                                                                                device=device)
        V2, F2 = targets.to(dtype=torchdtype, device=device).requires_grad_(True), new_faces.to(dtype=torch.int32,
                                                                                                device=device)

        L = torch.stack([lddmm_utils.lossVarifoldSurf(F1[i], V2[i], F2[i],
                                                      lddmm_utils.GaussLinKernel_current(sigma=sigma))(V1[i]) for i in
                         range(new_faces.shape[0])]).requires_grad_(True).sqrt().mean()

        return L

    loss_fn = loss_varifold

if args['loss'] == 'varifold':
    sigma = 0.1
    sigma = torch.tensor([sigma], dtype=torchdtype, device=device)

    import lddmm_utils

    # PyKeOps counterpart
    KeOpsdeviceId = device.index  # id of Gpu device (in case Gpu is  used)
    KeOpsdtype = torchdtype.__str__().split(".")[1]  # 'float32'

    new_faces = faces.repeat(args['batch_size'], 1, 1)

    def loss_varifold(outputs, targets):
        new_faces = faces.repeat(outputs.shape[0], 1, 1)
        V1, F1 = outputs.to(dtype=torchdtype, device=device).requires_grad_(True), new_faces.to(dtype=torch.int32,
                                                                                                device=device)
        V2, F2 = targets.to(dtype=torchdtype, device=device).requires_grad_(True), new_faces.to(dtype=torch.int32,
                                                                                                device=device)

        L = torch.stack([lddmm_utils.lossVarifoldSurf(F1[i], V2[i], F2[i],
                                                      lddmm_utils.GaussLinKernel_varifold_oriented(sigma=sigma))(V1[i]) for i in
                         range(new_faces.shape[0])]).requires_grad_(True).sqrt().mean()

        return L

    loss_fn = loss_varifold

params = sum(p.numel() for p in model.parameters() if p.requires_grad)

if args['mode'] == 'train':
    writer = SummaryWriter(summary_path)
    with open(os.path.join(args['results_folder'], 'checkpoints', args['name'] + '_params.json'), 'w') as fp:
        saveparams = copy.deepcopy(args)
        json.dump(saveparams, fp)

    if args['resume']:
        print('loading checkpoint from file %s' % (os.path.join(checkpoint_path, args['checkpoint_file'])))
        checkpoint_dict = torch.load(os.path.join(checkpoint_path, args['checkpoint_file'] + '.pth.tar'),
                                     map_location=device)
        start_epoch = checkpoint_dict['epoch'] + 1
        model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])
        optim.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
        print('Resuming from epoch %s' % (str(start_epoch)))
    else:
        start_epoch = 0

    if args['generative_model'] == 'autoencoder':
        train_autoencoder_dataloader(dataloader_train, dataloader_val,
                                     device, model, optim, loss_fn,
                                     bsize=args['batch_size'],
                                     start_epoch=start_epoch,
                                     n_epochs=args['num_epochs'],
                                     eval_freq=args['eval_frequency'],
                                     scheduler=scheduler,
                                     writer=writer,
                                     save_recons=True,
                                     shapedata=shapedata,
                                     metadata_dir=checkpoint_path, samples_dir=samples_path,
                                     checkpoint_path=args['checkpoint_file'])

args['mode'] = 'test'
if args['mode'] == 'test':
    print('loading checkpoint from file %s' % (os.path.join(checkpoint_path, args['checkpoint_file'] + '.pth.tar')))
    checkpoint_dict = torch.load(os.path.join(checkpoint_path, args['checkpoint_file'] + '.pth.tar'),
                                 map_location=device)
    model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])

    targets, predictions, norm_l1_loss, l2_loss = test_autoencoder_dataloader(device, model, dataloader_test,
                                                                              shapedata, mm_constant=1000)
    np.save(os.path.join(prediction_path, 'predictions'), predictions)
    np.save(os.path.join(prediction_path, 'targets'), targets)

    print('autoencoder: normalized loss', norm_l1_loss)

    print('autoencoder: euclidean distance in mm=', l2_loss)