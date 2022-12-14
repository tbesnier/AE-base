import torch
import copy
from tqdm import tqdm
import numpy as np

def test_autoencoder_dataloader(device, model, dataloader_test, shapedata, mm_constant = 1000):
    model.eval()
    l1_loss = 0
    l2_loss = 0
    shapedata_mean = torch.Tensor(shapedata.mean).to(device)[:,0:3]
    shapedata_std = torch.Tensor(shapedata.std).to(device)[:,0:3]
    #print(shapedata_mean.shape, shapedata_std.shape)
    with torch.no_grad():
        for i, sample_dict in enumerate(tqdm(dataloader_test)):
            tx = sample_dict['points'].to(device)
            prediction = model(tx)

            prediction_recons = prediction
            if dataloader_test.dataset.dummy_node:
                x_recon = prediction_recons[:,:-1]
                x = tx[:,:-1]
                x_points = tx[:, :, 0:3][:,:-1]
            else:
                x_recon = prediction_recons
                x = tx
                x_points = tx[:, :, 0:3]

            l1_loss+= torch.mean(torch.abs(x_recon-x_points))*x.shape[0]/float(len(dataloader_test.dataset))
            
            x_recon = (x_recon * shapedata_std + shapedata_mean) * mm_constant
            x_points = (x_points * shapedata_std + shapedata_mean) * mm_constant
            if i==0:
                predictions = copy.deepcopy(x_recon)
                target = copy.deepcopy(x_points)
            else:
                predictions = torch.cat([predictions, x_recon],0)
                target = torch.cat([target, x_points],0)
            #x = (x * shapedata_std + shapedata_mean) * mm_constant
            l2_loss+= torch.mean(torch.sqrt(torch.sum((x_recon - x_points)**2,dim=2)))*x_points.shape[0]/float(len(dataloader_test.dataset))
            
        predictions = predictions.cpu().numpy()
        target = target.cpu().numpy()
        l1_loss = l1_loss.item()
        l2_loss = l2_loss.item()
    
    return target, predictions, l1_loss, l2_loss