import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

import os

from dcgan import DCGAN
from config import gan_cfg, inference_cfg
from dataset import WaterBodyGeneratorDataset

# def test_segmentation_unet(data, model):

#     #data = iter(loader).next()

#     for img, mask in zip(data[0], data[1]):
#         prediction = model(img.unsqueeze(0).to("cuda"))
#         prediction = torch.where(prediction > 0.5, 1, 0)
#         show_image_mask(img, mask, prediction.detach().squeeze(0).squeeze(0).to("cpu"))




if __name__=='__main__':

    tf = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(
                mean    = [0.5, 0.5, 0.5],
                std     = [0.5, 0.5, 0.5])
        ]
    )

    path = "D:/AI_ML/Kaggle/Water Bodies Dataset_pruned_more/"
    dataset = WaterBodyGeneratorDataset(path, transform=tf)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                         shuffle=True, num_workers=0)

   
    fixed_noise = torch.randn(32, 100,1,1, device="cuda")
    content = []
   
    # CREATE MODELS
    models = [ DCGAN(config) for config in inference_cfg.configs]

    # GENERATE BATCHES
    for i, model in enumerate(models):
        model.load(os.path.join(inference_cfg.root_path, inference_cfg.models[i]), inference_cfg.epoch)
        batch = model.generate_batch(batch_size=inference_cfg.batch_size, threeshold=inference_cfg.threeshold,batch_gen_size=inference_cfg.batch_gen_size)
        batch = [(x[0],np.transpose(x[1],(1,2,0))) for x in batch]
        content.append(batch)
    
    # GENERATE GROUND TRUTH
    if inference_cfg.gt:
        gt = [("gt",np.transpose(o,(1,2,0))) for o in iter(dataloader).next()[:inference_cfg.batch_size]]
        content.append(gt)

    # DISPLAY IMAGES AS PLOT
    fig, axs = plt.subplots(len(content),inference_cfg.batch_size, figsize=(20,20))
    counter = 0
    for i in range(len(content)):
        for j in range(inference_cfg.batch_size):
            plt.axis("off")
            
            white = content[i][counter][1].max()
            black = content[i][counter][1].min()
            arr = (content[i][counter][1] - black)* (1/(white-black))
            if inference_cfg.gt and i== len(content) -1:
                axs[i][0].set_ylabel("ground truth")
            else:
                axs[i][0].set_ylabel(inference_cfg.models[i])

            axs[i][j].set_title(str(content[i][counter][0]))
            #print(arr)
            axs[i][j].imshow(arr)
            counter +=1
        counter = 0
    plt.show()



