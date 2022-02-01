import torchvision.utils as vutils
from dcgan import *

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
# def test_segmentation_unet(data, model):

#     #data = iter(loader).next()

#     for img, mask in zip(data[0], data[1]):
#         prediction = model(img.unsqueeze(0).to("cuda"))
#         prediction = torch.where(prediction > 0.5, 1, 0)
#         show_image_mask(img, mask, prediction.detach().squeeze(0).squeeze(0).to("cpu"))






def make_grid(data):
    return vutils.make_grid(data.detach().cpu(), padding=2, normalize=True)

def load_weights(model, path):
    model.load_state_dict(torch.load(path))


def generate_batch(generator, discrimintor, threeshold=0.55, batch_size=16, batch_gen_size=64):
    images = []

    while(len(images) < batch_size):
        fixed_noise = torch.randn(batch_gen_size, 100,1,1, device="cuda")
        out = generator(fixed_noise)
        weights = discrimintor(out)
        for i,o in enumerate(out):
            if weights[i] > threeshold:
                images.append((round(weights[i].item(),4),o.detach().cpu().numpy()))
        print(len(images))

    return images[:batch_size]

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

    # gen_name = "generator_size_64_fs_128"
    # disc_name = "discriminator_size_64_fs_64"
    root = "../../models/dcgan/out/"
    models = ["dcgan_FM_HA_Test", "old_model_20_batches_prunedmore"]
    epochs = [30]
    
    fixed_noise = torch.randn(32, 100,1,1, device="cuda")
    dict = {}
    # for model_name in models:
    #     dict[model_name] = {"grids":[], "out":[],"disc":[]}
    #     for epoch in epochs:
    #         load_weights(m, f"{root}{model_name}/{gen_name}_{epoch}.pth")        
    #         m.eval()
    #         out = m(fixed_noise)

    #         grid = make_grid(out)
    #         dict[model_name]["grids"].append(grid)
    #         dict[model_name]["out"].append(out.detach().cpu())
    #         dict[model_name]["disc"].append(d(out).detach().cpu())


    
    # ground_truth = make_grid(iter(dataloader).next())
    
    # fig, axs = plt.subplots(len(epochs), 3, figsize=(20,20))
    # for i in range(len(epochs)):
        
    #     axs[0].set_title(f"{models[0]}_{epochs[i]}")
        
    #     print(dict[models[0]]["grids"][i].min())
    #     print(dict[models[0]]["grids"][i].max())
    #     axs[0].imshow(np.transpose(dict[models[0]]["grids"][i], (1,2,0)))
    #     axs[1].set_title(f"{models[1]}_{epochs[i]}")
    #     axs[1].imshow(np.transpose(dict[models[1]]["grids"][i], (1,2,0)))
    #     axs[2].set_title("ground_truth")
    #     axs[2].imshow(np.transpose(ground_truth, (1,2,0)))   
    # plt.show()
    epoch = 150
    #m = Generator(100,3).to("cuda")
    #d = Discriminator(3).to("cuda")
    # load_weights(m, f"{root}{models[0]}/{gen_name}_{epoch}.pth")
    # load_weights(d,f"{root}{models[0]}/{disc_name}_{epoch}.pth")
    # m.eval()
    # d.eval()
    # batch = generate_batch(m,d,0.75,6,64)
    # batch = [(x[0],cv2.resize(np.transpose(x[1],(1,2,0)),(256,256),interpolation=cv2.INTER_AREA)) for x in batch]
    
    #load_weights(m, f"{root}{models[1]}/generator_size_128_fs_128_80.pth")
    #load_weights(d,f"{root}{models[1]}/discriminator_size_128_fs_32_80.pth")

    gan = DCGAN()
    # gan.load(os.path.join(root,models[1]),"140")

    BS = 4
    # batch2 = gan.generate_batch(batch_size=BS,threeshold=0.5,batch_gen_size=128)
    # batch2 = [(x[0],np.transpose(x[1],(1,2,0))) for x in batch2]


    gan.load(os.path.join(root,models[0]),"30")
    batch = gan.generate_batch(batch_size=BS*2,threeshold=0.0,batch_gen_size=256)
    batch = [(x[0],np.transpose(x[1],(1,2,0))) for x in batch]


    #gt = [("gt",np.transpose(o,(1,2,0))) for o in iter(dataloader).next()[:6]]
    content =batch
    #content = gt
    #random.shuffle(content)
    fig, axs = plt.subplots(2,BS, figsize=(20,20))
    counter = 0
    for i in range(2):
        for j in range(BS):
            plt.axis("off")
            
            white = content[counter][1].max()
            black = content[counter][1].min()
            arr = (content[counter][1] - black)* (1/(white-black))
            
            axs[i][j].set_title(str(content[counter][0]))
            #print(arr)
            axs[i][j].imshow(arr)
            counter +=1
    plt.show()


