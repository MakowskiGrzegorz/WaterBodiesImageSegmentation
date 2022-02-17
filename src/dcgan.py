from albumentations.augmentations.crops.transforms import CenterCrop
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Tanh
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import WaterBodyGeneratorDataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import random
import os
from tqdm import tqdm
from utils import save_weights, show_anim, show_loss
from config import GANConfig, gan_cfg, train_cfg, DEVICE


from discriminator import Discriminator
from generator import Generator

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1 and classname.find("Block") == -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1 and classname.find("Block") == -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#DEVICE = "cuda"

class DCGAN(nn.Module):
    """Some Information about DCGAN"""
    def __init__(self,config:GANConfig):
        super(DCGAN, self).__init__()
        self.config = config
        self.generator = Generator(config).to(DEVICE)
        self.generator.apply(weights_init)
        self.discriminator = Discriminator(config).to(DEVICE)
        self.discriminator.apply(weights_init)

        self.criterion = nn.BCELoss()

    def discriminator_train(self, x):
        # DISCRIMINATOR ON REAL
        batch_size = x.size(0)
        label = torch.full((batch_size,), 1.0, dtype=torch.float, device=DEVICE)
        err_real = self.criterion(self.discriminator(x).view(-1),label)
        # DISCRIMINATOR ON FAKE
        z = torch.randn(batch_size, self.config.latent_vector_size, 1, 1, device=DEVICE)
        fake_label = torch.full((batch_size,), 0.0, dtype=torch.float, device=DEVICE)
        err_fake = self.criterion(self.discriminator(self.generator(z).detach()).view(-1),fake_label)

        errD = (err_real + err_fake )/2
        #
        return errD


    def generator_train(self, x):
        batch_size = x.size(0)
        z = torch.randn(batch_size, self.config.latent_vector_size, 1, 1, device=DEVICE)
        fake_label = torch.full((batch_size,), 0.0, dtype=torch.float, device=DEVICE)
        fake = self.generator(z)
        output = self.discriminator(fake).view(-1)
        errG = self.criterion(output, fake_label)
        #errG.backward()
        return errG
            # self.generator.zero_grad()
            # label.fill_(1.0)
            # fake = self.generator(z)
            # output = self.discriminator(fake).view(-1)
            # errG = self.criterion(output, label)
            # errG.backward()

            # self.generator.optimizer.step()
    def forward(self, x):
        x = x.to(DEVICE)
        # TRAIN DISCRIMINATOR
        self.discriminator.zero_grad()
        errD  = self.discriminator_train(x)
        error_disc = errD.clone()
        errD.backward()         
        self.discriminator.optimizer.step() 

        # TRAIN GENERATOR
        self.generator.zero_grad()
        errG = self.generator_train(x)
        error_gen = errG.clone()
        errG.backward()

        self.generator.optimizer.step()

        return error_disc.item(), error_gen.item()


        # if historical_averging:
        #     self.discriminator.zero_grad()
        #     z = torch.randn(batch_size, self.config.latent_vector_size, 1, 1, device=DEVICE)
        #     err_real = self.criterion(self.discriminator(real_img).view(-1),label)
        #     fake_label = torch.full((batch_size,), 0.0, dtype=torch.float, device=DEVICE)
        #     err_fake = self.criterion(self.discriminator(self.generator(z).detach()).view(-1),fake_label)
        #     errD = (err_real + err_fake )/2
        #     if len(self.discriminator.history) >=1:
                
        #         self.discriminator.history = torch.cat((self.discriminator.history, errD.unsqueeze_(0)),dim=0)
        #         print(f"jestem  {self.discriminator.history.shape}")
        #         errD = torch.mean(errD-torch.mean(self.discriminator.history,dim=0))
        #     else:
        #         self.discriminator.history = torch.tensor(errD,device=DEVICE)
        #         self.discriminator.history.unsqueeze_(0)
        #         print(self.discriminator.history)
        #         self.discriminator.history
        #     errD.backward()
        #     self.discriminator.optimizer.step() 
            
        # else:
            # self.discriminator.zero_grad()
            # output = self.discriminator(real_img).view(-1)
            # errD_real = self.criterion(output, label)
            # errD_real.backward()
            # #train disc on fake
            # z = torch.randn(batch_size, 100, 1, 1, device=DEVICE)
            # fake = self.generator(z)
            # label.fill_(0.0)

            # output = self.discriminator(fake.detach()).view(-1)

            # errD_fake = self.criterion(output, label)
            # errD_fake.backward()

            # errD = errD_real + errD_fake
            # self.discriminator.optimizer.step()
            
        ## train gen
        # if not features_matching:
        #     #print("stary loss")


        #     self.generator.optimizer.step()
        # else:
            #print("nowy loss")
            # batch_size = x.size(0)
            # z = torch.randn(batch_size, self.config.latent_vector_size, 1, 1, device=DEVICE)
            # fake = self.generator(z)
            # out_fake = self.discriminator(fake)
            # out_real = self.discriminator(x)
            # m1 = torch.mean(out_fake,dim=0)
            # m2 = torch.mean(out_real,dim=0)
            # errG = torch.mean(torch.abs(m2 - m1))
            # errG.backward()


    def save(self, folder_path, postfix):
        gen_name = f"generator_fs_{self.config.generator_features_number}_{postfix}.pth"
        disc_name = f"discriminator_fs_{self.config.discriminator_features_number}_{postfix}.pth"
        torch.save(self.generator.state_dict(), os.path.join(folder_path, gen_name))
        torch.save(self.discriminator.state_dict(), os.path.join(folder_path, disc_name))

    def load(self, folder_path, postfix):
        gen_name = f"generator_fs_{self.config.generator_features_number}_{postfix}.pth"
        disc_name = f"discriminator_fs_{self.config.discriminator_features_number}_{postfix}.pth"
        self.generator.load_state_dict(torch.load(os.path.join(folder_path,gen_name)))
        self.discriminator.load_state_dict(torch.load(os.path.join(folder_path,disc_name)))

    def generate_batch(self,threeshold=0.55, batch_size=16, batch_gen_size=64):
        images = []

        while(len(images) < batch_size):
            fixed_noise = torch.randn(batch_gen_size, self.config.latent_vector_size,1,1, device="cuda")
            with torch.no_grad():
                out = self.generator(fixed_noise)
                weights = self.discriminator(out)
                for i,o in enumerate(out):
                    if weights[i] > threeshold:
                        images.append((round(weights[i].item(),4),o.detach().cpu().numpy()))
                print(len(images))

        return images[:batch_size]
    
    
if __name__=='__main__':

    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    folder_name = "dcgan_refactor_test"
    start_epoch = 0
    dcgan = DCGAN(gan_cfg).to(DEVICE)
    #dcgan.load("../../models/dcgan/out/{folder_name}",f"{start_epoch}")
    tf = transforms.Compose(
        [
            transforms.Resize(gan_cfg.image_size),
            transforms.CenterCrop(gan_cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean    = [0.5, 0.5, 0.5],
                std     = [0.5, 0.5, 0.5])
        ]
    )

    path = "D:/AI_ML/Kaggle/Water Bodies Dataset_pruned_more/"
    dataset = WaterBodyGeneratorDataset(path, transform=tf)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_cfg.batch_size,
                                         shuffle=True, num_workers=0)

    fixed_noise = torch.randn(64, gan_cfg.latent_vector_size,1,1, device="cuda")

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    for epoch in range(train_cfg.number_of_epochs):
        loop = tqdm(dataloader)
        for data in loop:
            loss_d, loss_g = dcgan(data)
            G_losses.append(loss_g)
            D_losses.append(loss_d)
            loop.set_postfix(Loss_D=loss_d, Loss_G= loss_g, epoch=epoch+1)#, D_x=D_x, D_G_z1=D_G_z1, D_G_z2=D_G_z2
        #g_loss, d_loss = dcgan_train(netG, netD,criterion=criterion, optimizerD=optimizerD, optimizerG=optimizerG, dataloader=dataloader, epoch=epoch)
        if( (epoch +1)%10 == 0):
            dcgan.save(f"../trained_models/dcgan/out/{folder_name}",f"{start_epoch + epoch+1}")
            #img_list.append(dcgan.generate_batch(batch_size=6))
            # save_weights(netG, f"models/dcgan/out/old_model_20_batches_prunedmore_biggerimg/generator_size_{image_size}_fs_{ngf}_{epoch+1}.pth")
            # save_weights(netD, f"models/dcgan/out/old_model_20_batches_prunedmore_biggerimg/discriminator_size_{image_size}_fs_{ndf}_{epoch+1}.pth")
            # with torch.no_grad():
            #     out = netG(fixed_noise)
            #     #img_list.append(make_grid(out))

    #show_anim(img_list)
    show_loss(G_losses, D_losses)
    









    # dataroot = "D:/AI_ML/Kaggle/celeba/"
    # image_size = 64
    # dataset = dset.ImageFolder(root=dataroot,
    #                        transform=transforms.Compose([
    #                            transforms.Resize(image_size),
    #                            transforms.CenterCrop(image_size),
    #                            transforms.ToTensor(),
    #                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                        ]))     
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)    


    # criterion = nn.BCELoss()
