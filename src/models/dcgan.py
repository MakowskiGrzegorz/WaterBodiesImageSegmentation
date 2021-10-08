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
from train import FIXED_NOISE, dcgan_train, dcganNew_train
from utils import save_weights, show_anim, show_loss
#from inference import make_grid
#### GLOBALS
nz= 100
ngf = 128
ndf = 64
nc = 3
lr = 2e-4
beta1 = 0.5
num_epochs = 30
batch_size = 10
image_size = 64
GENERATOR_FEATURES = [ 8, 4, 2]
DISCRIMINATOR_FEATURES = [1, 2, 4, 8] 
def weights_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class GeneratorBlock(nn.Module):
    """Some Information about GeneratorBlock"""
    def __init__(self, features_in, features_out, kernel_size, stride, padding):
        super(GeneratorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(features_in, features_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(features_out),
            nn.ReLU(True)
        )
    def forward(self, x):
        x = self.block(x)
        return x

class Generator(nn.Module):
    """Some Information about Generator"""
    def __init__(self, latent_vector_size, channels_out,feature_scaler):
        super(Generator, self).__init__()

        self.feature_scaler = GENERATOR_FEATURES
        self.input = GeneratorBlock(latent_vector_size, feature_scaler * self.feature_scaler[0], kernel_size=4, stride=1, padding=0)
        self.main = nn.ModuleList()
        self.main += [GeneratorBlock(feature_scaler * self.feature_scaler[i], feature_scaler * self.feature_scaler[i+1],kernel_size=4, stride=2, padding=1) for i in range(len(self.feature_scaler) -1)]
        self.main.append(GeneratorBlock(feature_scaler * self.feature_scaler[-1], feature_scaler, kernel_size=4, stride=2, padding=1))
        
        self.last = nn.Sequential(
            nn.ConvTranspose2d(feature_scaler, channels_out, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def forward(self, x):
        #print(x.shape)
        x = self.input(x)
        #print(x.shape)
        for block in self.main:
            x = block(x)
            #print(x.shape)
        
        x = self.last(x)
        #print(x.shape)
        return x


class GeneratorNew(nn.Module):
    """Some Information about GeneratorNew"""
    def __init__(self):
        super(GeneratorNew, self).__init__()
        self.init_size = image_size // 4

        self.input = nn.Sequential(nn.Linear(nz, 128 * self.init_size **2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.input(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(x)
        return img


class DiscriminatorBlockNew(nn.Module):
    """Some Information about DiscriminatorBlockNew"""
    def __init__(self, features_in, features_out, kernel_size=3, stride=2, padding=1, bn=True):
        super(DiscriminatorBlockNew, self).__init__()
            
        self.block = [nn.Conv2d(features_in, features_out, kernel_size, stride, padding),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Dropout2d(0.25),]
        if bn:
            self.block.append(nn.BatchNorm2d(features_out, 0.8))
        
        self.layer = nn.Sequential(*(self.block(features_in, features_out, kernel_size, stride, padding,bn)))
        
    def forward(self, x):

        x = self.layer(x)
        return x

class DiscriminatorNew(nn.Module):
    """Some Information about DiscriminatorNew"""
    def __init__(self):
        super(DiscriminatorNew, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = image_size //2** 4

        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        validity = self.adv_layer(x)
        return validity

class DiscriminatorBlock(nn.Module):
    """Some Information about DiscriminatorBlock"""
    def __init__(self, features_in, features_out, kernel_size, stride, padding):
        super(DiscriminatorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features_in, features_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(features_out),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Discriminator(nn.Module):
    """Some Information about Discriminator"""
    def __init__(self, input_size, feature_scaler):
        super(Discriminator, self).__init__()
        self.feature_scaler = DISCRIMINATOR_FEATURES

        self.input = nn.Sequential(
            nn.Conv2d(input_size, feature_scaler * self.feature_scaler[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.main = nn.ModuleList()
        self.main += [DiscriminatorBlock(feature_scaler * self.feature_scaler[i], feature_scaler * self.feature_scaler[i+1],kernel_size=4, stride=2, padding=1) for i in range(len(self.feature_scaler)-1)]

        self.last = nn.Sequential(
            nn.Conv2d(feature_scaler * self.feature_scaler[-1], 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))
        self.history = torch.Tensor()
    def forward(self, x,feature=False):
        x = self.input(x)
        for block in self.main:
            x = block(x)
        if feature:
            return x
        x = self.last(x)
        return x


DEVICE = "cuda"
class DCGAN(nn.Module):
    """Some Information about DCGAN"""
    def __init__(self,latent_vector_size=100,channels_out=3, disc_feature=64, gen_feature=128):
        super(DCGAN, self).__init__()
        self.disc_feature_scaler = disc_feature
        self.gen_feature_scaler = gen_feature
        self.latent_vector_size = latent_vector_size
        self.generator = Generator(latent_vector_size=latent_vector_size,channels_out=channels_out,feature_scaler=self.gen_feature_scaler).to(DEVICE)
        self.generator.apply(weights_init)
        self.discriminator = Discriminator(channels_out,self.disc_feature_scaler).to(DEVICE)
        self.discriminator.apply(weights_init)

        self.criterion = nn.BCELoss()

    def forward(self, x, features_matching=False, historical_averging=False):
        #train disc on real
        real_img = x.to(DEVICE)
        batch_size = real_img.size(0)
        label = torch.full((batch_size,), 1.0, dtype=torch.float, device=DEVICE)
        ## train disc
        if historical_averging:
            self.discriminator.zero_grad()
            z = torch.randn(batch_size, 100, 1, 1, device=DEVICE)
            err_real = self.criterion(self.discriminator(real_img).view(-1),label)
            fake_label = torch.full((batch_size,), 0.0, dtype=torch.float, device=DEVICE)
            err_fake = self.criterion(self.discriminator(self.generator(z).detach()).view(-1),fake_label)
            errD = (err_real + err_fake )/2
            if len(self.discriminator.history) >=1:
                
                self.discriminator.history = torch.cat((self.discriminator.history, errD.unsqueeze_(0)),dim=0)
                print(f"jestem  {self.discriminator.history.shape}")
                errD = torch.mean(errD-torch.mean(self.discriminator.history,dim=0))
            else:
                self.discriminator.history = torch.tensor(errD,device=DEVICE)
                self.discriminator.history.unsqueeze_(0)
                print(self.discriminator.history)
                self.discriminator.history
            errD.backward()
            self.discriminator.optimizer.step()
        
            
            
        else:
            self.discriminator.zero_grad()
            z = torch.randn(batch_size, 100, 1, 1, device=DEVICE)
            err_real = self.criterion(self.discriminator(real_img).view(-1),label)
            fake_label = torch.full((batch_size,), 0.0, dtype=torch.float, device=DEVICE)
            err_fake = self.criterion(self.discriminator(self.generator(z).detach()).view(-1),fake_label)
            errD = (err_real + err_fake )/2
            errD.backward()
            self.discriminator.optimizer.step()
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
        if not features_matching:
            #print("stary loss")
            self.generator.zero_grad()
            label.fill_(1.0)
            output = self.discriminator(fake).view(-1)
            errG = self.criterion(output, label)
            errG.backward()

            self.generator.optimizer.step()
        else:
            #print("nowy loss")
            fake = self.generator(z)
            self.generator.zero_grad()
            out_fake = self.discriminator(fake, features_matching)
            out_real = self.discriminator(real_img, features_matching)
            m1 = torch.mean(out_fake,dim=0)
            m2 = torch.mean(out_real,dim=0)
            errG = torch.mean(torch.abs(m2 - m1))
            errG.backward()
            self.generator.optimizer.step()
        return errD.item(),errG.item()

    def save(self, folder_path, postfix):
        gen_name = f"generator_fs_{self.gen_feature_scaler}_{postfix}.pth"
        disc_name = f"discriminator_fs_{self.disc_feature_scaler}_{postfix}.pth"
        torch.save(self.generator.state_dict(), os.path.join(folder_path, gen_name))
        torch.save(self.discriminator.state_dict(), os.path.join(folder_path, disc_name))

    def load(self, folder_path, postfix):
        gen_name = f"generator_fs_{self.gen_feature_scaler}_{postfix}.pth"
        disc_name = f"discriminator_fs_{self.disc_feature_scaler}_{postfix}.pth"
        self.generator.load_state_dict(torch.load(os.path.join(folder_path,gen_name)))
        self.discriminator.load_state_dict(torch.load(os.path.join(folder_path,disc_name)))

    def generate_batch(self,threeshold=0.55, batch_size=16, batch_gen_size=64):
        images = []

        while(len(images) < batch_size):
            fixed_noise = torch.randn(batch_gen_size, self.latent_vector_size,1,1, device="cuda")
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
    folder_name = "dcgan_FM_HA_Test"
    start_epoch = 0
    dcgan = DCGAN()
    #dcgan.load("../../models/dcgan/out/{folder_name}",f"{start_epoch}")
    tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean    = [0.5, 0.5, 0.5],
                std     = [0.5, 0.5, 0.5])
        ]
    )

    path = "D:/AI_ML/Kaggle/Water Bodies Dataset_pruned_more/"
    dataset = WaterBodyGeneratorDataset(path, transform=tf)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=0)

    fixed_noise = torch.randn(64, nz,1,1, device="cuda")

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    device= "cuda"
    for epoch in range(num_epochs):
        loop = tqdm(dataloader)
        for data in loop:
            loss_d, loss_g = dcgan(data,features_matching=True,historical_averging=True)
            G_losses.append(loss_g)
            D_losses.append(loss_d)
            loop.set_postfix(Loss_D=loss_d, Loss_G= loss_g, epoch=epoch+1)#, D_x=D_x, D_G_z1=D_G_z1, D_G_z2=D_G_z2
        #g_loss, d_loss = dcgan_train(netG, netD,criterion=criterion, optimizerD=optimizerD, optimizerG=optimizerG, dataloader=dataloader, epoch=epoch)
        if( (epoch +1)%10 == 0):
            dcgan.save(f"../../models/dcgan/out/{folder_name}",f"{start_epoch + epoch+1}")
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
