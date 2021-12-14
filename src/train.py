from torch._C import device
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import show_image_mask, show_img_row

import numpy as np
DEVICE = "cuda"
EPOCHS = 6
CHANNELS_IN = 3
CHANNELS_OUT = 1
LEARNING_RATE = 1e-5
INPUT_SIZE = (240,240)
TRAIN_BATCH_SIZE = 50
VALID_BATCH_SIZE = 29

def train_func(epoch, train_loader, optimizer, model, criterion, scaler):
    
    loop = tqdm(train_loader)
    running_loss = 0.0
    for batch_idx, (data, mask) in enumerate(loop):
        input = data.to(DEVICE)
        mask = mask.to(DEVICE).unsqueeze(1)

        with torch.cuda.amp.autocast():
            predictions = model(input)
            loss = criterion(predictions, mask)

        running_loss += loss.item()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        loop.set_postfix(train_loss=running_loss/(batch_idx+1),intermidiate_train_loss=loss.item(), epoch=epoch)


def valid_func(epoch, valid_loader, model, criterion):
    valid_loss = 0.0
    loop = tqdm(valid_loader)
    model.eval()
    for batch_idx, (data, mask) in enumerate(loop):
        input = data.to(DEVICE)
        mask = mask.to(DEVICE).unsqueeze(1)
        with torch.no_grad():
            predictions = model(input)
            loss = criterion(mask, predictions)
        valid_loss  += loss.item()
        loop.set_postfix(valid_loss=valid_loss/(batch_idx+1),epoch=epoch)
        
    model.train()
    #return valid_loss/len(valid_loader)



def compare_models(data, models):
    
    for img, mask in zip(data[0], data[1]):
        dict = {"input":img, "label":mask}
        for (i,model) in enumerate(models):
            pred = model(img.unsqueeze(0).to("cuda"))
            pred_scaled = torch.where(pred > 0.5, 1, 0)
            dict[f"model_{i}"]=pred.detach().squeeze(0).squeeze(0).to("cpu")
            dict[f"model_scaled{i}"]=pred_scaled.detach().squeeze(0).squeeze(0).to("cpu")
        show_img_row(dict)

FIXED_NOISE = torch.randn(64, 100, 1, 1, device=DEVICE)
Tensor = torch.cuda.FloatTensor# if cuda else torch.FloatTensor
from torch.autograd import Variable
def dcgan_train(generator:nn.Module, discriminator:nn.Module, criterion, optimizerD, optimizerG, dataloader:torch.utils.data.DataLoader, epoch):
    loop = tqdm(dataloader)
    gen_loss = 0.0
    disc_loss = 0.0
    gen_loss_asd = 0
    for batch_idx, data in enumerate(loop):
                # # Prepare real data
        real_img = data.to(DEVICE)
        batch_size = real_img.size(0)
        label = torch.full((batch_size,), 1.0, dtype=torch.float, device=DEVICE)
        

        ##### TRAIN DISCRIMINATOR ON REAL IMAGES
        discriminator.zero_grad()
        output = discriminator(real_img).view(-1)

        errD_real = criterion(output, label)
        errD_real.backward()

        # To print later
        D_x = output.mean().item()

        ##### TRAIN DISCRIMINATOR ON FAKE IMAGES
        noise = torch.randn(batch_size, 100, 1, 1, device=DEVICE)

        fake = generator(noise)
        label.fill_(0.0)

        output = discriminator(fake.detach()).view(-1)

        errD_fake = criterion(output, label)
        errD_fake.backward()

        # To print later
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        optimizerD.step()

        ##### TRAIN GENERATOR
        generator.zero_grad()
        label.fill_(1.0)
        output = discriminator(fake).view(-1)

        errG = criterion(output, label)
        errG.backward()

        optimizerG.step()

        # To print later
        D_G_z2 = output.mean().item()
        loop.set_postfix(Loss_D=errD.item(), Loss_G= errG.item(), epoch=epoch+1)#, real_loss = real_loss.item(), fake_loss =fake_loss.item())#, D_x=D_x, D_G_z1=D_G_z1, D_G_z2=D_G_z2

    return (D_G_z1, D_x)


        # #### Verbose
        # #if batch_idx % 50 == 0:
    #print(f"Loss_D: {errD.item()} Loss_G: {errG.item()}   D(x): {D_x}   D(G(z)): {D_G_z1}   |    {D_G_z2}")
        

def dcganNew_train(generator:nn.Module, discriminator:nn.Module, criterion, optimizerD, optimizerG, dataloader:torch.utils.data.DataLoader, epoch):
    loop = tqdm(dataloader)
    gen_loss = 0.0
    disc_loss = 0.0
    gen_loss_asd = 0
    for batch_idx, data in enumerate(loop):
        valid = Variable(Tensor(data.shape[0], 1).fill_(1.0),requires_grad=False)
                
        fake = Variable(Tensor(data.shape[0], 1).fill_(0.0),requires_grad=False)

        real_img = data.to(DEVICE)

        batch_size = real_img.size(0)
        z = torch.randn(real_img.size(0), 100, device=DEVICE)#Variable(Tensor(np.random.normal(0, 1, (data.shape[0], 100))))
        #noise = torch.randn(batch_size, 100, 1, 1, device=DEVICE)
        gen_imgs = generator(z)
        
        #if batch_idx % 3 == 0:
        optimizerG.zero_grad()
        
        errG = criterion(discriminator(gen_imgs), valid)

        errG.backward()
        optimizerG.step()
        gen_loss += errG.item()
        gen_loss_asd +=1

        optimizerD.zero_grad()
        real_loss = criterion(discriminator(real_img), valid)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        errD = (real_loss + fake_loss) / 2

        errD.backward()
        optimizerD.step()

        disc_loss += errD.item()
        loop.set_postfix(Loss_D=errD.item(), Loss_G= errG.item(), epoch=epoch+1, real_loss = real_loss.item(), fake_loss =fake_loss.item())#, D_x=D_x, D_G_z1=D_G_z1, D_G_z2=D_G_z2
    return (gen_loss/gen_loss_asd, disc_loss/len(loop))



