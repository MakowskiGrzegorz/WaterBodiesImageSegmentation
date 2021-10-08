from model import Unet
from dataset import WaterBodyDataset
import torch.optim as optim
import torch.nn as nn
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import *
from tqdm import tqdm
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
        
    #return running_loss/len(train_loader)

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

if __name__ == "__main__":
    unet = Unet(CHANNELS_IN, CHANNELS_OUT).to(DEVICE)
    #print(unet)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler()

    tf = A.Compose(
        [
            A.Resize(INPUT_SIZE[0], INPUT_SIZE[1]),
            A.Normalize(
             mean = [0.0,0.0,0.0],
             std=[1.0,1.0,1.0],
             max_pixel_value=255.0),
             ToTensorV2()   
        ]
    )
    data_root_path = "D:/AI_ML/Kaggle/Water Bodies Dataset/"
    water_dataset = WaterBodyDataset(data_root_path, tf)

    valid, train = random_split(water_dataset, [841, 2000], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=VALID_BATCH_SIZE, shuffle=True, num_workers=0)

    train_loss = []
    valid_loss = []
    for epoch in range(EPOCHS):
        loss = train_func(epoch=epoch, train_loader=train_loader, optimizer=optimizer, model=unet, criterion=criterion, scaler=scaler)
        train_loss.append(loss)
        valid = valid_func(epoch=epoch, valid_loader=valid_loader, model=unet, criterion=criterion)
        valid_loss.append(valid)

    torch.save(unet, "model.pth")


    