import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torch
def show_image_mask(input, label, prediction):
    fig, axs = plt.subplots(1, 3, figsize=(20,20))

    axs[0].set_title("input")
    
    img = input.numpy()
    img = img.transpose(1,2,0)
    
    axs[0].imshow(img)
    axs[1].set_title("label")
    axs[1].imshow(label,cmap=plt.get_cmap('gray'))
    axs[2].set_title("prediction")
    axs[2].imshow(prediction,cmap=plt.get_cmap('gray'))




def show_img_row(data:dict):
    fig, axs = plt.subplots(1, len(data.keys()), figsize=(20,20))

    nr = 0
    for k,v in data.items():
        
        axs[nr].set_title(k)
        if(len(v.shape) >2):
            img = v.numpy().transpose(1,2,0)
            axs[nr].imshow(img)
        else:
            axs[nr].imshow(v,cmap=plt.get_cmap('gray'))
        nr+=1



def show_anim(data):
    matplotlib.rcParams['animation.embed_limit'] = 2**128
    fig = plt.figure(figsize=(10,10))
    plt.axis("off")
    
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in data]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())
    plt.show()

def show_loss(loss_1, loss_2):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Mean Batch Loss During Training")
    plt.plot(loss_1,label="G")
    plt.plot(loss_2,label="D")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def show_batch(data):
    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.imshow(np.transpose(data,(1,2,0)))


def save_weights(model, path):
    torch.save(model.state_dict(), path)

def load_weights( path):
    torch.load(path)