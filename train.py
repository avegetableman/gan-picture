from pbq import *
from net import *
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torch.autograd as autograd
from PIL import Image

epochs=300
batch=128
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gamma=10

class CenterCropSquare:
    def __call__(self, img):
        width, height = img.size
        new_size = min(width, height)
        img = img.crop((0, 0, new_size, new_size)) 
        return img

#path=r"F:/6/AI/dataset/op/"
path=r"F:/6/AI/dataset/op/"
img_path=r"./picture/"
trans=transforms.Compose([CenterCropSquare(),transforms.Resize([130,130]),transforms.ToTensor()])
#data=Anime_dataset(path,transform=trans)
data=ImageFolder(path,transform=trans)
train_data=DataLoader(dataset=data,batch_size=batch,shuffle=True,drop_last=True)

def gradient_penalty(discriminator, real_data, fake_data, device="cpu"):
    batch_size = real_data.size(0)
    
    # 在[0, 1]之间随机插值真实数据和生成数据
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated = interpolated.requires_grad_(True)

    # 通过判别器获得插值样本的判别值
    interpolated_logits = discriminator(interpolated)

    # 计算判别值对插值样本的梯度
    gradients = autograd.grad(
        outputs=interpolated_logits,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_logits, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # 计算梯度的范数 ||∇D(interpolated)||2
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    # 梯度惩罚项 (||∇D(interpolated)||2 - 1)^2
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty
#pb=torch.load("pbq.pt")
#sc=torch.load("gan.pt")
if __name__ == "__main__":
    sc=Net_with_attention().to(device)
    pb=pbq_with_attention().to(device)
    #pb=torch.load("pbq.pt")
    #sc=torch.load("gan.pt")
    scop=torch.optim.Adam(sc.parameters(),lr=0.0002)
    pbop=torch.optim.Adam(pb.parameters(),lr=0.0002)
    gloss=[]
    dloss=[]
    for epoch in tqdm(range(epochs),desc="training:",unit="epochs"):
        tmp_gloss=0
        tmp_dloss=0
        for i,(real_img,_) in enumerate(train_data):
            seeds=torch.randn(batch,500,1,1).to(device)
            fake_img=sc(seeds).to(device)
            fake=pb(fake_img.detach().to(device)).to(device)
            real=pb(real_img.to(device)).to(device)
            gp = gradient_penalty(pb, real_img, fake_img, device)
            d_loss = -torch.mean(real.view(batch,-1)) + torch.mean(fake.view(batch,-1))+gamma*gp
            tmp_dloss+=d_loss.item()
            pbop.zero_grad()
            d_loss.backward(retain_graph=True)
            pbop.step()

            g_img=sc(seeds).to(device)
            g_loss= -torch.mean(pb(g_img.to(device)).view(batch,-1).to(device))
            tmp_gloss+=g_loss.item()
            scop.zero_grad()
            g_loss.backward()
            scop.step()
        gloss.append(tmp_gloss/(i+1))
        dloss.append(tmp_dloss/(i+1))
        #测试图集
        if (epoch % 30 == 0):
            plt.imshow(g_img[0].detach().cpu().permute(1,2,0).numpy())
            plt.savefig(img_path+str(epoch)+"epoch.jpg")
            plt.cla()
            plt.plot(range(len(gloss)),gloss) 
            plt.plot(range(len(dloss)),dloss) 
            plt.savefig("loss.jpg")
            plt.cla()
            torch.save(sc,"checkpoint_gan.pt")
            torch.save(pb,"checkpoint_pbq.pt")
    torch.save(sc,"gan.pt")
    torch.save(pb,"pbq.pt")
    plt.cla()
    plt.plot(range(epochs),gloss) 
    plt.plot(range(epochs),dloss) 
    plt.savefig("loss.jpg")
