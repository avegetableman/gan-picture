import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import torch
from PIL import Image

trans = transforms.Compose([
    transforms.Resize([130, 130]),
    transforms.ToTensor(),
    transforms.Normalize(mean=-1, std=2)
])

s = torch.load("gan.pt", map_location="cuda:0").cuda()

# 正常化
normalize = transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2))  # 这里需要调整mean和std为正确的格式

for i in range(100):
    seed = torch.randn(1, 256, 1, 1).cuda()
    out = s(seed).cuda()
    
    # 反向标准化并转换为 PIL 格式
    img = normalize(out[0]).permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).clip(0, 255).astype('uint8')  # 转换为 0-255 范围的整数
    pil_img = Image.fromarray(img)

    # 保存为 JPEG 文件
    pil_img.save(r"./p1_with_atten/"+str(i)+"epoch.jpg")
