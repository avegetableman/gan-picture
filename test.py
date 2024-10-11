from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
class CenterCropSquare:
    def __call__(self, img):
        width, height = img.size
        new_size = min(width, height)
        img = img.crop((0, 0, new_size, new_size)) 
        return img

#path=r"F:/6/AI/dataset/op/"
path=r"G:/6/AI/dataset/op/"
img_path=r"./picture/"
trans=transforms.Compose([CenterCropSquare(),transforms.Resize([130,130]),transforms.ToTensor(),transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
#data=Anime_dataset(path,transform=trans)
data=ImageFolder(path,transform=trans)
print(data[0][0].shape)
un = transforms.ToPILImage()
un(data[5][0]*0.5+0.5).show()