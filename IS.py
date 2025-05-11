import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

class ImageFolderDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def inception_score(dataloader, cuda=True, batch_size=32, splits=10):
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    if cuda:
        inception_model.cuda()

    def get_pred(x):
        with torch.no_grad():
            x = inception_model(x)
            return F.softmax(x, dim=1).cpu().numpy()
        
    preds = []
    for batch in dataloader:
        if cuda:
            batch = batch.cuda()
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        pred = get_pred(batch)
        preds.append(pred)
    preds = np.concatenate(preds, axis=0)

    scores = []
    N = preds.shape[0]
    for k in range(splits):
        part = preds[k * N // splits: (k+1) * N // splits, :]
        py = np.mean(part, axis=0)
        scores.append(np.exp(np.mean([kl(p, py) for p in part])))
    return np.mean(scores), np.std(scores)

def kl(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    return np.sum(p * (np.log(p + 1e-10) - np.log(q + 1e-10)))

if __name__ == '__main__':
    # 设 "generated_images" 文件夹里是你要评测的生成图像
    img_folder = '/home/czar/ML/GAN/HGGAN/results/HGGAN/test_latest/images/synthesized_image'

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = ImageFolderDataset(img_folder, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    is_mean, is_std = inception_score(dataloader, cuda=torch.cuda.is_available(), splits=10)
    print(f'Inception Score: {is_mean} ± {is_std}')