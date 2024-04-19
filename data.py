import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# https://cs.stanford.edu/~acoates/stl10/
# DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
class STL10Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 图片数据集的根目录路径
            transform (callable, optional): 数据预处理操作
        """
        self.root_dir = root_dir
        self.transform = transform
        # 读取数据文件
        with open(os.path.join(root_dir, 'train_X.bin'), 'rb') as f:
            self.images = np.fromfile(f, dtype=np.uint8)
            self.images = np.reshape(self.images, (-1, 3, 96, 96))
            self.images = np.transpose(self.images, (0, 2, 3, 1))  # 调整维度顺序，从(C, H, W)到(H, W, C)
        with open(os.path.join(root_dir, 'train_y.bin'), 'rb') as f:
            self.labels = np.fromfile(f, dtype=np.uint8)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        # 将numpy数组转换为PIL图像
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 创建数据集实例
    dataset = STL10Dataset(root_dir='data/stl10_binary/', transform=transform)
    
    # 创建数据加载器实例
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 遍历数据加载器以获取数据
    for images, labels in dataloader:
        # 在这里执行你的训练或评估步骤
        # images 是一个形状为 (batch_size, channels, height, width) 的张量
        # labels 是一个形状为 (batch_size,) 的张量
        # 在这里执行你的训练或评估步骤
        pass
