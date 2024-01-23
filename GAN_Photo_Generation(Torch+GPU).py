import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision.utils import save_image


# -------------------------------------------------------------------------
# 生成式神经网络
# [输入]
# latent_dim     Latent space中向量维数
# height,width,channels  图片的长款以及颜色通道数（黑白为1，RGB为3）
# Generator接受输入
# -------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, height, width, channels):
        super(Generator, self).__init__()
        # 基于输入尺寸预先计算Flatten后的尺寸
        self.init_size = (height // 4, width // 4)
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 256 * self.init_size[0] * self.init_size[1]))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256, 0.8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, channels, 7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, self.init_size[0], self.init_size[1])
        img = self.conv_blocks(out)
        return img

# -------------------------------------------------------------------------
# 判别式神经网络
# [输入]
# height,width,channels  图片的长宽以及颜色通道数（黑白为1，RGB为3）
# -------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, height, width, channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * (height // 4) * (width // 4), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

class GAN(nn.Module):
    def __init__(self, latent_dim, height, width, channels):
        super(GAN, self).__init__()
        self.generator = Generator(latent_dim, height, width, channels)
        self.discriminator = Discriminator(height, width, channels)
        self.discriminator.trainable = False

    def forward(self, x):
        generated_images = self.generator(x)
        gan_output = self.discriminator(generated_images)
        return gan_output

def load_image(path):
    image_files = os.listdir(path)
    images = []
    for file in image_files:  # 读取所有图片文件
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_width = 64
        new_height = 64
        resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        # 转换为CHW格式
        chw_format = np.transpose(resized_image, (2, 0, 1))
        images.append(chw_format)

    return images

def train(iterations,batch_size,load_path,save_path,latent_dim, height, width, channels):
    discriminator = Discriminator(height, width,channels)
    generator = Generator(latent_dim, height, width, channels)
    gan = GAN(latent_dim, height, width, channels)  # GAN类需要包含生成器和鉴别器

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #转移到gpu
    discriminator.to(device)
    generator.to(device)
    gan.to(device)

    # 创建优化器
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = torch.nn.BCELoss()

    train_set =load_image(load_path)  # 假设load_data()是加载数据的函数
    train_set = torch.tensor(train_set).float() / 255.0

    start = 0

    for step in range(iterations):
        # 生成假图片
        random_latent_vectors = torch.randn(batch_size, latent_dim).to(device)
        generated_images = generator(random_latent_vectors).to(device)

        # 真实图片
        real_images = train_set[start: start + batch_size].to(device)

        # 训练鉴别器
        optimizer_d.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        labels = torch.cat((real_labels, fake_labels), 0).to(device)
        combined_images = torch.cat((generated_images, real_images), 0).to(device)
        outputs = discriminator(combined_images)
        d_loss = criterion(outputs, labels)
        d_loss.backward()
        optimizer_d.step()

        # 训练生成器 (GAN)
        optimizer_g.zero_grad()
        misleading_labels = torch.ones(batch_size, 1).to(device)
        a_loss = criterion(gan(random_latent_vectors), misleading_labels)
        a_loss.backward()
        optimizer_g.step()

        # 更新起始索引
        start += batch_size
        if start > len(train_set) - batch_size:
            start = 0
        print(step, '/', iterations)
        # 打印损失和保存图片
        if step % 20 == 0:
            print(step,'/',iterations)
            print('discriminator loss:', d_loss.item())
            print('adversarial loss:', a_loss.item())
            save_image(generated_images[0], os.path.join(save_path, 'generated_avatar{}.png'.format(step)))
            save_image(real_images[0], os.path.join(save_path, 'real_avatar{}.png'.format(step)))

iterations=100000
batch_size=20
save_path='GAN_TEST'
load_path='Data/img_align_celeba/img_align_celeba'
latent_dim=32
height=64
width=64
channels=3

train(iterations,batch_size,load_path,save_path,latent_dim, height, width, channels)