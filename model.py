import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import save_image

# ハイパーパラメータの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
image_size = 1024
batch_size = 16
epochs = 100


# Generatorの定義
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        return img


# Discriminatorの定義
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


# モデルの初期化
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# 損失関数と最適化アルゴリズムの定義
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# データセットの読み込みと前処理
transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
dataset = ImageFolder("path_to_dataset_folder", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 学習のループ
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # 本物の画像データのラベルを作成
        valid = torch.ones(imgs.size(0), 1).to(device)
        # 生成された画像のラベルを作成
        fake = torch.zeros(imgs.size(0), 1).to(device)

        # 生成器の学習
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim, 1, 1).to(device)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # ディスクリミネータの学習
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(imgs.to(device)), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # 学習の進捗の表示
        batches_done = epoch * len(dataloader) + i
        if batches_done % 50 == 0:
            print(
                f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
            )

    # テストデータに対して生成器を適用し、画像を保存
    if epoch % 10 == 0:
        z = torch.randn(10, latent_dim, 1, 1).to(device)
        gen_imgs = generator(z)
        save_image(
            gen_imgs.data, f"output_images/epoch_{epoch}.fits", nrow=10, normalize=True
        )

print("学習が完了しました。")
