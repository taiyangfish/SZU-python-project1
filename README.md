# SZU Python 期末项目 1：OpenCLIP 图像编码器纯 NumPy 实现

## 📖 项目简介
本项目使用纯 NumPy（不依赖 PyTorch 等深度学习框架）完整复现了 OpenCLIP ViT-B/32 模型的图像编码器（Image Encoder）前向传播过程。代码实现了图像分块（Patchify）、多头自注意力机制、残差连接与 MLP 等核心逻辑，最终输出 512 维归一化图像特征向量。

## ⚠️ 重要提示：关于模型权重文件 (Weights)
由于 GitHub 免费版的单文件大小限制（100MB），作业原始附带的 `weights/image_encoder.npz`（约 300MB）未包含在此仓库中。
**复现前，请务必在根目录下创建 `weights/` 文件夹，并将原始提供的 `image_encoder.npz` 放入其中**，否则程序将因找不到模型权重而报错。

## ⚙️ 环境准备
确保你的 Python 版本在 3.9 或以上。在运行代码前，请先安装必要的依赖：
```bash
pip install -r requirements.txt
