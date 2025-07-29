import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, jit, random

import flax
from flax import linen as nn
from flax.training import train_state  # 方便管理训练状态
import optax

# -----------------------------------------------------------------------------
# 1. 数据加载部分
# -----------------------------------------------------------------------------

def numpy_collate(batch):
  """
  这是一个自定义的 collate_fn，它将 PyTorch tensors 转换为 NumPy arrays。
  这是连接 PyTorch 和 JAX 的关键桥梁。
  """
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

def get_dataloaders(batch_size=128):
    """创建并返回 MNIST 训练和测试的 DataLoader"""
    # 定义数据变换，将图片转换为 tensor 并进行归一化
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载并加载训练数据
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    # 下载并加载测试数据
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=numpy_collate, # 使用我们的自定义 collate 函数！
        drop_last=True, # JAX jit 编译要求输入形状固定，丢弃最后一个不完整的batch
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=numpy_collate,
        drop_last=True,
    )
    return train_loader, test_loader

# -----------------------------------------------------------------------------
# 2. 模型定义部分
# -----------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """一个简单的卷积神经网络"""
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # 展平
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x) # 10个类别
        return x

# -----------------------------------------------------------------------------
# 3. 训练和评估逻辑
# -----------------------------------------------------------------------------


class TrainState(train_state.TrainState):
    # 在这里添加额外的状态，比如 batch_stats
    pass

@jit
def train_step(state, batch):
    """单个训练步骤"""
    images, labels = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        return loss, logits

    # 计算损失和梯度
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # 更新模型状态（参数和优化器状态）
    state = state.apply_gradients(grads=grads)
    
    # 计算准确率
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    
    return state, loss, accuracy

@jit
def eval_step(state, batch):
    """单个评估步骤（无梯度计算）"""
    images, labels = batch
    logits = state.apply_fn({'params': state.params}, images)
    loss = optax.softmax_cross_entropy(logits=logits, labels=jax.nn.one_hot(labels, 10)).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, accuracy


def create_train_state(rng, learning_rate):
    """创建初始的 TrainState"""
    model = SimpleCNN()
    # JAX 要求输入形状是 (N, H, W, C)，而 PyTorch 是 (N, C, H, W)
    # 我们将在数据加载后调整它
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# -----------------------------------------------------------------------------
# 4. 主函数
# -----------------------------------------------------------------------------

def main():
    # 超参数
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    EPOCHS = 5

    # 获取 Dataloaders
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    # 创建初始状态
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, LEARNING_RATE)

    # 训练循环
    for epoch in range(1, EPOCHS + 1):
        # --- 训练 ---
        train_loss = 0.
        train_accuracy = 0.
        for batch_idx, (images, labels) in enumerate(train_loader):
            # JAX/Flax Conv2D 需要 (N, H, W, C) 格式
            # PyTorch DataLoader 输出 (N, C, H, W)
            # 所以我们需要转换一下维度
            images = np.transpose(images, (0, 2, 3, 1))
            
            state, loss, acc = train_step(state, (images, labels))
            train_loss += loss
            train_accuracy += acc

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_accuracy / len(train_loader)

        # --- 评估 ---
        test_loss = 0.
        test_accuracy = 0.
        for images, labels in test_loader:
            images = np.transpose(images, (0, 2, 3, 1))
            loss, acc = eval_step(state, (images, labels))
            test_loss += loss
            test_accuracy += acc
        
        avg_test_loss = test_loss / len(test_loader)
        avg_test_acc = test_accuracy / len(test_loader)

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc*100:.2f}% | "
            f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc*100:.2f}%"
        )


if __name__ == '__main__':
    main()