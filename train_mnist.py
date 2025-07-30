import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, pmap
from jax.lib import xla_bridge # 用于获取设备信息

import flax
from flax import linen as nn
from flax.training import train_state
import optax
from flax.jax_utils import replicate, unreplicate

# -----------------------------------------------------------------------------
# 1. 数据加载部分
# -----------------------------------------------------------------------------

def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

def get_dataloaders(batch_size=128):
    # <<< 注意：为了使用 pmap，batch_size 必须是设备数量的整数倍
    num_devices = jax.local_device_count()
    if batch_size % num_devices != 0:
        raise ValueError(f"Batch size ({batch_size}) must be divisible by the number of devices ({num_devices})")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=numpy_collate,
        drop_last=True,
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
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

# -----------------------------------------------------------------------------
# 3. 训练和评估逻辑
# -----------------------------------------------------------------------------

class TrainState(train_state.TrainState):
    pass

def train_step(state, batch):
    images, labels = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    grads = jax.lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    
    loss = jax.lax.pmean(loss, axis_name='batch')
    accuracy = jax.lax.pmean(jnp.mean(jnp.argmax(logits, -1) == labels), axis_name='batch')
    
    return state, loss, accuracy

def eval_step(state, batch):
    images, labels = batch
    logits = state.apply_fn({'params': state.params}, images)
    
    loss = jax.lax.pmean(optax.softmax_cross_entropy(logits=logits, labels=jax.nn.one_hot(labels, 10)).mean(), axis_name='batch')
    accuracy = jax.lax.pmean(jnp.mean(jnp.argmax(logits, -1) == labels), axis_name='batch')
    return loss, accuracy


def create_train_state(rng, learning_rate):
    model = SimpleCNN()
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# <<< 新增：一个辅助函数，用于将数据分片到所有设备上
def shard(data):
    """将一个数据批次分片到所有可用的 JAX 设备上"""
    # jax.tree_util.tree_map 可以对嵌套结构（如元组）的每个叶子节点应用函数
    return jax.tree_util.tree_map(
        lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), data
    )

# -----------------------------------------------------------------------------
# 4. 主函数
# -----------------------------------------------------------------------------

def main():
    print(f"JAX backend: {xla_bridge.get_backend().platform}")
    print(f"Number of devices: {jax.local_device_count()}")

    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    EPOCHS = 5

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, LEARNING_RATE)
    
    state = replicate(state)

    # axis_name='batch' 使得我们可以在函数内部使用 pmean
    p_train_step = pmap(train_step, axis_name='batch')
    p_eval_step = pmap(eval_step, axis_name='batch')

    for epoch in range(1, EPOCHS + 1):
        # --- 训练 ---
        train_loss = 0.
        train_accuracy = 0.
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = np.transpose(images, (0, 2, 3, 1))
            
            sharded_batch = shard((images, labels))
            
            # state, loss, acc 都是被复制到所有设备上的
            state, loss, acc = p_train_step(state, sharded_batch)
            
            train_loss += loss[0] # 从设备上取回数值（所有设备上的值都一样）
            train_accuracy += acc[0]

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_accuracy / len(train_loader)

        # --- 评估 ---
        test_loss = 0.
        test_accuracy = 0.
        for images, labels in test_loader:
            images = np.transpose(images, (0, 2, 3, 1))
            sharded_batch = shard((images, labels))
            
            # 调用 pmap 版本的函数
            loss, acc = p_eval_step(state, sharded_batch)
            
            test_loss += loss[0]
            test_accuracy += acc[0]
        
        avg_test_loss = test_loss / len(test_loader)
        avg_test_acc = test_accuracy / len(test_loader)

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc*100:.2f}% | "
            f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc*100:.2f}%"
        )
    
    # 训练结束后，如果需要用单个模型参数做其他事，可以把它从设备上取回
    # final_params = unreplicate(state.params)

if __name__ == '__main__':
    main()