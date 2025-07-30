import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np

import jax
import jax.numpy as jnp
from jax import pmap
from jax.lib import xla_bridge

import flax
from flax.training import train_state
import optax
from flax.jax_utils import replicate, unreplicate


from model import VisionTransformer

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
    num_devices = jax.local_device_count()
    if batch_size % num_devices != 0:
        raise ValueError(f"Batch size ({batch_size}) must be divisible by the number of devices ({num_devices})")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate, drop_last=True)
    return train_loader, test_loader

# -----------------------------------------------------------------------------
# 2. 训练和评估逻辑
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

# -----------------------------------------------------------------------------
# 3. 初始化和主循环
# -----------------------------------------------------------------------------
def create_train_state(rng, learning_rate):
    # 实例化我们的 ViT 模型
    model = VisionTransformer()
    # 用一个假的输入来初始化模型参数
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def shard(data):
    return jax.tree_util.tree_map(lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), data)

def main():
    print(f"JAX backend: {xla_bridge.get_backend().platform}")
    print(f"Number of devices: {jax.local_device_count()}")

    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    EPOCHS = 5

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    # 用一个随机 key 来初始化模型
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, LEARNING_RATE)
    
    # 复制状态到所有设备
    state = replicate(state)

    # 创建并行版本的训练/评估函数
    p_train_step = pmap(train_step, axis_name='batch')
    p_eval_step = pmap(eval_step, axis_name='batch')

    for epoch in range(1, EPOCHS + 1):
        # --- 训练 ---
        train_loss, train_accuracy = 0., 0.
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = np.transpose(images, (0, 2, 3, 1))
            sharded_batch = shard((images, labels))
            state, loss, acc = p_train_step(state, sharded_batch)
            train_loss += loss[0]
            train_accuracy += acc[0]
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_accuracy / len(train_loader)

        # --- 评估 ---
        test_loss, test_accuracy = 0., 0.
        for images, labels in test_loader:
            images = np.transpose(images, (0, 2, 3, 1))
            sharded_batch = shard((images, labels))
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

if __name__ == '__main__':
    main()
