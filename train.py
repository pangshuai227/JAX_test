import torch
from torch.utils.data import DataLoader
import numpy as np
from einops import rearrange
import webdataset as wds
from torchvision import transforms

import jax
import jax.numpy as jnp
from jax import pmap

import flax
from flax.training import train_state
import optax
from flax.jax_utils import replicate, unreplicate

from model import VisionTransformer

# -----------------------------------------------------------------------------
# 1. 数据加载部分(WebDataset)
# -----------------------------------------------------------------------------
def get_dataloaders(batch_size=128):
    num_devices = jax.local_device_count()
    if batch_size % num_devices != 0:
        raise ValueError(f"Batch size ({batch_size}) must be divisible by the number of devices ({num_devices})")
    
    global_batch_size = batch_size
    per_device_batch_size = global_batch_size // num_devices

    # 定义数据变换
    def apply_transform(sample):
        image = sample["png"]
        # PyTorch 的 transform 需要 PIL Image 对象
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        sample["png"] = transform(image)
        sample["cls"] = int(sample["cls"])
        return sample

    # --- 训练集加载器 ---
    train_urls = "./mnist_wds_train/mnist-train-{000000..000009}.tar"
    train_dataset = wds.WebDataset(urls=train_urls, resampled=True).shuffle(1000).decode("pil").map(apply_transform).to_tuple("png", "cls").batched(per_device_batch_size)
    
    # --- 测试集加载器 ---
    test_urls = "./mnist_wds_test/mnist-test-{000000..000001}.tar"
    test_dataset = wds.WebDataset(urls=test_urls).decode("pil").map(apply_transform).to_tuple("png", "cls").batched(per_device_batch_size)

    return train_dataset, test_dataset

# -----------------------------------------------------------------------------
# 2. 训练和评估逻辑
# -----------------------------------------------------------------------------
class TrainStateWithEMA(train_state.TrainState):
    ema_params: flax.core.FrozenDict

def train_step(state, batch, ema_decay=0.999):
    images, labels = batch
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        return loss, logits
    
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_state = state.apply_gradients(grads=grads)
    
    new_ema_params = jax.tree_util.tree_map(
        lambda ema, p: ema * ema_decay + p * (1. - ema_decay),
        state.ema_params, new_state.params
    )
    final_state = new_state.replace(ema_params=new_ema_params)

    loss = jax.lax.pmean(loss, axis_name='batch')
    accuracy = jax.lax.pmean(jnp.mean(jnp.argmax(logits, -1) == labels), axis_name='batch')
    
    return final_state, loss, accuracy

def eval_step(state, batch):
    images, labels = batch
    logits = state.apply_fn({'params': state.ema_params}, images)
    loss = jax.lax.pmean(optax.softmax_cross_entropy(logits=logits, labels=jax.nn.one_hot(labels, 10)).mean(), axis_name='batch')
    accuracy = jax.lax.pmean(jnp.mean(jnp.argmax(logits, -1) == labels), axis_name='batch')
    return loss, accuracy

# -----------------------------------------------------------------------------
# 3. 初始化和主循环
# -----------------------------------------------------------------------------
def create_train_state(rng, learning_rate, image_size, patch_size):
    model = VisionTransformer(patch_size=patch_size)
    dummy_input = jnp.ones([1, image_size, image_size, 1])
    params = model.init(rng, dummy_input)['params']
    tx = optax.adam(learning_rate)
    return TrainStateWithEMA.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        ema_params=params
    )

def main():
    print(f"JAX backend: {jax.default_backend()}")
    num_devices = jax.local_device_count()
    print(f"Number of devices: {num_devices}")

    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    EPOCHS = 5
    IMAGE_SIZE = 28
    PATCH_SIZE = 4

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, LEARNING_RATE, IMAGE_SIZE, PATCH_SIZE)
    state = replicate(state)

    p_train_step = pmap(train_step, axis_name='batch')
    p_eval_step = pmap(eval_step, axis_name='batch')

    for epoch in range(1, EPOCHS + 1):
        # --- 训练 ---
        train_loss, train_accuracy = 0., 0.
        num_train_steps = 60000 // BATCH_SIZE
        for i, (images, labels) in enumerate(train_loader):
            if i >= num_train_steps: break
            images = rearrange(images.numpy(), '(d b) c h w -> d b h w c', d=num_devices)
            labels = rearrange(labels, '(d b) -> d b', d=num_devices)
            
            state, loss, acc = p_train_step(state, (images, labels))
            train_loss += loss[0]
            train_accuracy += acc[0]
        
        avg_train_loss = train_loss / num_train_steps
        avg_train_acc = train_accuracy / num_train_steps

        # --- 评估 ---
        test_loss, test_accuracy = 0., 0.
        num_test_steps = 10000 // BATCH_SIZE
        for i, (images, labels) in enumerate(test_loader):
            if i >= num_test_steps: break
            images = rearrange(images.numpy(), '(d b) c h w -> d b h w c', d=num_devices)
            labels = rearrange(labels, '(d b) -> d b', d=num_devices)
            
            loss, acc = p_eval_step(state, (images, labels))
            test_loss += loss[0]
            test_accuracy += acc[0]
            
        avg_test_loss = test_loss / num_test_steps
        avg_test_acc = test_accuracy / num_test_steps

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc*100:.2f}% | "
            f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc*100:.2f}%"
        )

if __name__ == '__main__':
    main()
