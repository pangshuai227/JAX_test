import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange, repeat

class FeedForward(nn.Module):
    embed_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim, name="dense1")(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.embed_dim, name="dense2")(x)
        return x

class Attention(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        # x 的形状: (批次大小, 序列长度, 嵌入维度)
        batch, n_patches, _ = x.shape
        head_dim = self.embed_dim // self.num_heads
        
        # 1. 创建 Query, Key, Value
        qkv = nn.Dense(features=self.embed_dim * 3, name="qkv_dense")(x)
        
        # 2. 用 einops 将 Q, K, V 分离并重排成多头形式
        q, k, v = rearrange(qkv, 'b n (h d qkv) -> qkv b h n d', h=self.num_heads, qkv=3)

        # 3. 计算注意力分数 (scaled dot-product attention)
        dots = jnp.einsum('b h n d, b h m d -> b h n m', q, k) / jnp.sqrt(head_dim)
        attn_weights = nn.softmax(dots, axis=-1)

        # 4. 用注意力分数对 Value 进行加权求和
        output = jnp.einsum('b h n m, b h m d -> b h n d', attn_weights, v)

        # 5. 合并多头，恢复原始形状
        output = rearrange(output, 'b h n d -> b n (h d)')
        
        # 6. 最终的线性投射层
        output = nn.Dense(features=self.embed_dim, name="out_dense")(output)
        
        return output

class VisionTransformer(nn.Module):
    patch_size: int = 4
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 6
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            name='patch_embedding'
        )(x)
        
        x = rearrange(x, 'b h w c -> b (h w) c')
        
        cls_token = self.param('cls', nn.initializers.zeros, (1, 1, self.embed_dim))
        cls_token = repeat(cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = jnp.concatenate([cls_token, x], axis=1)
        
        pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (1, x.shape[1], self.embed_dim)
        )
        x = x + pos_embedding
        
        for i in range(self.num_layers):
            # 残差连接 1: 注意力模块
            x_norm = nn.LayerNorm(name=f"ln1_{i}")(x)
            x = x + Attention(embed_dim=self.embed_dim, num_heads=self.num_heads, name=f"attn_{i}")(x_norm)
            
            # 残差连接 2: FFN模块
            x_norm = nn.LayerNorm(name=f"ln2_{i}")(x)
            x = x + FeedForward(embed_dim=self.embed_dim, hidden_dim=self.embed_dim * 4, name=f"ffn_{i}")(x_norm)
            
        cls_output = x[:, 0]
        cls_output = nn.LayerNorm(name="ln_final")(cls_output)
        logits = nn.Dense(features=self.num_classes, name="classification_head")(cls_output)
        
        return logits
