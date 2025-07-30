import jax
import jax.numpy as jnp
import flax.linen as nn

class PatchEmbedding(nn.Module):
    patch_size: int = 4
    embed_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            name='patch_conv'
        )(x)
        
        batch_size, height, width, channels = x.shape
        x = x.reshape(batch_size, height * width, channels)
        
        return x

class AttentionBlock(nn.Module):
    embed_dim: int
    num_heads: int
    
    @nn.compact
    def __call__(self, x):
        x_norm = nn.LayerNorm()(x)
        x_attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(x_norm, x_norm)
        x = x + x_attn
        
        x_norm = nn.LayerNorm()(x)
        x_mlp = nn.Dense(features=self.embed_dim * 4)(x_norm)
        x_mlp = nn.gelu(x_mlp)
        x_mlp = nn.Dense(features=self.embed_dim)(x_mlp)
        x = x + x_mlp
        
        return x

class VisionTransformer(nn.Module):
    patch_size: int = 4
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 6
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
        # 1. 图片转补丁嵌入
        x = PatchEmbedding(patch_size=self.patch_size, embed_dim=self.embed_dim)(x)
        
        # 2. 添加可学习的位置编码
        pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (1, x.shape[1], self.embed_dim)
        )
        x = x + pos_embedding
        
        # 3. 通过 Transformer 编码器层
        for _ in range(self.num_layers):
            x = AttentionBlock(embed_dim=self.embed_dim, num_heads=self.num_heads)(x)
            
        # 4. 分类头
        x = x.mean(axis=1)
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=self.num_classes)(x)
        
        return x
