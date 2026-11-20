"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector
#q(Query)要找什么
#k(Key)是什么标签
#v(Value)实际内容

class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        # 【精度修复】将加载的权重统一转为 float32，防止后续矩阵乘法被污染
        self.qkv_weight = weights[prefix + '.in_proj_weight'].astype(np.float32)  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias'].astype(np.float32)      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight'].astype(np.float32)  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias'].astype(np.float32)
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        # TODO-02 contract: [B,N,D] to [B,H,N,D/H].
        # Image model: D=768, H=12, D/H=64.
        B,N,D=x.shape
        x=x.reshape(B,N,self.heads,self.head_dim)
        return x.transpose(0,2,1,3)
    def __call__(self, x, causal=False):
        x = x.astype(np.float32) # 【精度修复】
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        #q,k,v分离，让三个部分各自表达
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        scale=np.float32(np.sqrt(self.head_dim)) # 【精度修复】
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].
        #q,k点积，计算相关性(方向一致性)，scores就是相关度
        #然后进行缩放，因为点积后每项量级都很大，将分数整体压小，防止softmax过于极端造成one-hot
        #而缩放所除的就是q,k点积的标准差，可以把他的标准差拉回1
        #softmax的尖锐程度由差值决定(尖锐程度代表注意力分布的集中程度)
        scores=q@k.transpose(0,1,3,2)/scale
        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        #softmax
        #每个token对其他token的注意力权重
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # context: [B,H,N,64], merged: [B,N,D].
        #聚合v,把所有人的v按权重加起来，这一步可实现token融合全序列信息
        context=coefficients@v
        #合并头：将所有特征维度拼回到一起
        #合并前：(batch，头，token轴，每个头的特征维度)
        context=context.transpose(0,2,1,3)
        B,H,N,_=coefficients.shape
        merged=context.reshape(B,N,self.width)
        # TODO-03c: output projection, using out_weight and out_bias.
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        #Y = XWᵀ + b，out_weight 存为 [out,in] 所以要 .T 变成 [in,out] 才能和输入的内维对齐
        #out_weight[in,out]输出第d维对输入第k维的贡献
        #加权和+偏置
        output=merged@self.out_weight.T+self.out_bias
        return output.astype(np.float32) # 【精度修复】

class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        # 【精度修复】权转 float32
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight'].astype(np.float32)  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias'].astype(np.float32)
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight'].astype(np.float32)  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias'].astype(np.float32)
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        x = x.astype(np.float32) # 【精度修复】
        # Given: normalize each token and calculate its attention update.
        #norm:把每个token拉到均值0，方差1的同一尺度
        update = self.attention(self.norm1(x).astype(np.float32), causal=causal) # 【精度修复】

        # TODO-04a: first residual addition. Name the result residual.
        residual=x+update
        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual).astype(np.float32) # 【精度修复】

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].
        hidden=u@self.fc1_weight.T+self.fc1_bias
        #GELU激活
        hidden=self.gelu(hidden).astype(np.float32) # 【精度修复】
        mlp_update=hidden@self.fc2_weight.T+self.fc2_bias
        
        # 【精度修复】出口强转
        return (residual+mlp_update).astype(np.float32)

class ImageEncoder:
    def __init__(self, weights):
        # 【精度修复】权转 float32
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072).astype(np.float32)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768).astype(np.float32)
        self.position = weights['visual.positional_embedding'].astype(np.float32)  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj'].astype(np.float32)  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        # TODO-01: separate patch-grid axes; reorder; flatten.
        # Input [B,3,224,224], return [B,49,3072].
        B=images.shape[0]
        patch_size=32
        grid_size=224//patch_size
        #224=7*32,一张图切成7*7=49个patch
        #每个patch是3*32*32
        x=images.reshape(B,3,grid_size,patch_size,grid_size,patch_size)
        x=x.transpose(0,2,4,1,3,5)
        return x.reshape(B,grid_size*grid_size,3*patch_size*patch_size).astype(np.float32) # 【精度修复】
    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        patches=patches@self.patch_weight.T
        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        # Return [B,50,768]. self.position broadcasts over B.
        x=np.concatenate([cls,patches],axis=1)
        x=x+self.position
        return x.astype(np.float32) # 【精度修复】
    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images)).astype(np.float32) # 【精度修复】
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # Select CLS, apply post_norm, and project with self.projection.
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already [in,out].
        # Name the projected array output. The final unit-vector helper is given.
        #取出CLS，前面把它拼在了最前面
        cls_token=tokens[:,0,:]
        cls_token=self.post_norm(cls_token).astype(np.float32) # 【精度修复】
        output=cls_token@self.projection
        return unit_vector(output).astype(np.float32) # 【精度修复】