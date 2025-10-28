---
title: mmap映射模型文件
published: 2025-10-22
description: miniInfer中加载模型的总结
tags: [AI Infra, C++, miniInfer]
category: miniInfer
draft: false
---

## 1.mmap

  `mmap` 是用于内存映射的函数，它是一种在用户控件的程序和一个或多个文件或其他独享之间创建直接内存访问接口的方法。因为本质上是虚拟内存，采用分页管理的办法（突然让我想起了在cuda中，异步拷贝要使用固定内存的坑），所以可以加载很大的模型，这样我们就可以以字节为单位来访问我们的文件了，在我们的操作层面更加直观明了。

```c++
// 打开文件，只读方式
int fd = open(filename, O_RDONLY);
if (fd == -1) {
    perror("open");
    exit(EXIT_FAILURE);
}
// 映射文件到内存
void *mapped_address;
//offset通常以页为单位进行映射
mapped_address = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
if (mapped_address == MAP_FAILED) {
    perror("mmap");
    exit(EXIT_FAILURE);
}
```

  这样，我们就可以像指针一样，去访问我们的权重了，最后模型的参数地址都在mapped_address之中。

```c++
*(mapped_address + 32)
```

  同时，在我们的CPU测使用mmap之后，就不需要再手动分配或者管理那块内存的物理空间了，因为内核会完成一切的内存分配，加载，换出和释放等工作。



## 2.模型的导出

  我们经常会需要一个脚本和工具，用于将模型导出（负责将不同来源的pytorch模型）导出成为统一的`.bin` 文件格式，以便在后续的C环境中读取和推理。后续我们将会拆解模型导出的步骤。(在开发中的步骤也是，先用模型库读取，然后导出为C++能识别的bin模式)，下面我们以llama为例子，来拆接一下模型导出的步骤。

### 2.1模型导入方式

**checkpoint模型的导入**

```c++
.def load_checkpoint(checkpoint):
    checkpoint_dict = torch.load(checkpoint, map_location='cpu')
    gptconf = ModelArgs(**checkpoint_dict['model_args'])
    model = Transformer(gptconf)
    state_dict = checkpoint_dict['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model
```

`torch.load` 通常导入的是一个词典，包括了模型的配置，优化器（训练过程中的动态记录的学习信息），模型配置，当前的训练轮次，总而言之check_point里面会存放一些我训练框架需要的数据。然后从我们的模型数据中提取我们的模型层次信息，用Transformer进行初始化。然后从model里面取出我们的模型参数，相当于一个层名，一个参数。最后对权重进行操作，去掉一些多余的前缀，让前缀符合我们的模型标准（能够被load+state_dict读取）。最后开启推理模式，去掉一些我们不必要的计算。

**meta原版模型格式**

```bash
model_path/
├── params.json
├── consolidated.00.pth
├── consolidated.01.pth
├── tokenizer.model
```

`params.json` 里面一般是一些模型的配置和结构，`consolidated` 保存模型的权重，`tokenizer.model` 里面是我们的模型的分词器。

```python
model_paths = sorted(list(Path(model_path).glob('consolidated.*.pth')))
models = [torch.load(p, map_location='cpu') for p in model_paths]
```

  这里注意需要分片sort排序读取，这段代码就是把模型加载到内存之中，返回的就是一个dict.

  我们需要把llama中的权重沿着不同的维度拼接起来，从而恢复完整的LLama权重

```python
def concat_weights(models):
        state_dict = {}
        for name in list(models[0]):
            tensors = [model[name] for model in models]
            if len(tensors) == 1 or len(tensors[0].shape) == 1:
                state_dict[name] = tensors[0]
                continue
            is_axis_1 = (
                    name.startswith('tok_embeddings.')
                    or name.endswith('.attention.wo.weight')
                    or name.endswith('.feed_forward.w2.weight')
            )
            axis = 1 if is_axis_1 else 0
            state_dict[name] = torch.cat(tensors, dim=axis)
            for model in models:
                del model[name]
        return state_dict

```

- 首先我们直接获取第一个文件models[0]的key列表。
- 然后开始逐个参数逐个参数的去遍历。
- `len(tensors) > 1 and len(tensors[0].shape) == 1` 小切片不值得划分，模型会给所有的变量都拷贝一份，所以这里不需要合并了，这里我们删选一个拼接方式，dim筛选出我们需要拼接的维度，然后沿着我们的shape进行拼接（确保其他维度的shape完全相同）最后返回一个完整的state_dict就可以了。

然后再从`transformer` 里面读取我们的模型需要的一个数据

```python
config = ModelArgs()
# 模型隐藏的维度
config.dim = params["dim"]
# transformer的层数
config.n_layers = params["n_layers"]
# 每层attention分成多少个头
config.n_heads = params["n_heads"]
# key 和 value的头数
config.n_kv_heads = params.get('n_kv_heads') or params['n_heads']
# 前馈层维度对齐的背书
config.multiple_of = params["multiple_of"]
# LayerNorm的数值稳定项，防止除0
config.norm_eps = params["norm_eps"]

model = Transformer(config)
# 再从前面我们的static_dict之中加载我们的参数
model.tok_embeddings.weight = nn.Parameter(state_dict['tok_embeddings.weight'])
model.norm.weight = nn.Parameter(state_dict['norm.weight'])
```

从我们的tranformer模型中遍历模型的layer，装在我们的bin

```python
for layer in model.layers:
  i = layer.layer_id
  layer.attention_norm.weight = nn.Parameter(state_dict[f'layers.{i}.attention_norm.weight'])
  pass

model.output.weight = nn.Parameter(state_dict['output.weight'])
model.eval()
return model
```

  所以从meta中读取我们的参数，或者说有一些别的仓库中装载我们的模型，我们需要根据param文件去读取我们的配置，初始化我们的Transformer类，然后static_dict分块去拼接我们的tensort，最后遍历模型的层，给我们的transformer进行赋值就可以了

**从Hugging Face 模型进行导入**

```bash
model_path/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json / merges.txt（如果是BPE）
├── pytorch_model.bin
└── generation_config.json（可选）
```

  这里应该会方便很多，我们有一个HF的类

```python
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_dict = hf_model.state_dict()
```

  然后transformer就像前面一样读取配置即可，初始化transformer类，导入模型的权重。HuggingFace 为了配合自家的优化实现，会在加载时对权重进行一次“**排列**，也就是重新调整了 WQ 和 WK 的张量形状。所以在这里我们需要逆变换回去。

```python
# huggingface permutes WQ and WK, this function reverses it
def permute_reverse(w, n_heads=config.n_heads, dim1=config.dim, dim2=config.dim):
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)
```

**总结**

  在不同库之间的模型变换，我们都优先读取对方的配置，然后对参数调整，生成一个空壳模型，然后加载对方的权重文件[static_dic]，不过需要注意，权重可能进行的拆分存储，这个时候我们需要根据对方的存储方式进行逆变换回去。然后再讲权重装入我们的空壳模型之中。



## 3. 对模型进行不同精度的导出

**float32导出**

写入文件的头部信息

```python
out_file = open(filepath, 'wb')
out_file.write(struct.pack('I', 0x616b3432))
out_file.write(struct.pack('i', version))
header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                         n_kv_heads, p.vocab_size, p.max_seq_len)
out_file.write(header)
```

核心关键信息：

- p.dim:模型隐藏层的维度
- hidden_dim:FFN前馈层的w1输入权重
- p.n_layers:这里是transformer的层数
- p.n_head:注意力头数
- n_kv_heads:KV cache使用的透视
- vocab_size:词表大小
- max_seq_len:最大的薛烈长度

**输入和输出是否共享参数**

```python
hared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
out_file.write(struct.pack('B', int(shared_classifier)))
if not shared_classifier:
	weights.append(model.output.weight)
```

在很多的语言模型，输入词汇嵌入层和输出softmax层其实是可以共享参数的，共享参数就意味着我们可以在底层去共享内存。

**最后我们给weight打包**

```python
weights = [
    *[layer.attention_norm.weight for layer in model.layers],
    *[layer.ffn_norm.weight for layer in model.layers],
    model.norm.weight,
    model.tok_embeddings.weight,
    *[layer.attention.wq.weight for layer in model.layers],
    *[layer.attention.wk.weight for layer in model.layers],
    *[layer.attention.wv.weight for layer in model.layers],
    *[layer.attention.wo.weight for layer in model.layers],
    *[layer.feed_forward.w1.weight for layer in model.layers],
    *[layer.feed_forward.w2.weight for layer in model.layers],
    *[layer.feed_forward.w3.weight for layer in model.layers],
]
for w in weights:
    serialize_fp32(out_file, w)
    
def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)
```

最后这里记得用二进制进行存储，格式化字符串，连续发宝成4个float，然后再写入到我们的二进制文件之中。



**int8进行导出**

分组量化操作

```python
out_file.write(struct.pack('i', group_size))  # <-- 额外信息！
# 方便我们在逆变化层进行scale操作
for i, w in enumerate(weights):
    # quantize this weight
    q, s, err = quantize_q80(w, group_size)
    # save the int8 weights to file
    serialize_int8(out_file, q)  # save the tensor in int8
    serialize_fp32(out_file, s)  # save scale factors
    # logging
    ew.append((err, w.shape))
    print(f"{i + 1}/{len(weights)} quantized {tuple(w.shape)} to Q8_0 with max error {err}")
    
    
def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float()  # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:, None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:, None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr
```

- 我们的缩放因子计算取wmax / 127.0 来计算
- 我们用一个maxerr来存储最大误差，来评估模型的精度损失，不可能全存下来，因为这样就没有数据的节省了

## 4. 总结：

- 在本章中我们了解了模型的导入和导出的基本步骤
- int量化存储的时候，我们需要把一些无关紧要的层拉出来进行int分组存储
- 有一个scale的缩放因子就行了，最后便于我们进行还原，节省存储的空间，损失少量的精度。
