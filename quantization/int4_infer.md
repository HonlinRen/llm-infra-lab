# INT4 量化推理实验

目标：对比 FP16 与 INT4 在显存、加载时间、生成速度和回答质量上的差异。

## Transformers 4bit 示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
)
```

## 记录内容

- FP16 模型加载后显存占用。
- INT4 模型加载后显存占用。
- 同一个 prompt 下的 tokens/s。
- 回答是否出现明显质量下降。

## 面试表达

量化的核心价值是用更低精度表示权重，减少显存占用和访存压力；代价是可能带来精度损失，需要在速度、显存和效果之间做取舍。
