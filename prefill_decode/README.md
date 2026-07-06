这个问题是你现在做 **LLM Infra 最关键的一步：把“黑盒 generate”拆成“可观测 pipeline”**。

我给你一个可以直接放进 `llm-infra-lab/cache/prefill_decode/` 的**工程级设计方案**。

---

# 一、先讲清楚你要实现什么

你要做的是：

> **把一次 LLM 推理拆成两段，并分别计时**

```text id="goal1"
Prompt 输入
   ↓
[Prefill阶段] 处理整个prompt（计算attention KV）
   ↓
[Decode阶段] 一个token一个token生成
   ↓
输出结果
```

---

# 二、为什么要拆？

这是 AI Infra 面试高频点：

| 阶段      | 特点               |
| ------- | ---------------- |
| Prefill | 计算密集（GPU算力瓶颈）    |
| Decode  | 访存密集（KV Cache瓶颈） |

👉 优化方向完全不同

---

# 三、你可以怎么实现（核心思路）

你有 3 种实现方式，我给你推荐最适合 4060 + 学习用途的：

---

# ⭐ 方案1（推荐）：Transformers Hook + 时间拆分

## ✔ 核心思路

用：

```python
model.generate(...)
```

但通过：

* hook attention forward
* 或分两段调用

---

## ✔ 实现结构

```text id="code1"
prefill:
    input prompt 一次 forward

decode:
    每生成一个 token 单独 step
```

---

## ✔ 示例代码（简化版）

```python id="code2"
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

prompt = "Explain KV Cache in simple terms."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# =========================
# 1. Prefill
# =========================
start = time.time()

with torch.no_grad():
    outputs = model(
        **inputs,
        use_cache=True,
        return_dict=True
    )

prefill_time = time.time() - start

past_key_values = outputs.past_key_values
next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)

# =========================
# 2. Decode
# =========================
generated = [next_token.item()]
max_new_tokens = 50

start = time.time()

for _ in range(max_new_tokens):
    with torch.no_grad():
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )

    logits = outputs.logits
    past_key_values = outputs.past_key_values

    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
    generated.append(next_token.item())

decode_time = time.time() - start

text = tokenizer.decode(generated)

print("Prefill time:", prefill_time)
print("Decode time:", decode_time)
print("Output:", text)
```

---

# 四、你会看到什么现象（非常重要）

运行后你会观察：

## ✔ Prefill

```text id="obs1"
一次性较慢
GPU利用率高
计算量集中
```

---

## ✔ Decode

```text id="obs2"
每token较快
循环执行
KV Cache生效时明显加速
```

---

# 五、你这个实验的“关键指标设计”

你必须记录 3 个指标：

## 1️⃣ TTFT（First Token Time）

```text id="m1"
time to first token = prefill time
```

---

## 2️⃣ Decode latency

```text id="m2"
每token平均时间
```

---

## 3️⃣ Throughput

```text id="m3"
tokens / second
```

---
