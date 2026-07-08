import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


# 基座模型：LoRA 微调时，原始大模型权重大部分保持冻结，只训练很少量的 adapter 参数。
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# 训练数据：中文 Alpaca 风格的 instruction 数据集。
# 本实验只取前 500 条，并缓存到本地，避免每次训练都重新下载和切片。
dataset_name = "llm-wizard/alpaca-gpt4-data-zh"
dataset_cache_dir = "./data/alpaca_500"

# tokenizer 负责把格式化后的指令文本转成 token ids。
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    # Qwen instruct 模型可能没有单独的 pad token。
    # 对因果语言模型微调来说，复用 eos token 作为 pad token 是常见做法。
    tokenizer.pad_token = tokenizer.eos_token

# 加载基座模型。device_map="auto" 会在有 CUDA 时自动把模型放到 GPU。
# 没有 GPU 时使用 float32；CPU 上 fp16 算子支持不完整，容易变慢或报错。
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
)

# 优先读取本地缓存数据。如果缓存不存在，就从 Hugging Face 下载数据集，
# 取前 500 条后保存到本地，让脚本可以独立运行。
try:
    dataset = load_from_disk(dataset_cache_dir)
except FileNotFoundError:
    dataset = load_dataset(dataset_name)["train"].select(range(500))
    dataset.save_to_disk(dataset_cache_dir)


def format_prompt(example):
    """把一条 Alpaca 样本转换成 SFTTrainer 学习的文本格式。

    SFT 本质上还是因果语言模型训练：模型看到前面的 token，
    学习预测下一个 token。这里把 instruction/input/output 拼成统一格式，
    让模型学习“根据指令生成回答”。
    """

    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )

    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


# LoRA 配置：控制要训练的低秩 adapter 矩阵。
# 原始 Qwen 权重不直接更新，只在注意力投影层旁边训练少量 LoRA 参数，
# 这样能显著降低显存占用和训练成本。
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# 训练参数：
# - per_device_train_batch_size=1：单卡 batch 小，适合 8GB 显存实验
# - gradient_accumulation_steps=8：累积 8 步梯度，等效 batch size 约为 8
# - fp16：只有 CUDA 可用时启用，降低显存并提升速度
training_args = TrainingArguments(
    output_dir="./qwen-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=5,
    save_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# SFTTrainer 会完成 prompt 格式化、tokenization、labels 构造、LoRA 包装和训练循环。
# 训练完成后保存的是 LoRA adapter，不是完整模型；推理时需要“基座模型 + adapter”一起加载。
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    processing_class=tokenizer,
    formatting_func=format_prompt,
)

trainer.train()
trainer.save_model("./qwen-lora-adapter")
