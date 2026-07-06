import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
dataset_name = "llm-wizard/alpaca-gpt4-data-zh"
dataset_cache_dir = "./data/alpaca_500"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
)

try:
    dataset = load_from_disk(dataset_cache_dir)
except FileNotFoundError:
    dataset = load_dataset(dataset_name)["train"].select(range(500))
    dataset.save_to_disk(dataset_cache_dir)

def format_prompt(example):
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

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

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

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    processing_class=tokenizer,
    formatting_func=format_prompt
)

trainer.train()
trainer.save_model("./qwen-lora-adapter")
