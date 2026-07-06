import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "./qwen-lora-adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

question = "生成一个8个字符的密码。"

messages = [{"role": "user", "content": question}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

answer_ids = outputs[0][inputs["input_ids"].shape[-1]:]
answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

print(answer)
