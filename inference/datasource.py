from datasets import load_dataset

dataset = load_dataset("llm-wizard/alpaca-gpt4-data-zh")

small = dataset["train"].select(range(500))
small.save_to_disk("./data/alpaca_500")
print(dataset)