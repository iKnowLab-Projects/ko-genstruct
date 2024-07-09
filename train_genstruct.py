from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported

# 데이터 로드
dataset = load_dataset("iknow-lab/ko-genstruct-v1-alpaca", split="train")

# 토크나이저와 모델 로드
model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_seq_length = 4096 # 8192

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype=torch.bfloat16,
    # load_in_4bit = True,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 42,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# 데이터 전처리 함수
def preprocess_function(item):
    prompt = item["instruction"]
    output = item["output"]

    instruction = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    output = tokenizer.encode(output, add_special_tokens=False)

    inputs = instruction + output
    labels = [-100] * len(instruction) + output

    if len(inputs) > max_seq_length:
        inputs = inputs[:max_seq_length]
        labels = labels[:max_seq_length]
        
    return {
        "input_ids": inputs,
        "attention_mask": [1] * len(inputs),
        "labels": labels,
    }

def filter_no_labels(item):
    return sum(1 if x >= 0 else 0 for x in item["labels"]) > 0

# 데이터셋 전처리
tokenized_dataset = dataset.map(preprocess_function)
tokenized_dataset = tokenized_dataset.filter()

print(tokenizer.decode(tokenized_dataset[0]['input_ids']))

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    warmup_steps=40,
    weight_decay=0.0,
    logging_dir="./logs",
    logging_steps=1,
    save_strategy="epoch",
    learning_rate=1e-4,
    bf16=True,
    remove_unused_columns=True,
)

# Trainer 초기화 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# 모델 저장
model.save_pretrained("./bllossom-8b-lora-16-32-kogenstruct")
# model.push_to_hub("bllossom-8b-lora-16-32-kogenstruct")