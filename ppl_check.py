import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np
from tqdm.auto import tqdm
import jsonlines


# GPT-2 모델과 토크나이저 로드 (한국어 모델을 사용하는 것이 더 좋습니다)
model_name = "skt/kogpt2-base-v2"  # 한국어 GPT-2 모델
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def calculate_perplexity(text):
    encodings = tokenizer(text, return_tensors="pt")
    max_length = model.config.n_positions
    stride = 1024
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def filter_text_by_perplexity(items, threshold):
    filtered_texts = []
    ppls = []
    for item in tqdm(items):
        text = item["llm_response"]
        perplexity = calculate_perplexity(text)
        ppls.append(perplexity)
        if perplexity <= threshold:
            filtered_texts.append(item)
        else:
            print(f"Filtered out (Perplexity: {perplexity:.2f})")
            print(text)
    return filtered_texts, ppls


items = list(jsonlines.open("data/question_answers_multi_gpt-3.5-turbo.jsonl"))
threshold = 250  # 이 값은 실험을 통해 조정해야 합니다
filtered_items, ppl = filter_text_by_perplexity(items, threshold)

print("Perplexities:")
print(np.mean(ppl), np.std(ppl))
print("Quantiles:")
for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    print(q, np.quantile(ppl, q))

print(f"Filtered: {len(filtered_items)} / {len(items)}")

with jsonlines.open("data/question_answers_multi_gpt-3.5-turbo_filtered.jsonl", "w") as fout:
    for item in filtered_items:
        fout.write(item)