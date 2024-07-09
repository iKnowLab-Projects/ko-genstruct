from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json, jsonlines
from tqdm.auto import tqdm


# 모델 로드
model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

# 원본 텍스트
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a question, retrieve relevant articles that answer the question'

def filter_unrelated_question(original_text, questions):
    questions = [get_detailed_instruct(task, q) for q in questions]

    # 임베딩 생성
    original_embedding = model.encode([original_text])[0]
    question_embeddings = model.encode(questions)

    # 코사인 유사도 계산
    similarities = cosine_similarity([original_embedding], question_embeddings)[0]

    # 임계값 설정 (이 값은 조정 가능합니다)
    threshold = 0.80
    
    unrelated = []
    for question, similarity in zip(questions, similarities):
        if similarity <= threshold:
            unrelated.append((question, similarity))

    return unrelated
        

def check_item(item):
    response = item["llm_response"]
    if not response:
        return False
    if "Gemini Refusal" in response:
        return False
    try:
        if "```json" in response:
            response = response.split("```json", 1)[1].split("```")[0]
        obj = json.loads(response)
        return obj
    except:
        return False
    
def parse_item(item):
    response = item["llm_response"]
    response = response.replace("# 결과", "").strip()
    if not response:
        return False
    if "Gemini Refusal" in response:
        return False
    try:
        if "```json" in response:
            response = response.split("```json", 1)[1].split("```")[0]
        obj = json.loads(response)
        return obj
    except:
        return False
    
# 관련성 있는 질문과 관련성 없는 질문 분류
# original_text = "인공지능은 인간의 학습능력, 추론능력, 지각능력, 언어이해능력 등을 컴퓨터 프로그램으로 실현한 기술이다. 인공지능은 학습, 문제해결, 패턴인식 등의 분야에서 활용되고 있다."

# # 생성된 질문들
# questions = [
#     "인공지능의 정의는 무엇인가요?",
#     "인공지능은 어떤 분야에서 활용되나요?",
#     "인공지능의 한계는 무엇인가요?",
#     "날씨가 맑은 날 어디로 여행 가면 좋을까요?",
#     "인공지능은 어떤 능력을 모방하나요?"
# ]
# unrelated_questions = filter_unrelated_question(original_text, questions)


def handle_list(fout, items, source):
    for item in tqdm(items):
        response = parse_item(item)
        if not response:
            continue
        
        if isinstance(response, dict):
            try:
                questions = [response['question'].strip()]
            except:
                print("Weird response format")
                print(response)
                continue
        elif isinstance(response, list):
            questions = [r['question'].strip() for r in response]
        else:
            print("Weird response format")
            print(response)
            continue
        
        has_empty = False
        for q in questions:
            if not q.strip():
                # print("Empty question")
                # print(response)
                has_empty = True
                break
        
        if has_empty:
            continue

        original_text = item['text']
        unrelated_questions = filter_unrelated_question(original_text, questions)
        
        if unrelated_questions:
            print("관련성 없는 질문:", item["title"])
            for question, similarity in unrelated_questions:
                print(f"질문: {question}")
                print(f"유사도: {similarity:.4f}")
            print("---")
        else:
            fout.write({
                "title": item["title"],
                "text": item["text"],
                "questions": json.dumps(response, ensure_ascii=False),
                "source": source,
                "generator": item.get("model", "gemini-1.5-flash")
            })


fout = jsonlines.open("data/ko-genstruct-v1/train.json", "w")
items = jsonlines.open("data/question_list.jsonl")
handle_list(fout, items, "wikipedia_questions")
handle_list(fout, jsonlines.open("data/question_writing_list.jsonl"), "namuwiki_writing")

fout.close()