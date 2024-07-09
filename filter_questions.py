from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json, jsonlines
from tqdm.auto import tqdm
from datasets import load_dataset

# 모델 로드
model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

# 원본 텍스트
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a question, retrieve relevant articles that answer the question'

# 코사인 유사도를 사용하여 관련성이 없는 질문을 필터링
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
        
    
def get_questions(item):
    items = json.loads(item["questions"])
    if isinstance(items, dict):
        items = [items]
    questions = []
    for item in items:
        questions.append(item["question"])
        if "hard_questions" in item:
            questions.extend(item["hard_questions"])
        elif "hard_question" in item:
            questions.append(item["hard_question"])

    return questions
    
    
def check_dataset(items):
    for item in tqdm(items):
        questions = get_questions(item)
        
        # has_empty = False
        # for q in questions:
        #     if not q.strip():
        #         has_empty = True
        #         break
        
        # if has_empty:
        #     continue

        original_text = item['text']
        unrelated_questions = filter_unrelated_question(original_text, questions)
        
        if unrelated_questions:
            print("관련성 없는 질문:", item["title"])
            for question, similarity in unrelated_questions:
                print(f"질문: {question}")
                print(f"유사도: {similarity:.4f}")
            print("---")

# 중복 제거 함수
def remove_duplicates(questions, threshold=0.95):
    questions = [get_detailed_instruct(task, q) for q in questions]
    embeddings = model.encode(questions)

    unique_questions = []
    unique_indices = []

    for i, (question, embedding) in enumerate(zip(questions, embeddings)):
        if i == 0:
            unique_questions.append(question)
            unique_indices.append(i)
            continue

        similarities = cosine_similarity([embedding], embeddings[:i])[0]
        if not np.any(similarities > threshold):
            unique_questions.append(question)
            unique_indices.append(i)

    return unique_questions, unique_indices

dataset = load_dataset("iknow-lab/ko-genstruct-v1", split="train[:50]")
check_dataset(dataset)


questions = [q for item in dataset for q in get_questions(item)]

# 중복 제거 실행
unique_questions, unique_indices = remove_duplicates(questions)
print(f"Original questions: {len(questions)}")
print(f"Unique questions: {len(unique_questions)}")