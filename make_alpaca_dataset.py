from datasets import load_dataset, Dataset
import json
from tqdm.auto import tqdm

title = "감염병의예방및관리에관한법률위반"
text = "질병관리청장, 시·도지사 또는 시장·군수·구청장은 감염병의 전파 방지 및 예방을 위하여 감염병의심자에게 자가 또는 시설에 격리 조치를 하게 할 수 있고, 누구든지 위와 같은 격리조치를 위반하여서는 아니 된다. 피고인은 코로나19 확진자와 접촉한 격리대상자로 ‘2021. 2. 25.부터 2021. 3. 10.까지 광주 북구 B건물, C호에 자가격리 조치한다'는 내용의 격리통지서를 수령하였음에도 불구하고, 격리기간 중인 2021. 3. 10. 10:00경부터 같은 날 11:30경까지 피고인의 자가격리 사실을 모르는 사업 관계자를 만나기 위하여 피고인의 휴대전화에 설치된 자가격리 안전보호 앱을 끄고 위 장소에서 이탈하여 자차로 전북 남원까지 운전하는 등 자가격리 조치를 위반하였다."

PROMPT_QA = """당신은 시험문제 출제위원입니다. 다음 자료에 기반하여 전문가 수준의 시험문제를 출제할 것입니다. 자료를 바탕으로 지시사항에 맞는 결과물을 json 형식으로 반환해주세요.

1. 생성한 문제는 실생활에서 사용하는 질문의 말투를 사용해야 합니다(~무엇인가요? ~작성해주세요. ~ 어떻게 해야하죠?)
2. 먼저 고등학교 수준의 문제를 생성하고, 이를 전문가 수준으로 고난이도 문제로 향상해주세요. 각 문제는 반드시 제시된 자료를 바탕으로 만들어져야 합니다. 연관성이 적더라도, 창의적인 아이디어로 해당 자료를 활용하세요.
3. 문제에는 답안 작성에 필요한 내용을 주어진 자료에서 추출해서 함께 제공해야합니다.
4. 출제할 문제의 과목 후보는 다음과 같습니다: 글쓰기, 한국어, 영어, 수학, 사회과학, 과학, 역사 문화예술, 법, 도덕, 정치, 종교, 외국어, 경제, 경영, 의료, 공학, 인문학 등 - 후보에 없어도, 적절한 과목을 자유롭게 말할 수 있다.

# 제목: {title}
# 자료:
{text}"""

PROMPT_WRITING = """당신은 글쓰기 시험문제 출제위원입니다. 다음 자료에 기반하여 전문가 수준의 시험문제를 출제할 것입니다. 자료를 바탕으로 지시사항에 맞는 결과물을 json 형식으로 반환해주세요.

1. 생성한 문제는 실생활에서 사용하는 질문의 말투를 사용해야 합니다(~무엇인가요? ~작성해주세요. ~ 어떻게 해야하죠?)
2. 먼저 고등학교 수준의 문제를 생성하고, 이를 전문가 수준으로 고난이도 문제로 향상해주세요. 각 문제는 반드시 제시된 자료를 바탕으로 만들어져야 합니다. 연관성이 적더라도, 창의적인 아이디어로 해당 자료를 활용하세요.
3. 문제에는 글쓰기 작성에 필요한 내용을 주어진 자료에서 추출해서 함께 제공해야합니다.
4. 출제할 문제의 주제 후보는 다음과 같습니다. 이 중에서 적절한 주제를 3가지 선택하세요: 이력서, 노래가사, 시 혹은 소설, 에세이, 극본, 시나리오, 여행일기, 여행계획서, 요리레시피, 해설, 자기소개서, 편지, 이메일, 리뷰 및 평가, 소셜 미디어 포스트, 일기, 청원서, 항의서, 쇼핑 리스트, 메모, 연구 논문 및 계획서, 비즈니스 보고서 및 게획서, 기술 문서, 발표자료, 계약서 혹은 법률 문서, 편집 및 출판 문서, 광고 카피라이트, 웹 콘텐츠, 뉴스레터, 연설문, 자기계발서, 분석보고서, 기획안, 제안서

# 제목: {title}
# 자료:
{text}"""


dataset = load_dataset("iknow-lab/ko-genstruct-v1", split="train")
new_dataset = []

for instance in tqdm(dataset):
    items = json.loads(instance["questions"])
    PROMPT = PROMPT_QA if "wikipedia_questions" == instance["source"] else PROMPT_WRITING
    if isinstance(items, dict):
        items = [items]

    try:
        for item in items:
            output = item["question"]
            
            prompt = PROMPT.format(title=instance["title"], text=instance["text"])

            subject = item.get("subject") or item.get("topic")
            output = {
                "subject": subject,
                "question": item["question"],
                "hard_questions": item["hard_questions"],
            }
            new_dataset.append(dict(
                instruction=prompt,
                output=json.dumps(output, ensure_ascii=False, indent=2),
            ))
    except Exception as e:
        print(e)
        print(instance["questions"])
        continue

print(new_dataset[0]["output"])
new_ds = Dataset.from_list(new_dataset)
new_ds.push_to_hub("iknow-lab/ko-genstruct-v1-alpaca")