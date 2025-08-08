import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class TypeAgent:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def analyze(self, image_bytes):
        # 이미지를 base64로 인코딩
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        return self.analyze_with_base64(image_base64)

    def analyze_with_base64(self, image_base64):
        # 이미 인코딩된 이미지를 사용
        system_prompt = """
        당신은 장난감 종류 및 특성 분석 전문가입니다. 
        분석해야 할 항목:
        1. toy_type: {피규어, 모형(=피규어로 매핑), 자동차 장난감, 변신 로봇, 건전지 장난감, 비건전지 장난감, 인형, 블록, 공, 아동 도서, 플라스틱 부품, 나무 장난감, 보행기, 탈것, 기타}
        2. battery: {건전지, 비건전지, 불명}
        3. size: {작음, 중간, 큼, 불명}

        예시:
        - 모형 로봇 → toy_type: "모형", battery: "비건전지", size: "중간"
        - 건전지 자동차 → toy_type: "자동차 장난감", battery: "건전지", size: "중간"
        - 큰 블록 세트 → toy_type: "블록", battery: "비건전지", size: "큼"

        반드시 순수 JSON 형식으로만 답변하세요. 마크다운 코드 블록(```)을 사용하지 마세요:
        {"toy_type": "종류", "battery": "건전지여부", "size": "크기"}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "이 장난감의 종류, 건전지 사용 여부, 크기를 분석해주세요."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150,
                temperature=0.0
            )
            result = response.choices[0].message.content
            if result is not None:
                result = result.strip()
            else:
                result = '{"toy_type": "기타", "battery": "불명", "size": "불명"}'
            
            # 토큰 사용량 추가
            usage = response.usage
            token_info = {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0
            }
            print(f"TypeAgent 응답: {result} (토큰: {token_info['total_tokens']})")
            
            return result, token_info
        except Exception as e:
            print(f"TypeAgent 에러: {e}")
            return '{"toy_type": "기타", "battery": "불명", "size": "불명"}', {"total_tokens": 0} 