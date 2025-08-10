import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class DamageAgent:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def analyze(self, image_bytes):
        # 바이너리 이미지 데이터를 직접 사용
        system_prompt = """
        당신은 장난감 파손 여부 판별 전문가입니다.
        이미지를 분석하여 파손 상태를 판별하세요.

        가능한 상태:
        - 없음
        - 미세한 파손
        - 심각한 파손

        반드시 순수 JSON 형식으로만 답변하세요.
        마크다운 코드 블록(```)은 사용하지 마세요.

        예시:
        {"damage": "상태"}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "다음 이미지의 장난감 파손 여부를 분석해주세요."},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}",
                            "detail": "low"
                        }}
                    ]}
                ],
                max_completion_tokens=100,
                temperature=0.0
            )
            result = response.choices[0].message.content
            if result is not None:
                result = result.strip()
            else:
                result = '{"damage": "없음"}'
            
            # 토큰 사용량 추가
            usage = response.usage
            token_info = {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0
            }
            print(f"DamageAgent 응답: {result} (토큰: {token_info['total_tokens']})")
            
            return result, token_info
        except Exception as e:
            print(f"DamageAgent 에러: {e}")
            return '{"damage": "없음"}', {"total_tokens": 0}
