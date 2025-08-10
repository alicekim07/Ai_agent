import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SoilAgent:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def analyze(self, image_bytes):
        # 바이너리 이미지 데이터를 직접 사용
        system_prompt = """
        당신은 장난감 오염 상태 분석 전문가입니다.
        분석 결과를 다음 JSON 형식으로만 반환하세요.
        마크다운 코드 블록(```)은 절대 사용하지 마세요.

        {"soil": "상태"}

        상태 값:
        - 깨끗
        - 약간 더러움
        - 더러움
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "다음 이미지의 장난감 오염 상태를 분석해주세요."},
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
                result = '{"soil": "깨끗"}'
            
            # 토큰 사용량 추가
            usage = response.usage
            token_info = {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0
            }
            print(f"SoilAgent 응답: {result} (토큰: {token_info['total_tokens']})")
            
            return result, token_info
        except Exception as e:
            print(f"SoilAgent 에러: {e}")
            return '{"soil": "깨끗"}', {"total_tokens": 0}
