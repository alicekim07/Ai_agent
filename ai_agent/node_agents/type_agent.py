import os
import base64
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class TypeAgent:
    def __init__(self, model="gemini-1.5-flash"):
        self.model = model
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.client = genai.GenerativeModel(model_name=self.model)

    def analyze(self, front_image, left_image):
        # 2개 이미지를 분석하여 종합적인 판단
        system_prompt = """
        당신은 장난감 종류 및 특성 분석 전문가입니다. 
        앞면과 왼쪽 2개 각도에서 촬영된 이미지를 분석하여 종합적인 판단을 내려주세요.
        
        분석해야 할 항목:
        1. type: {피규어, 모형(=피규어로 매핑), 자동차 장난감, 변신 로봇, 건전지 장난감, 비건전지 장난감, 인형, 블록, 공, 아동 도서, 플라스틱 부품, 나무 장난감, 보행기, 탈것, 기타}
        2. battery: {건전지, 비건전지, 불명}
        3. size: {작음, 중간, 큼, 불명}

        각도별 분석 지침:
        - 앞면: 전체적인 모양과 특징 파악
        - 왼쪽: 측면 구조와 부품 상태 확인

        예시:
        - 모형 로봇 → type: "모형", battery: "비건전지", size: "중간"
        - 건전지 자동차 → type: "자동차 장난감", battery: "건전지", size: "중간"
        - 큰 블록 세트 → type: "블록", battery: "비건전지", size: "큼"

        반드시 순수 JSON 형식으로만 답변하세요. 마크다운 코드 블록(```)을 사용하지 마세요:
        {"type": "종류", "battery": "건전지여부", "size": "크기"}
        """

        try:
            # Gemini API용 이미지 데이터 준비
            images = []
            for img_bytes in [front_image, left_image]:
                try:
                    # 바이트 데이터를 PIL Image로 변환
                    import io
                    from PIL import Image
                    img = Image.open(io.BytesIO(img_bytes))
                    images.append(img)
                except Exception as e:
                    print(f"이미지 처리 오류: {e}")
                    # 기본 이미지 생성
                    img = Image.new('RGB', (256, 256), color='white')
                    images.append(img)
            
            # Gemini API 호출
            response = self.client.generate_content(
                contents=[
                    system_prompt,
                    "이 장난감의 종류, 건전지 사용 여부, 크기를 2개 각도에서 분석해주세요.",
                    *images
                ],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=150,
                    temperature=0.0
                )
            )
            
            result = response.text
            if result is not None:
                result = result.strip()
            else:
                result = '{"type": "기타", "battery": "불명", "size": "불명"}'
            
            # 마크다운 코드 블록 제거
            if result.startswith('```json'):
                result = result[7:]  # ```json 제거
            if result.startswith('```'):
                result = result[3:]   # ``` 제거
            if result.endswith('```'):
                result = result[:-3]  # 끝의 ``` 제거
            result = result.strip()
            
            # Gemini API의 토큰 사용량 정보 추출
            token_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                try:
                    token_info = {
                        "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                        "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                        "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                    }
                except Exception as e:
                    print(f"토큰 정보 추출 오류: {e}")
            
            print(f"TypeAgent 응답: {result} (토큰: {token_info['total_tokens']})")
            
            return result, token_info
        except Exception as e:
            print(f"TypeAgent 에러: {e}")
            return '{"type": "기타", "battery": "불명", "size": "불명"}', {"total_tokens": 0} 