import os
import base64
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class SoilAgent:
    def __init__(self, model="gemini-2.5-flash-lite"):
        self.model = model
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.client = genai.GenerativeModel(model_name=self.model)

    def analyze(self, front_image, left_image, rear_image, right_image):
        # 4개 이미지를 분석하여 오염 상태 판별
        system_prompt = """
        당신은 장난감 오염 상태 분석 전문가입니다.
        앞면, 왼쪽, 뒷면, 오른쪽 4개 각도에서 촬영된 이미지를 분석하여 종합적인 오염 상태를 판별하세요.
        
        분석 결과를 다음 JSON 형식으로만 반환하세요.
        마크다운 코드 블록(```)은 절대 사용하지 마세요.

        {"soil": "상태", "soil_detail": "상세설명"}

        상태 값:
        - 깨끗: 오염이 전혀 없거나 매우 적음
        - 보통: 약간의 오염이나 사용 흔적
        - 약간 더러움: 눈에 띄는 오염이나 얼룩
        - 더러움: 심한 오염이나 위생상 문제

        상세설명 예시:
        - 깨끗: "4개 각도 모두 오염 없음, 기부 적합"
        - 보통: "일부 각도에서 약간의 사용 흔적, 세척 후 기부 가능"
        - 약간 더러움: "여러 각도에서 얼룩이나 먼지 확인, 세척 필요"
        - 더러움: "여러 각도에서 심한 오염 확인, 위생상 기부 불가"

        반드시 순수 JSON 형식으로만 답변하세요.
        """
        try:
            # Gemini API용 이미지 데이터 준비
            images = []
            for img_bytes in [front_image, left_image, rear_image, right_image]:
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
                    "다음 4개 각도 이미지의 장난감 오염 상태를 종합적으로 분석해주세요.",
                    *images
                ],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=100,
                    temperature=0.0
                )
            )
            
            result = response.text
            if result is not None:
                result = result.strip()
            else:
                result = '{"soil": "깨끗", "soil_detail": "오염 없음, 기부 적합"}'
            
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
            
            print(f"SoilAgent 응답: {result} (토큰: {token_info['total_tokens']})")
            
            return result, token_info
        except Exception as e:
            print(f"SoilAgent 에러: {e}")
            return '{"soil": "깨끗", "soil_detail": "오염 없음, 기부 적합"}', {"total_tokens": 0}
