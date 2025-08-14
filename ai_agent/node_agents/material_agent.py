import os
import base64
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class MaterialAgent:
    def __init__(self, model="gemini-2.5-flash-lite"):
        self.model = model
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.client = genai.GenerativeModel(model_name=self.model)

    def analyze(self, front_image, left_image, rear_image, right_image):
        # 4개 이미지를 분석하여 재료 판별
        system_prompt = """
        당신은 장난감 재료 분석 전문가입니다. 
        앞면, 왼쪽, 뒷면, 오른쪽 4개 각도에서 촬영된 이미지를 분석하여 종합적인 재료 판별을 내려주세요.
        
        분석 결과를 다음 JSON 형식으로만 반환하세요.
        마크다운 코드 블록(```)은 절대 사용하지 마세요.
        
        {"material": "재료", "material_detail": "단일/혼합", "confidence": "높음/보통/낮음", "notes": "상세설명"}
        
        분석해야 할 항목:
        1. material: {플라스틱, 금속, 나무, 섬유/천, 실리콘, 유리, 고무, 플라스틱,금속, 플라스틱,섬유, 플라스틱,실리콘, 플라스틱,고무, 금속,나무, 플라스틱,유리}
        2. material_detail: {단일 소재, 혼합 소재}
        3. confidence: {높음, 보통, 낮음}
        4. notes: 구체적인 재료 분석 설명
        
        재료 분류 기준:
        **단일 소재** (100% 확신할 때만):
        - 플라스틱: 투명하거나 불투명한 합성 수지, 균일한 색상과 질감
        - 금속: 금속성 광택이나 색상, 금속 특유의 반사
        - 나무: 나무 질감이나 나뭇결, 자연스러운 나무 색상
        - 섬유/천: 천이나 실, 털 같은 질감, 부드러운 표면
        - 실리콘: 고무 같은 질감이나 탄성, 매트한 표면
        - 유리: 투명하고 단단한 재질, 빛 반사
        - 고무: 검은색이나 갈색의 탄성 재질, 매트한 표면

        **혼합 소재** (의심스러운 경우 즉시 선택):
        - 플라스틱,금속: 플라스틱 본체 + 금속 부품 (나사, 축, 바퀴 등)
        - 플라스틱,섬유: 플라스틱 본체 + 천/실 장식 (의류, 털, 끈 등)
        - 플라스틱,실리콘: 플라스틱 본체 + 실리콘 부품 (그립, 쿠션 등)
        - 플라스틱,고무: 플라스틱 본체 + 고무 바퀴나 그립
        - 금속,나무: 금속 프레임 + 나무 부품
        - 플라스틱,유리: 플라스틱 본체 + 유리 렌즈나 장식
        
        ⚠️ **특별 주의사항**:
        - 천이나 실이 조금이라도 보이면 즉시 "혼합 소재"로 분류
        - 실리콘/고무 부품이 조금이라도 보이면 즉시 "혼합 소재"로 분류
        - 금속 부품이 조금이라도 보이면 즉시 "혼합 소재"로 분류
        - 색상이나 질감이 다른 부분이 있으면 "혼합 소재"로 분류
        - 의심스러운 부분이 있으면 "혼합 소재"로 분류
        - 100% 확신하지 못하면 "혼합 소재"로 분류
        
        예시:
        {"material": "플라스틱", "material_detail": "단일 소재", "confidence": "높음", "notes": "4개 각도 모두 투명한 플라스틱으로만 구성, 100% 확신"}
        {"material": "플라스틱,천", "material_detail": "혼합 소재", "confidence": "높음", "notes": "4개 각도에서 플라스틱 본체와 천 장식 요소 확인"}
        {"material": "플라스틱,금속", "material_detail": "혼합 소재", "confidence": "높음", "notes": "플라스틱 본체에 금속 부품(나사, 축) 확인"}
        
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
                    "이 장난감의 재료를 4개 각도에서 분석해주세요.",
                    *images
                ],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=300,
                    temperature=0.0
                )
            )
            
            result = response.text
            if result is None:
                result = '{"material": "플라스틱", "material_detail": "단일 소재", "confidence": "낮음", "notes": "분석 실패로 인한 기본값"}'
            
            # 마크다운 코드 블록 제거
            result = result.strip()
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
            
            print(f"MaterialAgent 응답: {result} (토큰: {token_info['total_tokens']})")
            
            return result, token_info
            
        except Exception as e:
            print(f"MaterialAgent 에러: {e}")
            return '{"material": "플라스틱", "material_detail": "단일 소재", "confidence": "낮음", "notes": "분석 실패로 인한 기본값"}', {"total_tokens": 0}
