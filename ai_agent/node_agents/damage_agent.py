import os
import base64
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class DamageAgent:
    def __init__(self, model="gemini-2.5-flash-lite"):
        self.model = model
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.client = genai.GenerativeModel(model_name=self.model)

    def analyze(self, front_image, left_image, rear_image, right_image):
        # 4개 이미지를 분석하여 파손 상태 판별
        system_prompt = """
        당신은 장난감 파손 여부 및 부품 상태 판별 전문가입니다. 
        앞면, 왼쪽, 뒷면, 오른쪽 4개 각도에서 촬영된 이미지를 분석하여 종합적인 판단을 내려주세요.
        
        분석 결과를 다음 JSON 형식으로만 반환하세요.
        마크다운 코드 블록(```)은 절대 사용하지 마세요.
        
        {"damage": "상태", "damage_detail": "상세설명", "missing_parts": "부품누락여부"}
        
        분석해야 할 항목:
        1. damage: {없음, 미세한 파손, 경미한 파손, 부품 누락, 심각한 파손, 판단 불가}
        2. missing_parts: {없음, 있음, 불명}
        3. damage_detail: 구체적인 파손 상태 설명
        
        각도별 분석 지침:
        - 앞면: 전체적인 모양과 주요 부품 상태 파악
        - 왼쪽: 측면 구조와 부품 상태 확인
        - 뒷면: 뒷면 구조와 부품 상태 확인
        - 오른쪽: 오른쪽 측면 구조와 부품 상태 확인
        
        ⚠️ **특별 주의사항**:
        - 100% 확신할 때만 "없음"으로 분류
        - 의심스러운 부분이 있으면 "있음" 또는 "판단 불가"로 분류
        - 100% 확신하지 못하면 "부품 누락" 또는 "판단 불가"로 분류
        - 부품이 분리되거나 떨어져 보이면 "부품 누락"으로 분류
        - 미세한 흠집이나 스크래치가 보이면 "미세한 파손"으로 분류
        - 명확한 파손이나 균열이 보이면 "경미한 파손"으로 분류
        - 심각한 파손이나 부서진 부분이 보이면 "심각한 파손"으로 분류
        
        🔍 **분석 우선순위**:
        1. 부품 누락 여부 먼저 확인
        2. 파손 상태 세밀하게 관찰
        3. 확신이 서지 않으면 "판단 불가"로 분류
        
        📚 **Few-shot 예시**:
        예시 1: img_0027_front - 부품 누락
        - 특징: 장난감의 일부 부품이 누락되어 있음
        - 부품이 모두 있는가: "일부 없음"
        - 오염도: "보통" (재활용 가능하지만 부품 누락으로 기부 불가)
        - 결과: {"damage": "부품 누락", "damage_detail": "일부 부품 누락으로 기부 불가", "missing_parts": "있음"}
        
        예시 2: img_0047_front - 심각한 파손
        - 특징: 장난감에 경미한 파손이 있고, 오염도가 높음
        - 파손 상태: "경미한 파손" (부품 분리 등)
        - 오염도: "더러움 (재활용 불가)" - 심각한 오염으로 기부 불가
        - 결과: {"damage": "심각한 파손", "damage_detail": "경미한 파손 + 심각한 오염으로 재활용 불가", "missing_parts": "없음"}
        
        일반적인 예시:
        {"damage": "없음", "damage_detail": "2개 각도 모두 파손 없음, 모든 부품 완전", "missing_parts": "없음"}
        {"damage": "부품 누락", "damage_detail": "측면에서 머리 부품 누락 확인", "missing_parts": "있음"}
        {"damage": "미세한 파손", "damage_detail": "미세한 스크래치와 흠집 관찰", "missing_parts": "없음"}
        {"damage": "판단 불가", "damage_detail": "일부 각도에서 부품 상태 판단 어려움", "missing_parts": "불명"}
        
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
                    "이 장난감의 파손 상태와 부품 완전성을 4개 각도에서 분석해주세요. 위의 few-shot 예시를 참고하여 정확하게 분류해주세요.",
                    *images
                ],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=300,
                    temperature=0.0
                )
            )
            
            result = response.text
            if result is None:
                result = '{"damage": "없음", "damage_detail": "파손 없음", "missing_parts": "없음"}'
            
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
            
            print(f"DamageAgent 응답: {result} (토큰: {token_info['total_tokens']})")
            
            return result, token_info
            
        except Exception as e:
            print(f"DamageAgent 에러: {e}")
            return '{"damage": "없음", "damage_detail": "파손 없음", "missing_parts": "없음"}', {"total_tokens": 0}
