import json
import re
import concurrent.futures
import base64
from PIL import Image
import io
from node_agents.type_agent import TypeAgent
from node_agents.material_agent import MaterialAgent
from node_agents.damage_agent import DamageAgent
from node_agents.soil_agent import SoilAgent

def optimize_image_size(image_bytes, max_size=512):
    """이미지 크기를 최적화하여 인코딩 시간을 단축"""
    try:
        # 이미지 열기
        image = Image.open(io.BytesIO(image_bytes))
        
        # 이미지 크기 확인
        width, height = image.size
        
        # 이미 충분히 작으면 그대로 반환
        if width <= max_size and height <= max_size:
            return image_bytes
        
        # 비율 유지하면서 리사이즈
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        # 리사이즈
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 바이트로 변환
        output = io.BytesIO()
        resized_image.save(output, format='JPEG', quality=85, optimize=True)
        optimized_bytes = output.getvalue()
        
        print(f"이미지 최적화: {width}x{height} → {new_width}x{new_height}")
        return optimized_bytes
        
    except Exception as e:
        print(f"이미지 최적화 실패: {e}")
        return image_bytes

class SupervisorAgent:
    def __init__(self):
        self.type_agent = TypeAgent()
        self.material_agent = MaterialAgent()
        self.damage_agent = DamageAgent()
        self.soil_agent = SoilAgent()

    def process(self, image_bytes):
        # 0. 이미지 크기 최적화
        optimized_image_bytes = optimize_image_size(image_bytes)
        
        # 1. 이미지를 한 번만 base64로 인코딩
        image_base64 = base64.b64encode(optimized_image_bytes).decode('utf-8')
        
        # 2. 각 개별 에이전트를 병렬로 실행 (인코딩된 이미지 전달)
        print("개별 에이전트 분석 중...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # 4개 에이전트를 동시에 실행 (타임아웃 90초)
            future_type = executor.submit(self.type_agent.analyze_with_base64, image_base64)
            future_material = executor.submit(self.material_agent.analyze_with_base64, image_base64)
            future_damage = executor.submit(self.damage_agent.analyze_with_base64, image_base64)
            future_soil = executor.submit(self.soil_agent.analyze_with_base64, image_base64)
            
            # 결과 수집 (타임아웃 90초)
            try:
                type_response, type_tokens = future_type.result(timeout=90)
                material_response, material_tokens = future_material.result(timeout=90)
                damage_response, damage_tokens = future_damage.result(timeout=90)
                soil_response, soil_tokens = future_soil.result(timeout=90)
            except concurrent.futures.TimeoutError:
                print("일부 에이전트가 타임아웃되었습니다. 순차 처리로 전환합니다.")
                # 타임아웃 시 순차 처리로 전환
                type_response, type_tokens = self.type_agent.analyze_with_base64(image_base64)
                material_response, material_tokens = self.material_agent.analyze_with_base64(image_base64)
                damage_response, damage_tokens = self.damage_agent.analyze_with_base64(image_base64)
                soil_response, soil_tokens = self.soil_agent.analyze_with_base64(image_base64)
        
        # 3. JSON 파싱
        try:
            type_result = json.loads(type_response)
        except:
            type_result = {"type": "기타", "battery": "불명"}
            
        try:
            material_result = json.loads(material_response)
        except:
            material_result = {"material": "불명"}
            
        try:
            damage_result = json.loads(damage_response)
        except:
            damage_result = {"damage": "불명"}
            
        try:
            soil_result = json.loads(soil_response)
        except:
            soil_result = {"soil": "깨끗"}

        # 4. 통합 판단 로직 적용
        toy_type = type_result.get("type", "기타")
        battery = type_result.get("battery", "불명")
        material = material_result.get("material", "불명")
        damage = damage_result.get("damage", "불명")
        soil = soil_result.get("soil", "깨끗")
        
        # 기본값 설정
        size = "중간"  # 기본값
        notes = ""
        
        # 5. 기부 판단 로직 (통합 에이전트와 동일한 규칙)
        donate, donate_reason, repair_or_disassemble = self._judge_donation(
            toy_type, battery, material, damage, soil
        )
        
        # 6. 토큰 사용량 계산 (개별 에이전트들의 합계)
        token_usage = {
            "type_agent": type_tokens.get("total_tokens", 0),
            "material_agent": material_tokens.get("total_tokens", 0),
            "damage_agent": damage_tokens.get("total_tokens", 0),
            "soil_agent": soil_tokens.get("total_tokens", 0),
            "total": type_tokens.get("total_tokens", 0) + material_tokens.get("total_tokens", 0) + damage_tokens.get("total_tokens", 0) + soil_tokens.get("total_tokens", 0)
        }

        # 7. 결과 반환
        return {
            "장난감 종류": toy_type,
            "건전지 여부": battery,
            "재료": material,
            "파손": damage,
            "오염도": soil,
            "관찰사항": notes,
            "크기": size,
            "기부 가능 여부": "가능" if donate else "불가능",
            "기부 불가 사유": donate_reason,
            "수리/분해": repair_or_disassemble,
            "토큰 사용량": token_usage
        }

    def _judge_donation(self, toy_type, battery, material, damage, soil):
        # 기부 불가 조건들
        if material == "나무":
            return False, "나무 소재는 안전상 기부 불가", "분해/부품 추출(업사이클)"
            
        if "천" in material or material == "섬유":
            return False, "천/섬유 소재는 위생상 기부 불가", "분해/부품 추출(업사이클)"
            
        if "심각" in damage or damage == "심각한 파손":
            return False, "심각한 파손으로 완제품 기부 불가", "분해/부품 추출(업사이클)"
            
        if soil == "더러움":
            return False, "오염 상태로 위생상 기부 불가", "분해/부품 추출(업사이클)"
            
        if toy_type in ["인형", "아동 도서", "보행기", "탈것"]:
            return False, f"{toy_type}은 기부 불가 종류", "분해/부품 추출(업사이클)"
            
        # 기부 가능 조건들
        if material == "플라스틱" and damage == "없음" and soil == "깨끗":
            if battery == "건전지":
                return True, "플라스틱 건전지 장난감, 상태 양호", "수리 불필요(완제품)"
            else:
                return True, "플라스틱 비건전지 장난감, 상태 양호", "수리 불필요(완제품)"
                
        # 경미한 파손 (다양한 표현 매칭)
        if "경미" in damage or "미세" in damage or "경미한 파손" in damage or "미세한 파손" in damage:
            return False, "경미한 파손으로 수리 필요", "경미 수리 권장"
            
        # 불확실한 경우
        return False, "정보 부족으로 추가 검토 필요", "추가 검토 필요" 