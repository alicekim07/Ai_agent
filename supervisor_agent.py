import json
import re
import concurrent.futures
import base64
from node_agents.type_agent import TypeAgent
from node_agents.material_agent import MaterialAgent
from node_agents.damage_agent import DamageAgent

def _extract_json_text(raw_text: str) -> str:
    """Extract JSON object text from a raw LLM response."""
    if not raw_text:
        return None
    text = raw_text.strip()

    # Strip code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Extract JSON object between first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return None

class SupervisorAgent:
    def __init__(self):
        self.type_agent = TypeAgent()
        self.material_agent = MaterialAgent()
        self.damage_agent = DamageAgent()

    def process(self, image_bytes):
        # 0. 이미지를 한 번만 base64로 인코딩
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # 1. 각 개별 에이전트를 병렬로 실행 (인코딩된 이미지 전달)
        print("개별 에이전트 분석 중...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # 3개 에이전트를 동시에 실행 (타임아웃 90초)
            future_type = executor.submit(self.type_agent.analyze_with_base64, image_base64)
            future_material = executor.submit(self.material_agent.analyze_with_base64, image_base64)
            future_damage = executor.submit(self.damage_agent.analyze_with_base64, image_base64)
            
            # 결과 수집 (타임아웃 90초)
            try:
                type_response = future_type.result(timeout=90)
                material_response = future_material.result(timeout=90)
                damage_response = future_damage.result(timeout=90)
            except concurrent.futures.TimeoutError:
                print("일부 에이전트가 타임아웃되었습니다. 순차 처리로 전환합니다.")
                # 타임아웃 시 순차 처리로 전환
                type_response = self.type_agent.analyze_with_base64(image_base64)
                material_response = self.material_agent.analyze_with_base64(image_base64)
                damage_response = self.damage_agent.analyze_with_base64(image_base64)
        
        # 2. JSON 파싱
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

        # 3. 통합 판단 로직 적용
        toy_type = type_result.get("type", "기타")
        battery = type_result.get("battery", "불명")
        material = material_result.get("material", "불명")
        damage = damage_result.get("damage", "불명")
        
        # 기본값 설정
        soil = "깨끗"  # 기본값
        size = "중간"  # 기본값
        notes = ""
        
        # 4. 기부 판단 로직 (통합 에이전트와 동일한 규칙)
        donate, donate_reason, repair_or_disassemble = self._judge_donation(
            toy_type, battery, material, damage, soil
        )
        
        # 5. 토큰 사용량 계산 (개별 에이전트들의 합계)
        token_usage = {
            "type_agent": 0,      # 실제로는 각 에이전트에서 추출 필요
            "material_agent": 0,  # 실제로는 각 에이전트에서 추출 필요
            "damage_agent": 0,    # 실제로는 각 에이전트에서 추출 필요
            "total": 0            # 실제로는 각 에이전트에서 추출 필요
        }

        # 6. 결과 반환
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
            
        if damage == "심각":
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
                
        # 경미한 파손
        if damage == "경미":
            return False, "경미한 파손으로 수리 필요", "경미 수리 권장"
            
        # 불확실한 경우
        return False, "정보 부족으로 추가 검토 필요", "추가 검토 필요" 