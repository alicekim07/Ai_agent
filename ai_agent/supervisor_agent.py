import json
import concurrent.futures

from ai_agent.node_agents.type_agent import TypeAgent
from ai_agent.node_agents.material_agent import MaterialAgent
from ai_agent.node_agents.damage_agent import DamageAgent
from ai_agent.node_agents.soil_agent import SoilAgent
from ai_agent.image_input import optimize_image_size

class SupervisorAgent:
    def __init__(self):
        self.type_agent = TypeAgent()
        self.material_agent = MaterialAgent()
        self.damage_agent = DamageAgent()
        self.soil_agent = SoilAgent()

    def process(self, image_bytes):
        # 0. 이미지 크기 최적화
        optimized_image_bytes = optimize_image_size(image_bytes)
        
        # 2. 각 개별 에이전트를 병렬로 실행 (바이너리 이미지 데이터 직접 전달)
        print("개별 에이전트 분석 중...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # 4개 에이전트를 동시에 실행
            future_type = executor.submit(self.type_agent.analyze, optimized_image_bytes)
            future_material = executor.submit(self.material_agent.analyze, optimized_image_bytes)
            future_damage = executor.submit(self.damage_agent.analyze, optimized_image_bytes)
            future_soil = executor.submit(self.soil_agent.analyze, optimized_image_bytes)
            
            # 결과 수집 (타임아웃 30초)
            try:
                type_response, type_tokens = future_type.result(timeout=30)
                material_response, material_tokens = future_material.result(timeout=30)
                damage_response, damage_tokens = future_damage.result(timeout=30)
                soil_response, soil_tokens = future_soil.result(timeout=30)
            except concurrent.futures.TimeoutError:
                print("일부 에이전트가 타임아웃되었습니다. 순차 처리로 전환합니다.")
                # 타임아웃 시 순차 처리로 전환
                type_response, type_tokens = self.type_agent.analyze(optimized_image_bytes)
                material_response, material_tokens = self.material_agent.analyze(optimized_image_bytes)
                damage_response, damage_tokens = self.damage_agent.analyze(optimized_image_bytes)
                soil_response, soil_tokens = self.soil_agent.analyze(optimized_image_bytes)
        
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
        size = type_result.get("size", "중간")  # AI 분석 결과 사용
        material = material_result.get("material", "불명")
        damage = damage_result.get("damage", "불명")
        soil = soil_result.get("soil", "깨끗")
        
        # 의미있는 관찰사항 생성
        notes = self._generate_meaningful_notes(toy_type, battery, material, damage, soil)
        
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

    def _generate_meaningful_notes(self, toy_type, battery, material, damage, soil):
        """분석 결과를 바탕으로 의미있는 관찰사항을 생성합니다."""
        notes = []
        
        # 장난감 종류별 특성
        if toy_type == "피규어" or toy_type == "모형":
            notes.append("피규어/모형류는 세밀한 파손 여부 확인 필요")
        elif toy_type == "자동차 장난감":
            notes.append("자동차 장난감은 바퀴와 동작부위 상태 중요")
        elif toy_type == "변신 로봇":
            notes.append("변신 로봇은 관절부위 파손 여부 확인 필요")
        elif toy_type == "블록":
            notes.append("블록류는 결합부위 마모 상태 확인")
        elif toy_type == "공":
            notes.append("공류는 압축성과 표면 상태 확인")
        
        # 건전지 관련
        if battery == "건전지":
            notes.append("건전지 장난감은 전자부품 상태 확인 필요")
        elif battery == "비건전지":
            notes.append("비건전지 장난감은 기계적 동작 확인")
        
        # 재료별 특성
        if material == "플라스틱":
            notes.append("플라스틱 소재는 균열이나 변형 확인")
        elif material == "금속":
            notes.append("금속 소재는 녹이나 변형 상태 확인")
        elif material == "나무":
            notes.append("나무 소재는 균열이나 부식 상태 확인")
        
        # 파손 상태별
        if damage == "없음":
            notes.append("파손 없음 - 기부 적합")
        elif "미세" in damage or "경미" in damage:
            notes.append("경미한 파손 - 수리 후 기부 가능")
        elif "심각" in damage:
            notes.append("심각한 파손 - 부품 추출 후 업사이클")
        
        # 오염도별
        if soil == "깨끗":
            notes.append("오염 없음 - 기부 적합")
        elif soil == "약간 더러움":
            notes.append("약간 더러움 - 세척 후 기부 가능")
        elif soil == "더러움":
            notes.append("심한 오염 - 위생상 기부 불가")
        
        # 종합 판단
        if len(notes) > 0:
            return " | ".join(notes)
        else:
            return "기본적인 상태 확인 완료" 