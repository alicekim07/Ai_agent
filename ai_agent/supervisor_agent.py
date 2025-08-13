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

    def process(self, front_image, left_image):
        # 0. 각 이미지 크기 최적화
        optimized_front = optimize_image_size(front_image)
        optimized_left = optimize_image_size(left_image)
        
        # 1. 각 개별 에이전트를 병렬로 실행 (2개 이미지 사용)
        print("개별 에이전트 분석 중... (2개 이미지 통합 분석)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # 4개 에이전트를 동시에 실행, 각각 2개 이미지 전달
            future_type = executor.submit(self.type_agent.analyze, optimized_front, optimized_left)
            future_material = executor.submit(self.material_agent.analyze, optimized_front, optimized_left)
            future_damage = executor.submit(self.damage_agent.analyze, optimized_front, optimized_left)
            future_soil = executor.submit(self.soil_agent.analyze, optimized_front, optimized_left)
            
            # 결과 수집 (타임아웃 30초)
            try:
                type_response, type_tokens = future_type.result(timeout=30)
                material_response, material_tokens = future_material.result(timeout=30)
                damage_response, damage_tokens = future_damage.result(timeout=30)
                soil_response, soil_tokens = future_soil.result(timeout=30)
            except concurrent.futures.TimeoutError:
                print("일부 에이전트가 타임아웃되었습니다. 순차 처리로 전환합니다.")
                # 타임아웃 시 순차 처리로 전환
                type_response, type_tokens = self.type_agent.analyze(optimized_front, optimized_left)
                material_response, material_tokens = self.material_agent.analyze(optimized_front, optimized_left)
                damage_response, damage_tokens = self.damage_agent.analyze(optimized_front, optimized_left)
                soil_response, soil_tokens = self.soil_agent.analyze(optimized_front, optimized_left)
        
        # 3. JSON 파싱
        try:
            type_result = json.loads(type_response)
        except:
            type_result = {"type": "기타", "battery": "불명"}
            
        try:
            material_result = json.loads(material_response)
        except:
            material_result = {"material": "불명", "material_detail": "불명"}
            
        try:
            damage_result = json.loads(damage_response)
        except:
            damage_result = {"damage": "불명", "damage_detail": "불명", "missing_parts": "불명"}
            
        try:
            soil_result = json.loads(soil_response)
        except:
            soil_result = {"soil": "깨끗", "soil_detail": "오염 없음"}

        # 4. 통합 판단 로직 적용
        toy_type = type_result.get("type", "기타")
        battery = type_result.get("battery", "불명")
        size = type_result.get("size", "중간")  # AI 분석 결과 사용
        material = material_result.get("material", "불명")
        material_detail = material_result.get("material_detail", "불명")
        material_confidence = material_result.get("confidence", "불명")
        material_notes = material_result.get("notes", "")
        damage = damage_result.get("damage", "불명")
        damage_detail = damage_result.get("damage_detail", "불명")
        missing_parts = damage_result.get("missing_parts", "불명")
        soil = soil_result.get("soil", "깨끗")
        soil_detail = soil_result.get("soil_detail", "오염 없음")
        
        # 의미있는 관찰사항 생성
        notes = self._generate_meaningful_notes(toy_type, battery, material, material_detail, damage, damage_detail, missing_parts, soil, soil_detail)
        
        # Material Agent의 추가 정보를 관찰사항에 포함
        if material_notes and material_notes not in notes:
            notes = f"{notes} | 재료 분석: {material_notes}" if notes else f"재료 분석: {material_notes}"
        
        # 5. 기부 판단 로직 (통합 에이전트와 동일한 규칙)
        donate, donate_reason, repair_or_disassemble = self._judge_donation(
            toy_type, battery, material, material_detail, damage, damage_detail, missing_parts, soil, soil_detail
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

    def _judge_donation(self, toy_type, battery, material, material_detail, damage, damage_detail, missing_parts, soil, soil_detail):
        """
        가중치 기반 기부 가능 여부 판단 시스템
        소재(40%) + 부품상태(30%) + 파손(20%) + 오염도(10%) 순으로 평가
        """
        
        # 1. 즉시 기부 불가 조건들 (가중치 100%)
        if material == "나무":
            return False, "나무 소재는 안전상 기부 불가", "분해/부품 추출(업사이클)"
            
        # 천/섬유 소재 감지 강화
        if ("천" in material or "섬유" in material or 
            material == "섬유" or "섬유" in material_detail or
            "천" in str(material_detail) or "섬유" in str(material_detail)):
            return False, "천/섬유 소재는 위생상 기부 불가", "분해/부품 추출(업사이클)"
            
        # 실리콘 소재 감지 강화
        if ("실리콘" in material or "실리콘" in str(material_detail) or
            "고무" in material or "고무" in str(material_detail)):
            return False, "실리콘/고무 소재는 재활용 불가능", "분해/부품 추출(업사이클)"
            
        # 혼합 소재 판단 기준 완화 - 플라스틱+금속은 허용, 위생상 문제가 있는 소재만 제한
        if (("천" in material or "섬유" in material) and 
            ("," in material or "혼합" in material_detail or "혼합" in material)):
            return False, "천/섬유가 포함된 혼합 소재는 위생상 기부 불가", "분해/부품 추출(업사이클)"
            
        if (("실리콘" in material or "고무" in material) and 
            ("," in material or "혼합" in material_detail or "혼합" in material)):
            return False, "실리콘/고무가 포함된 혼합 소재는 재활용 불가능", "분해/부품 추출(업사이클)"
            
        # 플라스틱+금속 조합은 기부 가능하도록 허용 (금속 부품이 주요 기능에 영향을 주지 않는 경우)
        # 다른 혼합 소재는 점수 계산에서 처리
            
        if "심각" in damage or damage == "심각한 파손" or "심각" in damage_detail:
            return False, "심각한 파손으로 완제품 기부 불가", "분해/부품 추출(업사이클)"
            
        if soil == "더러움" or "더러움" in soil or "재활용 불가" in str(soil_detail):
            return False, "오염 상태로 위생상 기부 불가", "분해/부품 추출(업사이클)"
            
        if toy_type in ["인형", "아동 도서", "보행기", "탈것"]:
            return False, f"{toy_type}은 기부 불가 종류", "분해/부품 추출(업사이클)"
            
        # 특수 케이스 추가: 장난감부품, 용도 불분명
        if ("부품" in str(toy_type) or "용도 불분명" in str(toy_type) or 
            "불분명" in str(toy_type)):
            return False, f"{toy_type}은 기부 불가", "분해/부품 추출(업사이클)"
            
        # 2. 가중치 기반 점수 계산
        score = 0
        max_score = 100
        
        # 소재 점수 (40점 만점)
        if material == "플라스틱" and "단일" in material_detail:
            score += 40
        elif material == "금속" and "단일" in material_detail:
            score += 35
        elif "플라스틱,금속" in material or ("플라스틱" in material and "금속" in material):
            score += 35  # 플라스틱+금속은 재활용 용이하여 높은 점수 (단일 플라스틱과 거의 동등)
        elif "플라스틱,천" in material or "플라스틱,섬유" in material or ("플라스틱" in material and ("천" in material or "섬유" in material)):
            score += 0   # 플라스틱+천/섬유는 위생상 문제로 0점
        elif "플라스틱,실리콘" in material or ("플라스틱" in material and "실리콘" in material):
            score += 25   # 플라스틱+실리콘은 소량 부품 포함 시 기부 가능 (그립, 쿠션 등)
        elif "혼합" in material_detail or "," in material:
            score += 20  # 기타 혼합 소재는 중간 점수 (플라스틱+금속 등)
        else:
            score += 20
            
        # 부품 상태 점수 (40점 만점) - 부품 누락 감지 강화
        if missing_parts == "있음" or "부품" in damage or "일부" in damage:
            score += 0  # 부품 누락은 즉시 0점
        elif missing_parts == "불명" or "판단 불가" in damage:
            score += 5   # 판단 불가는 매우 낮은 점수
        elif missing_parts == "없음" and damage == "없음":
            score += 40  # 부품 완전 + 파손 없음 = 최고 점수
        elif missing_parts == "없음" and ("미세" in damage or "경미" in damage):
            score += 25  # 부품 완전 + 경미한 파손 = 높은 점수
        else:
            score += 15  # 기본 점수
            
        # 파손 상태 점수 (20점 만점)
        if damage == "없음":
            score += 20
        elif "미세" in damage or "경미" in damage or "미세" in damage_detail or "경미" in damage_detail:
            score += 15  # 경미한 파손은 높은 점수 (수리 후 기부 가능)
        elif "파손" in damage or "부서" in damage or "파손" in damage_detail or "부서" in damage_detail:
            score += 5   # 파손이 있지만 기부 가능한 수준
        else:
            score += 10
            
        # 오염도 점수 (10점 만점)
        if soil == "깨끗" or "깨끗" in soil or "깨끗" in soil_detail:
            score += 10
        elif soil == "보통" or "약간 더러움" in soil or "보통" in soil_detail:
            score += 5
        else:
            score += 0
            
        # 3. 특수 케이스 보정
        # 용도 불분명한 경우
        if "용도 불분명" in str(toy_type) or "불분명" in str(toy_type):
            score -= 30
            
        # 완제품이 아닌 경우
        if "부품" in str(toy_type) or "완제품" in str(toy_type):
            score -= 25
            
        # 건전지 장난감의 경우 (전자부품 상태 중요)
        if battery == "건전지" and (missing_parts == "있음" or "부품" in damage):
            score -= 20
            
        # 4. 최종 판단
        if score >= 75:  # 80에서 75로 낮춤
            if battery == "건전지":
                return True, "플라스틱 건전지 장난감, 상태 양호", "수리 불필요(완제품)"
            else:
                return True, "플라스틱 비건전지 장난감, 상태 양호", "수리 불필요(완제품)"
        elif score >= 55:  # 60에서 55로 낮춤
            return False, "경미한 문제로 수리 필요", "경미 수리 권장"
        elif score >= 35:  # 40에서 35로 낮춤
            return False, "여러 문제로 기부 어려움", "분해/부품 추출(업사이클)"
        else:
            return False, "심각한 문제로 기부 불가", "분해/부품 추출(업사이클)"

    def _generate_meaningful_notes(self, toy_type, battery, material, material_detail, damage, damage_detail, missing_parts, soil, soil_detail):
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
        if material == "플라스틱" and "단일" in material_detail:
            notes.append("플라스틱 소재는 균열이나 변형 확인")
        elif material == "금속" and "단일" in material_detail:
            notes.append("금속 소재는 녹이나 변형 상태 확인")
        elif material == "나무":
            notes.append("나무 소재는 균열이나 부식 상태 확인")
        elif "혼합" in material_detail or "," in material:
            notes.append("혼합 소재는 각 재료별 상태 확인 필요")
        elif "섬유" in material_detail:
            notes.append("섬유 소재는 위생상태와 마모 확인")
        
        # 파손 상태별
        if damage == "없음" and missing_parts == "없음":
            notes.append("파손 없음, 부품 완전 - 기부 적합")
        elif missing_parts == "있음":
            notes.append("부품 누락 - 부품 보완 후 기부 가능")
        elif missing_parts == "불명":
            notes.append("부품 상태 불명 - 추가 확인 필요")
        elif "미세" in damage or "경미" in damage or "미세" in damage_detail or "경미" in damage_detail:
            notes.append("경미한 파손 - 수리 후 기부 가능")
        elif "심각" in damage or "심각" in damage_detail:
            notes.append("심각한 파손 - 부품 추출 후 업사이클")
        
        # 오염도별
        if soil == "깨끗" or "깨끗" in soil or "깨끗" in soil_detail:
            notes.append("오염 없음 - 기부 적합")
        elif soil == "보통" or "보통" in soil_detail:
            notes.append("약간의 사용 흔적 - 세척 후 기부 가능")
        elif soil == "약간 더러움" or "약간 더러움" in soil_detail:
            notes.append("약간 더러움 - 세척 후 기부 가능")
        elif soil == "더러움" or "더러움" in soil or "더러움" in soil_detail:
            notes.append("심한 오염 - 위생상 기부 불가")
        
        # 종합 판단
        if len(notes) > 0:
            return " | ".join(notes)
        else:
            return "기본적인 상태 확인 완료" 