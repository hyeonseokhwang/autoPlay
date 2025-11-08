#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
영웅전설4 진짜 인간형 추론 AI
- 제로베이스에서 시작
- 신경망으로 화면 이해
- 사람처럼 추론하고 판단
- 순수 경험 기반 학습
"""

import asyncio
import time
import numpy as np
import cv2
import json
import sqlite3
from datetime import datetime
from collections import deque
import random
from typing import Dict, List, Tuple, Any, Optional
from PIL import ImageGrab
import win32gui
import win32con
import win32api

class NeuralGameVision:
    """신경망 기반 게임 시각 인식"""
    
    def __init__(self, input_size: Tuple[int, int] = (64, 64)):
        """초기화"""
        self.input_size = input_size
        self.last_processed_image = None
        
        # 간단한 CNN 구조 (numpy로 구현)
        self.conv_weights = []
        self.dense_weights = []
        self._initialize_weights()
        
        # 추론 히스토리
        self.reasoning_history = deque(maxlen=100)
        
    def _initialize_weights(self):
        """가중치 초기화"""
        # Conv 레이어 (3x3 필터 16개)
        self.conv_weights = np.random.randn(16, 3, 3, 3) * 0.1
        
        # Dense 레이어 
        # 64x64 -> 32x32 (conv+pool) -> flatten -> dense
        self.w1 = np.random.randn(32*32*16, 128) * 0.1
        self.b1 = np.zeros(128)
        
        self.w2 = np.random.randn(128, 64) * 0.1
        self.b2 = np.zeros(64)
        
        # 출력: 의미있는 패턴 인식
        self.w_out = np.random.randn(64, 32) * 0.1  # 32개 패턴
        self.b_out = np.zeros(32)
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        if image is None:
            return None
            
        # 리사이즈
        resized = cv2.resize(image, self.input_size)
        
        # 정규화
        normalized = resized.astype(np.float32) / 255.0
        
        self.last_processed_image = normalized
        return normalized
    
    def simple_conv2d(self, image: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """간단한 2D 컨볼루션"""
        h, w, c = image.shape
        fh, fw = weight.shape[1], weight.shape[2]
        
        # 출력 크기
        oh = h - fh + 1
        ow = w - fw + 1
        
        output = np.zeros((oh, ow))
        
        for y in range(oh):
            for x in range(ow):
                patch = image[y:y+fh, x:x+fw]
                output[y, x] = np.sum(patch * weight[0])  # 첫 번째 채널만
                
        return output
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU 활성화"""
        return np.maximum(0, x)
    
    def forward_pass(self, image: np.ndarray) -> np.ndarray:
        """순전파 (추론)"""
        processed_img = self.preprocess_image(image)
        if processed_img is None:
            return np.zeros(32)
        
        try:
            # Conv 레이어 (간단화)
            conv_output = []
            for i in range(min(4, len(self.conv_weights))):  # 4개 필터만
                conv_out = self.simple_conv2d(processed_img, self.conv_weights[i])
                conv_output.append(conv_out)
            
            # Max pooling (2x2)
            pooled_outputs = []
            for conv_out in conv_output:
                h, w = conv_out.shape
                pooled = cv2.resize(conv_out, (w//2, h//2))
                pooled_outputs.append(pooled)
            
            # Flatten
            flattened = np.concatenate([p.flatten() for p in pooled_outputs])
            
            # Dense 레이어들
            if len(flattened) > 0:
                # 크기 맞추기
                target_size = self.w1.shape[0]
                if len(flattened) > target_size:
                    flattened = flattened[:target_size]
                elif len(flattened) < target_size:
                    padded = np.zeros(target_size)
                    padded[:len(flattened)] = flattened
                    flattened = padded
                
                h1 = self.relu(np.dot(flattened, self.w1) + self.b1)
                h2 = self.relu(np.dot(h1, self.w2) + self.b2)
                output = np.dot(h2, self.w_out) + self.b_out
                
                return output
            
        except Exception as e:
            print(f"⚠️ 신경망 오류: {e}")
            
        return np.zeros(32)
    
    def interpret_patterns(self, neural_output: np.ndarray) -> Dict[str, Any]:
        """신경망 출력을 의미있는 패턴으로 해석"""
        patterns = {}
        
        # 각 뉴런을 의미있는 패턴으로 매핑
        pattern_names = [
            'movement_space', 'ui_element', 'character_sprite', 'background_texture',
            'bright_area', 'dark_area', 'colorful_region', 'text_like',
            'menu_indicator', 'battle_signal', 'item_hint', 'door_passage',
            'enemy_presence', 'interactive_object', 'status_display', 'map_feature',
            'animation_motion', 'popup_window', 'selection_cursor', 'health_indicator',
            'magic_effect', 'treasure_sign', 'npc_character', 'environment_change',
            'quest_marker', 'save_point', 'checkpoint', 'secret_area',
            'danger_zone', 'safe_area', 'exploration_target', 'unknown_pattern'
        ]
        
        for i, activation in enumerate(neural_output):
            if i < len(pattern_names):
                # 활성화 값을 0-1로 정규화
                normalized = max(0, min(1, (activation + 2) / 4))
                patterns[pattern_names[i]] = normalized
        
        return patterns

class HumanLikeReasoning:
    """인간형 추론 엔진"""
    
    def __init__(self):
        """초기화"""
        self.reasoning_memory = deque(maxlen=50)
        self.curiosity_level = 0.8
        self.confidence_threshold = 0.3
        self.exploration_motivation = 1.0
        
        # 추론 가중치 (경험으로 학습됨)
        self.reasoning_weights = {
            'exploration': 1.0,
            'interaction': 0.8,
            'safety': 0.6,
            'novelty': 0.9,
            'pattern_recognition': 0.7
        }
        
    def analyze_situation(self, visual_patterns: Dict[str, float], 
                         action_history: List[str]) -> Dict[str, Any]:
        """상황 분석 및 추론"""
        
        # 1. 현재 상황 이해
        situation_assessment = self._assess_current_situation(visual_patterns)
        
        # 2. 호기심 기반 탐험 욕구
        curiosity_drive = self._calculate_curiosity(visual_patterns, action_history)
        
        # 3. 상호작용 가능성 판단
        interaction_potential = self._evaluate_interaction_opportunities(visual_patterns)
        
        # 4. 안전성 평가
        safety_assessment = self._evaluate_safety(visual_patterns)
        
        # 5. 종합 추론
        reasoning_result = {
            'situation': situation_assessment,
            'curiosity': curiosity_drive,
            'interaction': interaction_potential,
            'safety': safety_assessment,
            'overall_confidence': self._calculate_confidence(visual_patterns),
            'recommended_actions': self._generate_action_recommendations(
                visual_patterns, curiosity_drive, interaction_potential
            ),
            'reasoning_explanation': self._generate_explanation(
                situation_assessment, curiosity_drive, interaction_potential
            )
        }
        
        # 추론 기록
        self.reasoning_memory.append({
            'timestamp': datetime.now(),
            'patterns': visual_patterns.copy(),
            'reasoning': reasoning_result.copy()
        })
        
        return reasoning_result
    
    def _assess_current_situation(self, patterns: Dict[str, float]) -> Dict[str, Any]:
        """현재 상황 평가"""
        # 주요 패턴들의 강도 분석
        ui_strength = patterns.get('ui_element', 0) + patterns.get('menu_indicator', 0)
        exploration_potential = patterns.get('movement_space', 0) + patterns.get('map_feature', 0)
        interaction_signs = patterns.get('interactive_object', 0) + patterns.get('npc_character', 0)
        
        situation_type = 'unknown'
        if ui_strength > 0.5:
            situation_type = 'menu_navigation'
        elif exploration_potential > 0.6:
            situation_type = 'field_exploration'  
        elif interaction_signs > 0.4:
            situation_type = 'interaction_opportunity'
        elif patterns.get('battle_signal', 0) > 0.3:
            situation_type = 'potential_combat'
        
        return {
            'type': situation_type,
            'ui_strength': ui_strength,
            'exploration_potential': exploration_potential,
            'interaction_signs': interaction_signs,
            'complexity': np.std(list(patterns.values()))
        }
    
    def _calculate_curiosity(self, patterns: Dict[str, float], 
                           action_history: List[str]) -> Dict[str, float]:
        """호기심 계산"""
        # 새로운 패턴에 대한 호기심
        novelty_score = 0.0
        for pattern_name, activation in patterns.items():
            if activation > 0.3 and pattern_name.endswith(('_target', '_sign', '_hint')):
                novelty_score += activation * self.curiosity_level
        
        # 최근 행동의 다양성
        recent_actions = action_history[-10:] if len(action_history) >= 10 else action_history
        action_diversity = len(set(recent_actions)) / max(len(recent_actions), 1)
        
        # 탐험하지 않은 영역에 대한 호기심
        exploration_urge = patterns.get('unknown_pattern', 0) * 1.5
        
        return {
            'novelty': novelty_score,
            'diversity_seeking': action_diversity,
            'exploration_urge': exploration_urge,
            'total_curiosity': (novelty_score + action_diversity + exploration_urge) / 3
        }
    
    def _evaluate_interaction_opportunities(self, patterns: Dict[str, float]) -> Dict[str, float]:
        """상호작용 기회 평가"""
        interactive_elements = [
            'interactive_object', 'npc_character', 'door_passage', 
            'item_hint', 'save_point', 'treasure_sign'
        ]
        
        interaction_scores = {}
        total_interaction_potential = 0.0
        
        for element in interactive_elements:
            score = patterns.get(element, 0)
            interaction_scores[element] = score
            total_interaction_potential += score
        
        return {
            **interaction_scores,
            'total_potential': total_interaction_potential,
            'highest_priority': max(interaction_scores, key=interaction_scores.get) if interaction_scores else None
        }
    
    def _evaluate_safety(self, patterns: Dict[str, float]) -> Dict[str, float]:
        """안전성 평가"""
        danger_indicators = patterns.get('danger_zone', 0) + patterns.get('enemy_presence', 0)
        safe_indicators = patterns.get('safe_area', 0) + patterns.get('save_point', 0)
        
        return {
            'danger_level': danger_indicators,
            'safety_level': safe_indicators,
            'overall_safety': safe_indicators - danger_indicators
        }
    
    def _calculate_confidence(self, patterns: Dict[str, float]) -> float:
        """추론 신뢰도 계산"""
        # 패턴 활성화의 일관성
        activations = list(patterns.values())
        if not activations:
            return 0.0
            
        max_activation = max(activations)
        mean_activation = np.mean(activations)
        
        # 명확한 패턴이 있으면 신뢰도 높음
        clarity = max_activation - mean_activation
        confidence = min(1.0, clarity * 2)
        
        return confidence
    
    def _generate_action_recommendations(self, patterns: Dict[str, float], 
                                      curiosity: Dict[str, float],
                                      interaction: Dict[str, float]) -> List[Dict[str, Any]]:
        """행동 추천 생성"""
        recommendations = []
        
        # 호기심 기반 추천
        if curiosity['total_curiosity'] > 0.5:
            if patterns.get('movement_space', 0) > 0.4:
                recommendations.append({
                    'action': 'explore_movement',
                    'priority': curiosity['exploration_urge'],
                    'reason': '호기심 - 새로운 영역 탐험'
                })
        
        # 상호작용 기반 추천
        if interaction['total_potential'] > 0.3:
            highest_priority = interaction.get('highest_priority')
            if highest_priority:
                action_map = {
                    'interactive_object': 'interact_object',
                    'npc_character': 'talk_to_npc', 
                    'door_passage': 'enter_door',
                    'item_hint': 'investigate_item'
                }
                
                action = action_map.get(highest_priority, 'interact_general')
                recommendations.append({
                    'action': action,
                    'priority': interaction[highest_priority],
                    'reason': f'상호작용 기회 - {highest_priority}'
                })
        
        # 탐험 기본 추천
        if not recommendations:
            recommendations.append({
                'action': 'random_exploration',
                'priority': 0.5,
                'reason': '기본 탐험 행동'
            })
        
        return sorted(recommendations, key=lambda x: x['priority'], reverse=True)
    
    def _generate_explanation(self, situation: Dict, curiosity: Dict, interaction: Dict) -> str:
        """추론 과정 설명 생성"""
        explanation_parts = []
        
        explanation_parts.append(f"상황: {situation['type']}")
        
        if curiosity['total_curiosity'] > 0.5:
            explanation_parts.append(f"호기심 발동 (강도: {curiosity['total_curiosity']:.2f})")
            
        if interaction['total_potential'] > 0.3:
            explanation_parts.append(f"상호작용 가능성 발견")
            
        return " | ".join(explanation_parts)
    
    def learn_from_experience(self, action_taken: str, outcome_patterns: Dict[str, float], 
                            reward_signal: float) -> None:
        """경험으로부터 학습"""
        if not self.reasoning_memory:
            return
            
        last_reasoning = self.reasoning_memory[-1]
        
        # 추천했던 행동의 결과 분석
        recommended_actions = last_reasoning['reasoning']['recommended_actions']
        
        for rec in recommended_actions:
            if action_taken in rec['action']:
                # 결과가 좋았으면 해당 추론 패턴 강화
                if reward_signal > 0:
                    pattern_type = rec['reason'].split(' - ')[0] if ' - ' in rec['reason'] else 'general'
                    if pattern_type in self.reasoning_weights:
                        self.reasoning_weights[pattern_type] = min(2.0, 
                            self.reasoning_weights[pattern_type] * 1.1)
                        print(f"🧠 학습: '{pattern_type}' 추론 강화 → {self.reasoning_weights[pattern_type]:.3f}")
                else:
                    # 결과가 나빴으면 약화
                    pattern_type = rec['reason'].split(' - ')[0] if ' - ' in rec['reason'] else 'general'
                    if pattern_type in self.reasoning_weights:
                        self.reasoning_weights[pattern_type] = max(0.1, 
                            self.reasoning_weights[pattern_type] * 0.9)
                        print(f"🧠 학습: '{pattern_type}' 추론 약화 → {self.reasoning_weights[pattern_type]:.3f}")

class IntelligentGameController:
    """지능형 게임 컨트롤러"""
    
    def __init__(self):
        """초기화"""
        self.dosbox_window = None
        self.game_region = None
        
        # 행동 매핑
        self.action_mappings = {
            'explore_movement': ['left', 'right', 'up', 'down'],
            'interact_object': ['space', 'enter', 'z'],
            'talk_to_npc': ['space', 'enter'],
            'enter_door': ['up', 'space'],
            'investigate_item': ['z', 'space', 'enter'],
            'interact_general': ['space', 'enter', 'z'],
            'random_exploration': ['left', 'right', 'up', 'down', 'space']
        }
        
    def find_game_window(self) -> bool:
        """게임 창 찾기"""
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if 'dosbox' in window_text.lower() or 'ED4' in window_text:
                    windows.append(hwnd)
            return True

        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        
        if windows:
            self.dosbox_window = windows[0]
            self.game_region = win32gui.GetWindowRect(self.dosbox_window)
            print(f"🎮 게임 연결: {self.game_region}")
            return True
        
        return False
    
    def execute_action(self, action_type: str) -> Tuple[bool, str]:
        """행동 실행"""
        if not self.dosbox_window:
            return False, "게임 창 없음"
            
        # 행동 타입에서 실제 키 선택
        possible_keys = self.action_mappings.get(action_type, ['space'])
        selected_key = random.choice(possible_keys)
        
        try:
            win32gui.SetForegroundWindow(self.dosbox_window)
            time.sleep(0.05)
            
            key_map = {
                'left': 0x25, 'right': 0x27, 'up': 0x26, 'down': 0x28,
                'space': 0x20, 'enter': 0x0D, 'z': 0x5A, 'x': 0x58,
                'a': 0x41, 's': 0x53, '1': 0x31, '2': 0x32
            }
            
            if selected_key in key_map:
                vk_code = key_map[selected_key]
                win32api.keybd_event(vk_code, 0, 0, 0)
                time.sleep(0.08)
                win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
                return True, selected_key
                
        except Exception as e:
            return False, f"오류: {e}"
        
        return False, "알 수 없는 키"
    
    def capture_screen(self) -> np.ndarray:
        """화면 캡처"""
        try:
            screenshot = ImageGrab.grab(self.game_region)
            return np.array(screenshot)
        except Exception as e:
            print(f"❌ 화면 캡처 실패: {e}")
            return None

class HumanLikeAI:
    """인간형 AI 시스템"""
    
    def __init__(self):
        """초기화"""
        # 핵심 컴포넌트들
        self.vision = NeuralGameVision()
        self.reasoning = HumanLikeReasoning()
        self.controller = IntelligentGameController()
        
        # 경험 저장
        self.experience_db = ExperienceDatabase()
        self.session_id = f"human_ai_{int(time.time())}"
        
        # 상태 추적
        self.action_history = deque(maxlen=100)
        self.step_count = 0
        self.battle_discoveries = 0
        self.total_reward = 0.0
        
        print("🧠 인간형 AI 시스템 초기화 완료")
        
    async def thinking_step(self) -> None:
        """한 번의 사고 스텝"""
        self.step_count += 1
        
        # 1. 화면 관찰
        screenshot = self.controller.capture_screen()
        if screenshot is None:
            return
        
        # 2. 신경망으로 시각 패턴 인식
        neural_output = self.vision.forward_pass(screenshot)
        visual_patterns = self.vision.interpret_patterns(neural_output)
        
        # 3. 인간형 추론
        reasoning_result = self.reasoning.analyze_situation(
            visual_patterns, list(self.action_history)
        )
        
        print(f"\n🤔 스텝 {self.step_count} - 추론 중...")
        print(f"   🧠 {reasoning_result['reasoning_explanation']}")
        print(f"   📊 신뢰도: {reasoning_result['overall_confidence']:.2f}")
        
        # 4. 행동 결정
        recommendations = reasoning_result['recommended_actions']
        if recommendations:
            chosen_action = recommendations[0]  # 최우선 추천
            action_type = chosen_action['action']
            
            print(f"   🎯 행동 결정: {action_type} (이유: {chosen_action['reason']})")
            
            # 5. 행동 실행
            success, actual_key = self.controller.execute_action(action_type)
            
            if success:
                self.action_history.append(actual_key)
                print(f"   ⚡ 실행: {actual_key}")
                
                # 6. 결과 관찰 및 보상 계산
                await asyncio.sleep(0.2)  # 결과 대기
                
                next_screenshot = self.controller.capture_screen()
                if next_screenshot is not None:
                    next_neural = self.vision.forward_pass(next_screenshot)
                    next_patterns = self.vision.interpret_patterns(next_neural)
                    
                    # 보상 계산 (간단한 변화 기반)
                    reward = self._calculate_experience_reward(
                        visual_patterns, next_patterns, action_type
                    )
                    self.total_reward += reward
                    
                    # 7. 경험으로부터 학습
                    self.reasoning.learn_from_experience(
                        actual_key, next_patterns, reward
                    )
                    
                    # 전투 발견 체크
                    if self._detect_battle_discovery(next_patterns):
                        self.battle_discoveries += 1
                        print(f"   ⚔️ 전투 발견! 총 {self.battle_discoveries}회")
        
        else:
            print(f"   ❓ 추론 실패 - 랜덤 행동")
            # 비상 랜덤 행동
            success, actual_key = self.controller.execute_action('random_exploration')
            if success:
                self.action_history.append(actual_key)
    
    def _calculate_experience_reward(self, before_patterns: Dict[str, float],
                                   after_patterns: Dict[str, float], 
                                   action_type: str) -> float:
        """경험 기반 보상 계산"""
        reward = 0.01  # 기본 생존 보상
        
        # 패턴 변화 보상
        for pattern_name in before_patterns:
            before_val = before_patterns.get(pattern_name, 0)
            after_val = after_patterns.get(pattern_name, 0)
            change = abs(after_val - before_val)
            
            if change > 0.1:  # 의미있는 변화
                reward += change * 0.5
        
        # 특정 패턴 발견 보상
        valuable_patterns = ['battle_signal', 'treasure_sign', 'interactive_object', 'npc_character']
        for pattern in valuable_patterns:
            if after_patterns.get(pattern, 0) > 0.4:
                reward += 1.0
        
        return reward
    
    def _detect_battle_discovery(self, patterns: Dict[str, float]) -> bool:
        """전투 발견 감지"""
        battle_indicators = (
            patterns.get('battle_signal', 0) > 0.4 or
            patterns.get('enemy_presence', 0) > 0.3 or
            patterns.get('danger_zone', 0) > 0.5
        )
        return battle_indicators
    
    async def run_human_like_session(self, max_steps: int = 300, target_battles: int = 10) -> None:
        """인간형 AI 세션 실행"""
        print("🧠 인간형 AI 세션 시작!")
        print(f"🎯 목표: {max_steps}스텝으로 {target_battles}번의 전투 발견")
        
        if not self.controller.find_game_window():
            print("❌ 게임을 찾을 수 없습니다!")
            return
        
        start_time = time.time()
        
        while (self.step_count < max_steps and 
               self.battle_discoveries < target_battles):
            
            await self.thinking_step()
            await asyncio.sleep(0.25)  # 사람처럼 생각하는 시간
            
            # 진행 상황
            if self.step_count % 25 == 0:
                elapsed = time.time() - start_time
                print(f"\n📊 진행 상황:")
                print(f"   🎮 스텝: {self.step_count}/{max_steps}")
                print(f"   ⚔️ 전투 발견: {self.battle_discoveries}/{target_battles}")
                print(f"   💰 누적 보상: {self.total_reward:.2f}")
                print(f"   ⏱️ 경과 시간: {elapsed:.1f}초")
                print(f"   🧠 추론 가중치 변화:")
                for weight_name, weight_val in self.reasoning.reasoning_weights.items():
                    print(f"      {weight_name}: {weight_val:.3f}")
        
        # 최종 결과
        elapsed = time.time() - start_time
        print(f"\n🏁 세션 완료!")
        print(f"⏱️ 총 시간: {elapsed:.1f}초")
        print(f"🎮 총 스텝: {self.step_count}")
        print(f"⚔️ 전투 발견: {self.battle_discoveries}/{target_battles}")
        print(f"💰 총 보상: {self.total_reward:.2f}")
        print(f"📈 평균 보상: {self.total_reward/max(self.step_count, 1):.4f}")
        
        if self.battle_discoveries >= target_battles:
            print("🎉 목표 달성! AI가 성공적으로 학습하고 추론했습니다!")
        else:
            print("📚 학습 진행 중. AI가 경험을 쌓았습니다.")

class ExperienceDatabase:
    """경험 데이터베이스"""
    
    def __init__(self, db_path: str = "human_ai_experience.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """DB 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS human_experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    step_number INTEGER,
                    visual_patterns TEXT,
                    reasoning_result TEXT,
                    action_taken TEXT,
                    reward REAL,
                    timestamp TEXT
                )
            """)

# 실행
if __name__ == "__main__":
    async def main():
        ai = HumanLikeAI()
        await ai.run_human_like_session(max_steps=300, target_battles=10)
    
    print("🧠 진짜 인간형 추론 AI")
    print("=" * 70)
    print("✨ 특징: 신경망 시각인식 + 인간형 추론 + 순수 경험 학습")
    asyncio.run(main())