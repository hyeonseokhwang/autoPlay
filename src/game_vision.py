"""
게임 화면 분석 모듈
캡처된 화면에서 게임 요소들을 인식하고 분석합니다.
"""

import cv2
import numpy as np
from PIL import Image
import json
import os
import time


class GameVision:
    def __init__(self, config_path="config/settings.json"):
        """
        게임 비전 분석 클래스 초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.game_resolution = self.config.get("game_settings", {}).get("game_resolution", [640, 480])
        self.ui_elements = self.config.get("game_settings", {}).get("ui_elements", {})
        
        # 템플릿 이미지들 (나중에 추가할 예정)
        self.templates = {}
        
    def _load_config(self, config_path):
        """설정 파일을 로드합니다."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"설정 파일을 찾을 수 없습니다: {config_path}")
            return {}
    
    def preprocess_image(self, image):
        """
        이미지 전처리를 수행합니다.
        
        Args:
            image (numpy.ndarray): 입력 이미지
            
        Returns:
            dict: 전처리된 이미지들
        """
        if image is None:
            return None
            
        processed = {}
        
        # 원본 이미지
        processed['original'] = image.copy()
        
        # 그레이스케일 변환
        processed['gray'] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # HSV 변환 (색상 기반 분석용)
        processed['hsv'] = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 엣지 검출
        processed['edges'] = cv2.Canny(processed['gray'], 50, 150)
        
        return processed
    
    def detect_character(self, image):
        """
        화면에서 캐릭터를 감지합니다.
        
        Args:
            image (numpy.ndarray): 분석할 이미지
            
        Returns:
            dict: 캐릭터 정보 (위치, 상태 등)
        """
        character_info = {
            'found': False,
            'position': None,
            'direction': None,
            'state': 'unknown'
        }
        
        if image is None:
            return character_info
        
        # 캐릭터 영역 추출
        char_area = self.ui_elements.get('character_area', [50, 50, 150, 150])
        x, y, w, h = char_area
        
        if x + w <= image.shape[1] and y + h <= image.shape[0]:
            character_region = image[y:y+h, x:x+w]
            
            # 여기서 실제 캐릭터 감지 로직을 구현
            # 현재는 기본값 반환
            character_info['found'] = True
            character_info['position'] = (x + w//2, y + h//2)
            
        return character_info
    
    def detect_minimap(self, image):
        """
        미니맵을 분석합니다.
        
        Args:
            image (numpy.ndarray): 분석할 이미지
            
        Returns:
            dict: 미니맵 정보
        """
        minimap_info = {
            'found': False,
            'player_position': None,
            'enemies': [],
            'items': [],
            'exits': []
        }
        
        if image is None:
            return minimap_info
        
        # 미니맵 영역 추출
        minimap_area = self.ui_elements.get('minimap_area', [500, 50, 600, 150])
        x, y, w, h = minimap_area
        
        if x + w <= image.shape[1] and y + h <= image.shape[0]:
            minimap_region = image[y:y+h, x:x+w]
            minimap_info['found'] = True
            
            # 미니맵 분석 로직 구현 예정
            
        return minimap_info
    
    def detect_ui_elements(self, image):
        """
        게임 UI 요소들을 감지합니다.
        
        Args:
            image (numpy.ndarray): 분석할 이미지
            
        Returns:
            dict: UI 요소 정보
        """
        ui_info = {
            'menu_open': False,
            'dialog_open': False,
            'inventory_open': False,
            'health_bar': None,
            'mana_bar': None
        }
        
        if image is None:
            return ui_info
        
        # 전처리
        processed = self.preprocess_image(image)
        gray = processed['gray']
        
        # 메뉴 감지 (색상 기반)
        # 예: 특정 색상 범위로 메뉴 창 감지
        
        # 대화창 감지
        # 예: 텍스트 박스 형태 감지
        
        # 상태바 분석
        status_area = self.ui_elements.get('status_bar', [0, 450, 640, 480])
        x, y, w, h = status_area
        
        if x + w <= image.shape[1] and y + h <= image.shape[0]:
            status_region = image[y:y+h, x:x+w]
            # 체력, 마나 바 분석 로직
            
        return ui_info
    
    def detect_enemies(self, image):
        """
        적 캐릭터들을 감지합니다.
        
        Args:
            image (numpy.ndarray): 분석할 이미지
            
        Returns:
            list: 적 캐릭터 정보 리스트
        """
        enemies = []
        
        if image is None:
            return enemies
        
        # 색상 기반 적 감지
        processed = self.preprocess_image(image)
        hsv = processed['hsv']
        
        # 적 캐릭터의 색상 범위 정의 (예시)
        # 실제 게임에서는 적 캐릭터의 색상을 분석해서 범위 설정
        enemy_color_ranges = [
            ([0, 50, 50], [10, 255, 255]),    # 빨간색 계열
            ([160, 50, 50], [180, 255, 255])  # 빨간색 계열 (HSV 순환)
        ]
        
        for lower, upper in enemy_color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # 최소 크기 필터링
                    x, y, w, h = cv2.boundingRect(contour)
                    enemies.append({
                        'position': (x + w//2, y + h//2),
                        'size': (w, h),
                        'area': area
                    })
        
        return enemies
    
    def detect_items(self, image):
        """
        아이템들을 감지합니다.
        
        Args:
            image (numpy.ndarray): 분석할 이미지
            
        Returns:
            list: 아이템 정보 리스트
        """
        items = []
        
        if image is None:
            return items
        
        # 아이템 감지 로직 구현 예정
        # 보통 반짝이는 효과나 특정 색상으로 구별됨
        
        return items
    
    def analyze_game_state(self, image):
        """
        게임 상태를 종합적으로 분석합니다.
        
        Args:
            image (numpy.ndarray): 분석할 이미지
            
        Returns:
            dict: 게임 상태 정보
        """
        game_state = {
            'timestamp': time.time(),
            'character': None,
            'minimap': None,
            'ui': None,
            'enemies': [],
            'items': [],
            'scene_type': 'unknown',  # field, town, dungeon, battle
            'is_battle': False,
            'is_field': False
        }
        
        if image is None:
            return game_state
        
        # 화면 타입 먼저 판별
        game_state['is_battle'] = self.detect_battle_screen(image)
        game_state['is_field'] = self.detect_field_screen(image)
        
        # 각 요소 분석
        game_state['character'] = self.detect_character(image)
        game_state['minimap'] = self.detect_minimap(image)
        game_state['ui'] = self.detect_ui_elements(image)
        game_state['enemies'] = self.detect_enemies(image)
        game_state['items'] = self.detect_items(image)
        
        # 씬 타입 결정 (개선된 로직)
        if game_state['is_battle']:
            game_state['scene_type'] = 'battle'
        elif game_state['is_field']:
            game_state['scene_type'] = 'field'
        else:
            game_state['scene_type'] = self._determine_scene_type(game_state)
        
        return game_state
    
    def _determine_scene_type(self, game_state):
        """
        게임 상태를 바탕으로 현재 씬 타입을 추정합니다.
        
        Args:
            game_state (dict): 게임 상태 정보
            
        Returns:
            str: 씬 타입 ('field', 'town', 'dungeon', 'battle', 'menu')
        """
        # UI 상태 확인
        if game_state['ui']['menu_open']:
            return 'menu'
        
        # 적이 있으면 전투 상황
        if len(game_state['enemies']) > 0:
            return 'battle'
        
        # 미니맵 정보로 판단
        if game_state['minimap']['found']:
            # 추가적인 분석 로직
            pass
        
        # 기본값
        return 'field'
    
    def detect_battle_screen(self, image):
        """
        전투 화면인지 감지합니다.
        
        Args:
            image (numpy.ndarray): 분석할 이미지
            
        Returns:
            bool: 전투 화면이면 True
        """
        if image is None:
            return False
        
        # 전투 화면 특징 감지
        # 1. 하단 UI가 있는지 확인 (캐릭터 상태창)
        height, width = image.shape[:2]
        bottom_area = image[int(height * 0.7):, :]  # 하단 30% 영역
        
        # HSV 변환
        hsv = cv2.cvtColor(bottom_area, cv2.COLOR_BGR2HSV)
        
        # 갈색/오렌지색 UI 프레임 감지 (영웅전설4의 UI 색상)
        brown_lower = np.array([10, 50, 50])
        brown_upper = np.array([25, 255, 255])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        
        brown_pixels = cv2.countNonZero(brown_mask)
        total_pixels = bottom_area.shape[0] * bottom_area.shape[1]
        brown_ratio = brown_pixels / total_pixels
        
        # UI 프레임이 충분히 있으면 전투 화면으로 판단
        return brown_ratio > 0.15  # 15% 이상이면 전투 화면
    
    def detect_field_screen(self, image):
        """
        필드(오픈월드) 화면인지 감지합니다.
        
        Args:
            image (numpy.ndarray): 분석할 이미지
            
        Returns:
            bool: 필드 화면이면 True
        """
        if image is None:
            return False
        
        # 필드 화면 특징 감지
        # 1. 상단 중앙 영역이 게임 필드인지 확인
        height, width = image.shape[:2]
        field_area = image[int(height * 0.1):int(height * 0.8), int(width * 0.1):int(width * 0.9)]
        
        # HSV 변환
        hsv = cv2.cvtColor(field_area, cv2.COLOR_BGR2HSV)
        
        # 녹색 계열 (숲, 풀) 감지
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # 갈색 계열 (땅, 길) 감지
        brown_lower = np.array([8, 50, 50])
        brown_upper = np.array([25, 255, 200])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        
        green_pixels = cv2.countNonZero(green_mask)
        brown_pixels = cv2.countNonZero(brown_mask)
        total_pixels = field_area.shape[0] * field_area.shape[1]
        
        nature_ratio = (green_pixels + brown_pixels) / total_pixels
        
        # 자연 요소가 충분히 있으면 필드 화면으로 판단
        return nature_ratio > 0.3  # 30% 이상이면 필드 화면


# 테스트 코드
if __name__ == "__main__":
    from screen_capture import ScreenCapture
    
    # 화면 캡처 및 분석 테스트
    capture = ScreenCapture()
    vision = GameVision()
    
    if capture.find_window():
        print("게임 화면 분석 시작...")
        
        image = capture.capture_screen()
        if image is not None:
            # 게임 상태 분석
            game_state = vision.analyze_game_state(image)
            
            print("=== 게임 상태 분석 결과 ===")
            print(f"씬 타입: {game_state['scene_type']}")
            print(f"캐릭터 발견: {game_state['character']['found']}")
            print(f"미니맵 발견: {game_state['minimap']['found']}")
            print(f"적 수: {len(game_state['enemies'])}")
            print(f"아이템 수: {len(game_state['items'])}")
            
            # 분석 결과를 이미지에 표시
            result_img = image.copy()
            
            # 캐릭터 위치 표시
            if game_state['character']['found'] and game_state['character']['position']:
                cv2.circle(result_img, game_state['character']['position'], 10, (0, 255, 0), 2)
            
            # 적 위치 표시
            for enemy in game_state['enemies']:
                cv2.circle(result_img, enemy['position'], 8, (0, 0, 255), 2)
            
            # 결과 표시
            cv2.imshow("게임 상태 분석 결과", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("게임 윈도우를 찾을 수 없습니다.")