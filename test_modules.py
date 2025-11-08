"""
테스트 스크립트
각 모듈의 기본 기능을 테스트합니다.
"""

import sys
import os

# 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.screen_capture import ScreenCapture
from src.game_vision import GameVision
from src.input_controller import InputController


def test_screen_capture():
    """화면 캡처 모듈 테스트"""
    print("=== 화면 캡처 테스트 ===")
    
    capture = ScreenCapture()
    
    # 윈도우 찾기 테스트
    if capture.find_window():
        print("✓ DOSBox 윈도우 발견")
        
        # 화면 캡처 테스트
        image = capture.capture_screen()
        if image is not None:
            print(f"✓ 화면 캡처 성공 - 이미지 크기: {image.shape}")
            return True
        else:
            print("❌ 화면 캡처 실패")
            return False
    else:
        print("❌ DOSBox 윈도우를 찾을 수 없습니다")
        return False


def test_game_vision():
    """게임 비전 모듈 테스트"""
    print("\n=== 게임 비전 테스트 ===")
    
    # 가상의 이미지로 테스트 (실제로는 화면 캡처 이미지 사용)
    import numpy as np
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    vision = GameVision()
    game_state = vision.analyze_game_state(test_image)
    
    print(f"✓ 게임 상태 분석 완료")
    print(f"  - 씬 타입: {game_state['scene_type']}")
    print(f"  - 캐릭터 발견: {game_state['character']['found']}")
    print(f"  - 미니맵 발견: {game_state['minimap']['found']}")
    print(f"  - 적 수: {len(game_state['enemies'])}")
    
    return True


def test_input_controller():
    """입력 제어 모듈 테스트"""
    print("\n=== 입력 제어 테스트 ===")
    
    controller = InputController()
    
    # 설정 로드 테스트
    print("✓ 입력 제어 모듈 초기화 완료")
    print(f"  - 키 지연시간: {controller.key_delay}초")
    print(f"  - 이동 키: {controller.movement_keys}")
    print(f"  - 액션 키: {controller.action_keys}")
    
    # 실제 키 입력은 테스트에서 제외 (게임이 실행 중이지 않을 수 있음)
    print("✓ 입력 제어 모듈 테스트 완료")
    
    return True


def main():
    """메인 테스트 함수"""
    print("영웅전설4 AI 플레이어 모듈 테스트")
    print("=" * 50)
    
    tests = [
        ("화면 캡처 모듈", test_screen_capture),
        ("게임 비전 모듈", test_game_vision),
        ("입력 제어 모듈", test_input_controller)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 오류: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("테스트 결과 요약")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n총 {len(results)}개 테스트 중 {passed}개 통과")
    
    if passed == len(results):
        print("\n🎉 모든 테스트가 통과했습니다!")
        print("이제 main.py를 실행하여 AI 플레이어를 시작할 수 있습니다.")
    else:
        print(f"\n⚠️ {len(results) - passed}개 테스트가 실패했습니다.")
        print("실패한 테스트를 확인하고 문제를 해결해주세요.")


if __name__ == "__main__":
    main()