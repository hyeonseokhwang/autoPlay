"""
영웅전설4 AI 플레이어 메인 실행 파일
"""

import sys
import os
import time

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.screen_capture import ScreenCapture
from src.game_vision import GameVision
from src.input_controller import InputController
from src.ai_player import AIPlayer


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("   영웅전설4 AI 플레이어")
    print("=" * 50)
    print()
    
    # 시스템 요구사항 확인
    print("시스템 요구사항 확인 중...")
    
    try:
        # 모듈 초기화
        print("모듈 초기화 중...")
        screen_capture = ScreenCapture("DOSBox")
        game_vision = GameVision("config/settings.json")
        input_controller = InputController("config/settings.json")
        
        print("✓ 모듈 초기화 완료")
        
        # DOSBox 윈도우 찾기
        print("\nDOSBox 윈도우 검색 중...")
        if not screen_capture.find_window():
            print("❌ DOSBox 윈도우를 찾을 수 없습니다.")
            print("\n다음 사항을 확인해주세요:")
            print("1. DOSBox가 실행되어 있는지 확인")
            print("2. 영웅전설4 게임이 로드되어 있는지 확인")
            print("3. 윈도우 제목에 'DOSBox'가 포함되어 있는지 확인")
            return False
        
        print("✓ DOSBox 윈도우 발견")
        
        # 화면 캡처 테스트
        print("\n화면 캡처 테스트 중...")
        test_image = screen_capture.capture_screen()
        if test_image is None:
            print("❌ 화면 캡처 실패")
            return False
        
        print("✓ 화면 캡처 성공")
        
        # 게임 상태 분석 테스트
        print("\n게임 화면 분석 테스트 중...")
        game_state = game_vision.analyze_game_state(test_image)
        print(f"✓ 게임 씬 타입: {game_state['scene_type']}")
        print(f"✓ 캐릭터 감지: {game_state['character']['found']}")
        
        # AI 플레이어 생성
        print("\nAI 플레이어 초기화 중...")
        ai_player = AIPlayer(screen_capture, game_vision, input_controller)
        print("✓ AI 플레이어 초기화 완료")
        
        # 사용자 확인
        print("\n" + "=" * 50)
        print("AI 플레이어가 준비되었습니다!")
        print("=" * 50)
        print("\n주의사항:")
        print("- AI가 게임을 자동으로 플레이합니다")
        print("- 언제든지 Ctrl+C로 중지할 수 있습니다")
        print("- DOSBox 윈도우가 활성화됩니다")
        print()
        
        response = input("AI 플레이어를 시작하시겠습니까? (y/N): ").strip().lower()
        if response not in ['y', 'yes', '예']:
            print("사용자가 취소했습니다.")
            return True
        
        print("\nAI 플레이어 시작...")
        print("중지하려면 Ctrl+C를 누르세요.")
        print()
        
        # 카운트다운
        for i in range(5, 0, -1):
            print(f"시작까지 {i}초...")
            time.sleep(1)
        
        print("AI 플레이어 시작!")
        
        # AI 플레이어 실행
        ai_player.start()
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n사용자가 중단했습니다.")
        return True
    except ImportError as e:
        print(f"❌ 필수 모듈이 설치되지 않았습니다: {e}")
        print("\n다음 명령으로 필요한 패키지를 설치해주세요:")
        print("pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")
        return False


if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n프로그램을 종료합니다.")
        input("계속하려면 Enter를 누르세요...")
        sys.exit(1)
    
    print("\n프로그램을 종료합니다.")
    print("감사합니다!")