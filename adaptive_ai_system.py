"""
강화학습 기반 자율 진화 AI
DQN (Deep Q-Network)을 사용한 게임 AI 학습 시스템
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import cv2

class DQNetwork(nn.Module):
    """Deep Q-Network 모델"""
    
    def __init__(self, input_shape=(84, 84, 4), num_actions=8):
        super(DQNetwork, self).__init__()
        
        # CNN 레이어들
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Fully Connected 레이어들
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class ReinforcementLearningAI:
    """강화학습 기반 게임 AI"""
    
    def __init__(self, num_actions=8):
        """
        Actions:
        0: move_left, 1: move_right, 2: move_up, 3: move_down
        4: attack, 5: defend, 6: wait, 7: retreat
        """
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DQN 네트워크
        self.q_network = DQNetwork(num_actions=num_actions).to(self.device)
        self.target_network = DQNetwork(num_actions=num_actions).to(self.device)
        
        # 옵티마이저
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)
        
        # 경험 리플레이 버퍼
        self.replay_buffer = deque(maxlen=50000)
        
        # 하이퍼파라미터
        self.batch_size = 32
        self.gamma = 0.99  # 할인 인수
        self.epsilon = 1.0  # 탐험률
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # 학습 관련
        self.learn_step = 0
        self.target_update_frequency = 1000
        
        # 프레임 스택 (상태 이력)
        self.frame_stack = deque(maxlen=4)
        
        # 보상 체계
        self.reward_system = {
            "battle_win": 100,
            "battle_loss": -50,
            "find_enemy": 10,
            "explore_new_area": 5,
            "idle": -1,
            "death": -100
        }
    
    def preprocess_state(self, screen):
        """화면을 신경망 입력으로 전처리"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        # 크기 조정
        resized = cv2.resize(gray, (84, 84))
        
        # 정규화
        normalized = resized / 255.0
        
        return normalized
    
    def get_state(self, screen):
        """현재 상태 생성 (4프레임 스택)"""
        processed_frame = self.preprocess_state(screen)
        
        # 첫 번째 프레임인 경우 4번 복사
        if len(self.frame_stack) == 0:
            for _ in range(4):
                self.frame_stack.append(processed_frame)
        else:
            self.frame_stack.append(processed_frame)
        
        # 4개 프레임을 스택으로 합침
        state = np.stack(self.frame_stack, axis=0)
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def select_action(self, state, training=True):
        """행동 선택 (ε-greedy 정책)"""
        if training and random.random() < self.epsilon:
            # 랜덤 탐험
            return random.randrange(self.num_actions)
        
        # Q-값 기반 선택
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.max(1)[1].item()
    
    def calculate_reward(self, game_state_before, game_state_after, action):
        """보상 계산"""
        reward = 0
        
        # 전투 관련 보상
        if game_state_before.get("hp", 0) > game_state_after.get("hp", 0):
            reward -= 10  # HP 감소 패널티
        
        if not game_state_before.get("is_battle") and game_state_after.get("is_battle"):
            reward += self.reward_system["find_enemy"]  # 전투 발견 보상
        
        # 탐험 보상 (새로운 영역)
        if self.is_new_area(game_state_after):
            reward += self.reward_system["explore_new_area"]
        
        # 아무것도 하지 않음 패널티
        if action == 6:  # wait
            reward += self.reward_system["idle"]
        
        return reward
    
    def is_new_area(self, game_state):
        """새로운 영역인지 확인 (간단한 구현)"""
        # 실제로는 화면 해시나 위치 정보를 사용
        return random.random() < 0.1  # 10% 확률로 새 영역으로 간주
    
    def store_experience(self, state, action, reward, next_state, done):
        """경험을 리플레이 버퍼에 저장"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        """신경망 학습"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 미니배치 샘플링
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 현재 Q값
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 다음 상태의 최대 Q값 (타겟 네트워크 사용)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 손실 계산 및 역전파
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 타겟 네트워크 업데이트
        self.learn_step += 1
        if self.learn_step % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # ε 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, path):
        """모델 저장"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step': self.learn_step
        }, path)
    
    def load_model(self, path):
        """모델 로드"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.learn_step = checkpoint['learn_step']


class EvolutionaryGameAI:
    """진화형 게임 AI (유전 알고리즘 기반)"""
    
    def __init__(self, population_size=20):
        self.population_size = population_size
        self.generation = 0
        
        # 개체군 (각 개체는 행동 전략을 나타냄)
        self.population = self.initialize_population()
        
        # 적응도 점수
        self.fitness_scores = [0] * population_size
    
    def initialize_population(self):
        """초기 개체군 생성"""
        population = []
        for _ in range(self.population_size):
            # 각 개체는 상황별 행동 전략을 가짐
            individual = {
                "battle_strategy": self.random_strategy(),
                "exploration_strategy": self.random_strategy(),
                "survival_strategy": self.random_strategy()
            }
            population.append(individual)
        return population
    
    def random_strategy(self):
        """랜덤 전략 생성"""
        return {
            "aggression": random.uniform(0, 1),
            "caution": random.uniform(0, 1),
            "exploration": random.uniform(0, 1),
            "patience": random.uniform(0, 1)
        }
    
    def select_action_evolutionary(self, individual, game_state):
        """진화된 개체의 전략에 따른 행동 선택"""
        if game_state.get("is_battle"):
            strategy = individual["battle_strategy"]
            if strategy["aggression"] > 0.7:
                return "attack"
            elif strategy["caution"] > 0.6:
                return "defend"
        else:
            strategy = individual["exploration_strategy"]
            if strategy["exploration"] > 0.5:
                return random.choice(["move_left", "move_right"])
        
        return "wait"
    
    def evaluate_fitness(self, individual, performance_data):
        """개체의 적응도 평가"""
        fitness = 0
        
        # 생존 시간
        fitness += performance_data.get("survival_time", 0) * 10
        
        # 전투 승률
        win_rate = performance_data.get("battles_won", 0) / max(1, performance_data.get("total_battles", 1))
        fitness += win_rate * 100
        
        # 탐험 점수
        fitness += performance_data.get("areas_explored", 0) * 5
        
        return fitness
    
    def evolve_population(self):
        """개체군 진화"""
        # 엘리트 선택 (상위 20%)
        elite_count = self.population_size // 5
        elite_indices = np.argsort(self.fitness_scores)[-elite_count:]
        elite_population = [self.population[i] for i in elite_indices]
        
        # 새로운 개체군 생성
        new_population = elite_population.copy()
        
        # 교배와 돌연변이
        while len(new_population) < self.population_size:
            parent1 = random.choice(elite_population)
            parent2 = random.choice(elite_population)
            
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
    def crossover(self, parent1, parent2):
        """교배 연산"""
        child = {}
        for strategy_type in parent1.keys():
            child[strategy_type] = {}
            for param in parent1[strategy_type].keys():
                # 부모의 유전자를 랜덤하게 선택
                if random.random() < 0.5:
                    child[strategy_type][param] = parent1[strategy_type][param]
                else:
                    child[strategy_type][param] = parent2[strategy_type][param]
        return child
    
    def mutate(self, individual, mutation_rate=0.1):
        """돌연변이 연산"""
        for strategy_type in individual.keys():
            for param in individual[strategy_type].keys():
                if random.random() < mutation_rate:
                    # 10% 확률로 돌연변이 발생
                    individual[strategy_type][param] = random.uniform(0, 1)
        return individual


# 통합 사용 예시
def create_adaptive_ai():
    """적응형 AI 생성"""
    
    print("🧠 적응형 게임 AI 초기화...")
    
    # 강화학습 AI
    rl_ai = ReinforcementLearningAI()
    print("✓ 강화학습 AI 준비 완료")
    
    # 진화 AI
    evo_ai = EvolutionaryGameAI()
    print("✓ 진화형 AI 준비 완료")
    
    # LLM AI (선택적)
    llm_available = False
    try:
        llm_ai = LLMGameAI()
        print("✓ LLM AI 준비 완료")
        llm_available = True
    except:
        print("⚠ LLM AI 비활성화 (로컬 서버 필요)")
    
    return {
        "reinforcement": rl_ai,
        "evolutionary": evo_ai,
        "llm": llm_ai if llm_available else None
    }

if __name__ == "__main__":
    print("🚀 자율 학습 게임 AI 시스템")
    print("\n구현된 방법들:")
    print("1. 강화학습 (DQN) - 경험을 통한 학습")
    print("2. 진화 알고리즘 - 세대별 전략 진화")  
    print("3. LLM 연동 - 상황 판단 및 추론")
    
    adaptive_ai = create_adaptive_ai()
    print(f"\n사용 가능한 AI: {list(adaptive_ai.keys())}")