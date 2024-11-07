import torch
import torch.nn as nn
import torch.nn.functional as F

# PPO 에이전트 네트워크 정의
class PPOAgent(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(PPOAgent, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.action_head = nn.Linear(hidden_dims[1], output_dim)
        self.value_head = nn.Linear(hidden_dims[1], 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        value = self.value_head(x)
        return action_logits, value

    def select_action(self, state):
        # 입력 상태의 크기가 input_dim과 일치하는지 검사
        if state.size(-1) != self.input_dim:
            raise ValueError(f"Expected input dimension is {self.input_dim}, but got {state.size(-1)}")
        
        action_logits, value = self.forward(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

# 네트워크 초기화 함수
def create_agent(input_dim, hidden_dims, output_dim):
    agent = PPOAgent(input_dim, hidden_dims, output_dim).to(device)
    return agent

def test_agent():
    # 에이전트 생성
    input_dim = 24  # 예시로 24차원 상태 벡터를 가정
    hidden_dims = [128, 64]  # 은닉층 차원 설정
    output_dim = 4  # 예를 들어, 4개의 가능한 행동을 가정

    agent = create_agent(input_dim, hidden_dims, output_dim)

    for i in range(50):
        # 더미 입력 데이터 생성 (랜덤)
        dummy_state = torch.rand((1, input_dim)).to(device)

        # 행동 선택 및 출력
        action, log_prob, value = agent.select_action(dummy_state)
        print("Selected Action:", action)
        # print("Log Probability of Selected Action:", log_prob)
        # print("Value of the current state:", value)

if __name__ == "__main__":
    # GPU 장치 설정 (GPU가 있으면 GPU 사용, 없으면 CPU 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_agent()

