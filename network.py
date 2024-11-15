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
        # return action.item(), dist.log_prob(action), value
        return action.item(), action, dist.log_prob(action).unsqueeze(0), value
    

    def update(self, states, actions, returns, advantages, log_probs, old_log_probs, old_values, optimizer, batch_size, clip_epsilon):
        # returns, advantages, log_probs, old_log_probs, old_values는 모두 텐서의 리스트 또는 텐서입니다.
        # Tensor로 변환하는 과정
        # print("States Tensor Shape:", len(states))
        states = torch.stack(states, dim=0).detach()
        # print("States Tensor Shape:", states.shape)
        actions = torch.cat(actions).detach()  # 행동 인덱스 텐서
        returns = torch.cat(returns).detach()
        advantages = torch.cat(advantages).detach()
        log_probs = torch.cat(log_probs)
        old_log_probs = torch.cat(old_log_probs).detach()
        old_values = torch.cat(old_values).detach()

        # 정책 및 가치 네트워크의 손실을 계산
        for _ in range(10):  # PPO는 일반적으로 여러 에폭동안 같은 샘플로 업데이트를 수행
            indices = torch.randperm(len(states)).to(states.device)  # 데이터를 셔플
            for i in range(0, len(states), batch_size):  # 미니 배치 학습
                sampled_indices = indices[i:i + batch_size]
                sampled_states = states[sampled_indices]
                sampled_actions = actions[sampled_indices]
                sampled_returns = returns[sampled_indices]
                sampled_advantages = advantages[sampled_indices]
                sampled_log_probs = log_probs[sampled_indices]
                sampled_old_log_probs = old_log_probs[sampled_indices]
                sampled_old_values = old_values[sampled_indices]

                # 새로운 log_prob, values 계산
                action_logits, values = self.forward(sampled_states)
                new_log_probs = torch.log(torch.softmax(action_logits, dim=-1) + 1e-10)  # log_softmax와 동일


                # 샘플링된 행동의 로그 확률 추출
                # sampled_actions = sampled_actions.unsqueeze(1)  # 인덱스를 위해 차원 추가
                sampled_actions = sampled_actions.to(torch.int64).unsqueeze(1)
                sampled_new_log_probs = torch.gather(new_log_probs, 1, sampled_actions).squeeze(1)  # 적절한 로그 확률 추출
                
                
                # 확률비 계산
                ratios = torch.exp(sampled_new_log_probs - sampled_old_log_probs)

                # PPO 클리핑된 목적 함수
                surr1 = ratios * sampled_advantages
                surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * sampled_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 가치 손실
                # value_loss = F.mse_loss(values, sampled_returns)
                value_loss = F.mse_loss(values.squeeze(1), sampled_returns)


                # 전체 손실
                loss = policy_loss + 0.5 * value_loss

                # 업데이트
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



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
        action, action_tensor, log_prob, value = agent.select_action(dummy_state)
        print("Selected Action:", action)
        # print("Log Probability of Selected Action:", log_prob)
        # print("Value of the current state:", value)

if __name__ == "__main__":
    # GPU 장치 설정 (GPU가 있으면 GPU 사용, 없으면 CPU 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_agent()

