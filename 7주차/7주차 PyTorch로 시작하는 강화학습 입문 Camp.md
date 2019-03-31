# 7주차 PyTorch로 시작하는 강화학습 입문 Camp

**출석 : 7명** (조교 2명 제외)

------

#### 커리큘럼

<span style="color:magenta">:: 금 주 진도 </span>

##### **Part 1. Basics**

> > <u>2017년 이전 까지의 연구를 "기초"라 표현</u>

-  강화학습 의 기반이 된 아이디어들 MDP, ML
-  가치 기반 방법론들 (Q-Learning, DQN)
-  정책 기반 방법론들(TRPO, PPO)

###### **Part 2. Advanced (다양한 주제)**

- Off-policy 정책기반 방법론들 (ACER, SAC)
- 탐험 기법들 (엔트로피 정규화, 내적 동기 부여)
- **강화학습과 불확실성 (Distributional DQN)**
  - <span style="color:magenta">Uncertainty Basic </span>
  - <span style="color:magenta">Distributional RL</span>

###### **Part 3. Applications(적용의 확장)**

- 모방학습, 역강화 학습 (GAIL, IRL)
- 다중 에이전트 강화 학습(Regret, MCTS)

#### 피드백 

------

- 강의는 5시 20분에 종료되었습니다. 
- 

#### 강의 내용 

------

보상함수를 현실적으로 정의하기 힘들거나, 보상이 충분한 정보를 주지 못하는 경우가 존재(축구 같은 거 )

**BC**

보상함수를 지우는 것 부터 시작 

대신에 $\tau_E$ (전문가의 트라젝토리를 쓰자 대신 보상은 없다)  ... (s_0, a_0, s_1, a_1)

- 경험하지 않은 상태에 대해 취약(covariate shift)
- 데이터가 많이 필요

**IRL**

 데이터가 있는 상태들을 통해 보상 함수를 학습하고
•데이터가 없는 상태에 대해서도 학습된 보상 함수를 기반으로 학습
•문제(보상 함수)를 보고 답(정책)을 찾는 것이 아니라
•답을 보고 문제를 찾는 역문제(Inverse problem)

두 정책이 비슷한 기준을 보상함수로 잡는다. 



#### Advanced GAIL

::: GAN related paper review 

Vanishing Gradient problem in GANs

학습이 거의 끝나도 discriminator의 로스가 더 내려가는 현상에 주목 

노이즈에 따라 학습이 fluctuating -> Instance noise를 근사해서 



VDB - VAE를 RL에 

(Deep variational information bottleneck)

이렇게 해도 instance Noise와 유사한 효과가 발생한다. 



GAIL loss BC learning>??? 문제 (1:00)

#### 질문

---

보상함수를 정의한다는 것이 정확히 이해가 가지 않는다.. 

- 보상함수를 정의할 때는 행동에 대한 점수..... (6분 30초) 이를 정의하지 않는 것은 전문가의 데이터와의 거리를 줄이는 방향으로 학습할 것 이다. 

IRL에서 IL 넘어가는 이유는 ?

- 31:30, 33:00

GAIL - off policy가 안되는 이유 ? 

- 1:21:00

