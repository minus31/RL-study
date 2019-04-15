## For multi agent 

**Markov Game/Stochastic Game (MG)**

- 상태집합 ($S$)

- 에이전트 집합($N$) - 에이전트들의 집합

- 에이전트 별 행동집합($A^i$) - 각 에이전트들이 고를 수 있는 모든 행동 집합

- 행동 집합($A$) : 모든 에이전트가 고를 수 있는 모든 행동의 조합 

- 전이 확률($T^a_{s, s'}​$) : 상태, 행동이 정해졌을 때 환경이 다음 상태를 고르는 규칙 

- 보상($r_s^a$) 상태와 행동이 정해졌을 때 환경이 각 에이전트에게 주는 효용 

  > 다음 상태의 전이 확률과 각 보상은 모든 에이전트의 행동이 고려되어 정해지는 것(따라서 한 에이전트의 행동이 변하지 않아도 보상은 다를 수 있다.)

- 할인율($\gamma$): 현재 시점과 다음 시점의 보상 비율 

**게임의 종류** 

에이전트가 여러명, 모두의 행동이 모두의 보상합수에 영향을 주는 것 (예로 가위바위보, 바둑, Starcraft )
- 행동을 결정하는 순서에 따라 
  - 동시 게임 : 가위 바위 보 (실제로 행동이 동시가 아니어도 서로의 행동을 아는 순간이 동시이면 된다.)
  - 순차 게임 : 바둑
- 정보의 수준에 따라 
  - 완전정보게임(perfect information) : 바둑 
  - 불완전 정보 게임(imperfect information) : 스타크래프트
- 보상 함수의 성질에 따라 
  - 영합게임(Zero-sum) : 바둑 
  - 비영합 게임(non-Zero-sum) :  죄수의 딜레마

**Nash equilibria**

- 순수 내쉬 균형

   : 각 에이전트들 모두가 바뀌지 않아도 내가 바뀔 동기가 없는 상태

   가정 : 각 에이전트는 행동을 결정론적으로 선택 

  $\large r_i(a) \geq r_i(a_i', a_{-i}), \forall \in N, \forall a_i \in A^i$

  (수식에서 $a_i$ 는 i번째 에이전트이고  $a_{-i}$ 는 i를 제외한 나머지 에이전트를 의미한다.)

- 혼합 내쉬 균형 

   : 각 에이전트들 모두가 바뀌지 않아도 내가 바뀔 동기가 없는 상태

  가정 : 각 에이전트는 행동을 확률적으로 선택 

  $\large \mathbb{E}_{a_i \sim \pi^i, a_{-i} \sim \pi^{-i}}[r_i(a)] \geq \mathbb{E}_{a_i' \sim \pi^{i'}, a_{-i} \sim \pi^{-i}}[r_i(a_i',a_{-i})], \forall i \in N$

  : 순수 내쉬 균형은 있는 경우 없는 경우가 있지만, 혼합내쉬균형은 모두 있다는 것을 내쉬가 증명했다. 

**UCB**

: 불확실한 행동일 수록 좋은 행동일 가능성이 높다고 간주하고 가장 좋을 가능성이 높은 행동을 고르는 알고리즘 (행동가치함수에 불확실성을 추가해서 고려)

$\large Q^\pi(s,a) + \sqrt{\dfrac{2\log(t)}{N_t(a)}}$

동시게임에 대해서는 바로 적용이 가능하다. 이를 순차게임(바둑, 포커)에 대해서 적용하기 위한 것이 UCT

**Upper Confidence Bounds for Trees(UTC)**

주어진 상태에서 한 번의 행동은 UCB 로 결정하고(안해본 행동이 있으면 우선 선택하고 아니면 UCB로 행동을 선택) 그 이후의 행동은 Default policy 로 빠르게 시뮬레이션한다. 그 후에는 가치를 업데이트한다. 

<img src="https://www.dropbox.com/s/vk3tgtzvzkudwqa/Screenshot%202019-04-11%2002.02.40.png?raw=1">

* AlphaGo Fan

  UTC에서 학습하는 것 

  Tree Policy

  $\large Q^\pi(s,a) + c P(s,a) \sqrt{\dfrac{\sum_b N_r(s,b) }{ 1 + N_r(s,a) }}$

  - 상태가치 함수 

  - 불확실성에 대한 인센티브

  Defalut policy 

* AlphaGo Zero 

  AlphaGo Fan과의 차이

  - 강화학습만 사용
  - 네트워크 구조 개선(Residual block, BN)
  - PPO처럼 Policy/Value Net을 하나로 구성 
  - Tree/Default policy를 구분하지 않음 
  - Policy를 RL할 때 UCB의 search probability(시뮬레이션에서 얼마나 좋은 행동이었는가) 를 Value로 두었다.
  - Rollout 단계를 삭제 

**Counterfactual Regret(CFR)**

그냥 regret은 UCB처럼 상태를 고려할 수 없으므로 상태가 주어졌을 때 regret을 고려하기 위해, 각 상태마다 행동에 대한 regret을 정의 (수식을 보면 그냥 Advantage 함수)

$\large Q^t(s,a) - \displaystyle\sum_{a \in A} Q^t(s,a) \pi^t(s,a) = Q^t(s,a) - V^t(s)$

한 에피소드의 (상태 행동) 별 후회

$\large (R_t^i)^{(CF)} (I, a) = \displaystyle\sum_{i=1}^T Q^i_{\sigma_t} (I,a) - V^i_{\sigma_t}(I)​$

한 에피소드의 상태 별 후회

$\large (R^i_T)^{(CF)}(I,a) = \displaystyle\sum_{t=1}^T Q^i_{\sigma_t} (I,a) - V^i_{\sigma_t}(I)$

어떤 행동의 Regret이 크다는 것은 그 행동의 누적 이익이 크다는 것이다. 

**CFR minimization**

1. Regret/CFR이 가장 작은 행동을 많이하도록 정책을 학습 

   (TD learning과 유사하게) 후회함수를 학습

​	$\large \rho_t(s,a) = \rho_{t-1}(s,a) + (Q_t(s,a) - V_t(s))  $

​	(SAC와 유사하게 정책함수를 학습)

​	$\large \pi_{t+1} (s | s) \doteq \dfrac{\exp(\rho_t(s, a))}{\sum_{a \in A} \exp(\rho_t(s,a))}$

​	이를 반복하는 것이  No-regret algorithm 이다. 

2. Entropy regularization등으로 탐험 

**CRF와 다중에이전트 상황의 관련**

CFR minimization을 통해 no-regret을 달성하면 내쉬균형을 찾을 수 있다.(in 2인 제로섬에서)

그 이상의 경우에도 균형을 찾을 가능성이 있다. 

**상관균형 ; Correlated Equilibria** 

혼합내쉬균형 

 : 각 에이전트의 정책이 독립적일 때 정의, 

에이전트들의 행동이 독립적이지 않은 상황에서는 정의되지 않음 (예로 신호등, 사회계약)

상관균형

: 각 에이전트들은 공통된 확률적 현상을 관찰하고 현상의 결과에 따라서 결정론적으로 행동 

​	$\large \mathbb{E}_{s \sim \sigma} [r_i(\delta(s))] \geq \mathbb{E}_{s \sim \sigma}[r_i(\delta_i'(s), \delta_{-i}(s)] , \forall i \in N$

: 모든 혼합 내쉬 균형은 상관균형(부분집합)

**Coarse Correlated Equilibria(CCE)**

상관 균형을 조금 더 일반화, 각 에이전트들이 독립적이지 않은 정책에 대해 바귈 동기가 없는 경우. 

​	$\large \mathbb{E}_{a\sim \pi}[r_i(a)] \geq \mathbb{E}_{a_{-i} \sim \pi^{-i}}[r_i (a'_i, a_{-i})], \forall i \in N$

- 모든 상관균형은 CCE

**No-regret & Equilibria**

각 에이전트가 no-regret 알고리즘으로 반복하면 

- 각 에이전트들은 매 에피소드 때 후회를 계산
- 각 에이전트들은 후회를 기반으로 정책을 수정(CRF minimization)
- 결과적으로 에이전트들의 정책의 평균은 CCE로 수렴하고 
- 만약 각 에이전트들의 정책이 수렴하면 MNE로 수렴한다. 
