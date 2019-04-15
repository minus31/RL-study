#### 

#GAIL

금주에 다루는 내용은 보상함수를 현실적으로 정의하기 힘들거나, 보상이 충분한 정보를 주지 못하는 경우에 사용할 수 있는 방법론이다. (바둑, 주차 등의 예)

#### BC, IRL and IL

보상함수를 정의하기 힘든 경우에 보상함수 대신에 전문가의 데이터를 배우는 방향으로 학습을 할 수 있다. 전문가의 데이터($\tau_E $), 보상은 없고 $\tau_E = (s_0, a_0, s_1, a_1, \cdots)$

**Behavior Cloning**

전문가의 정책을 교사학습 

$\large L(\theta) = \mathbb{E}_{s \sim d^{\pi_E}}[D_{KL}[\pi_E(\cdot|s) || \pi_\theta(\cdot | s)]]$

한계점

- 경험하지 않은 상태에 대해 취약

  covariate shift : 모델이 경험한 것과 다른 데이터에는 제대로된 에측할 수 없는 것을 Covariant shift라고 표현

- 데이터가 굉장히 많이 필요

**Inverse Reinforcement Learning**

BC의 한계를 전문가 데이터가 없는 상태에 대해서도 학습함으로써 극복하려함. 그래도 목적은 전문가의 정책과 유사한 정책 찾기 

1. 데이터가 있는 상태들을 통해 보상함수를 학습 
2. 데이터가 없는 상태에 대해서도 학습된 보상 함수를 기반으로 학습 
3. 정책으로 보고 보상을 찾는 역문제(기존의 보상을 기준으로 최적정책을 찾는 것과 역순)

두 정책이 비슷한 정도를 보상함수로 함. 두 정책이 만드는 데이터가 비슷하면 얻게되는 수확도 비슷할 것이다.

IRL의 가정 - **보상 = Feature들의 선형결합** 

$\large R(s_t, a_t) = w_1, \phi_1 (s_t, a_t) + w_2 \phi_2 (s_t, a_t) + w_3 \phi_3 (s_t, a_t) + \cdots \\ \large \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;= w^\top \phi(s_t, a_t)  \\ \large \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;||w||  \leq 1$

:: $w_i​$는 $i​$ 번째 피쳐의 중요도

한 정책이 에피소드 동안 평균적으로 겪게되는 보상은

$\large J(\pi) = \mathbb{E}_{d^\pi}[w^\top \phi(s_t, a_t)] \\ \large \,\;\;\;\;\;\;\;\ = w^\top \mathbb{E}_{d^\pi} [\phi(s_t, a_t)] \\ \large \,\;\;\;\;\;\;\;\ = w^\top \mu(\pi)$

$\large \mu(\pi) = ​$ Feature expectation 

전문가의 정책과 학습 중인 정책의 차이는 가능한 보상함수 중 두 정책을 가장 크게 차이내는 보상함수.

$\large |J(\pi_E) - J(\pi)| = | w^\top \mu(\pi_E) - w^\top \mu(\pi)  | \\ \large  \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\,   \leq ||w|| \ || \mu(\pi_E) - \mu(\pi) ||$

사실 보상함수가 중요한게 아니라 두 정책을 같게 하는데 집중하는 것이다. 그 과정에서 보상함수가 존재하는 것이다. 이 때 위 수식 처럼 실제로 두 정책의 차이를 나타내는 어떤 보상함수가 있겠지만 어떤 것인지 찾을 수가 없기 때문에 IRL은 여기까지만 사용되고,

**Imitation Learning** 

 Imitation Learning으로 넘어가 못찾겠는 보상함수대신 두 정책의 각 Feature Expectation 차이가 가장 큰 것을 상한으로 두면 어떤 보상함수가 올진 몰라도 어쨋든 그것보다는 차이가 적을 것이기 때문에. 두 정책의 목적함수의 차이를 줄이기 위해 상한을 최소화한다. 정리하면 가장 차이가 적게나는 정책에 대해서 가장 차이가 많이나는 보상함수를 기준으로 정책을 갱신한다. 

$\large |J(\pi_E) - J(\pi)| = | w^\top \mu(\pi_E) - w^\top \mu(\pi)  | \\ \large  \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\,   \leq ||w|| \ || \mu(\pi_E) - \mu(\pi) || \\ \large \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\,   \leq \displaystyle \max_{||w|| \leq 1}||w|| \ || \mu(\pi_E) - \mu(\pi) || $

여기서 Feature들 또한 학습하고, 두 정책 사이의 진짜 거리를 GAN 방식으로 학습한 것이 GAIL 이다. 

먼저 notation 정리 

$\large \pi$가 고정되면 $\large d_\pi(s)$ 가 고정되고 여기에 $\large \pi$ 를 곱하면 Occupancy measure (정책이 만드는 경로의 분포)$\large \rho_\pi (s,a) = d_\pi (s) \pi(a |s)$ 가 정의된다. 기존의 IR은 Feature를 고정해서 $\large \mu(\pi)$를 계산했었다. 

#### GAIL 

GAN과 비교 

GAN - 현재를 기준으로 가장 구분을 잘하는 판별자를 찾는 문제 (Discriminator)

$\large \rightarrow \displaystyle\max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)} [\log D(x)] + \mathbb{E}_{z \sim p_{z} (z)} [\log (1 - D(G(z)))]​$

GAIL - 가장 차이가 많이나는 보상함수를 찾는 문제 

$\large IRL_\phi (\pi_E) = \displaystyle\arg\max_{c \in \mathbb{R}^{S \times A}} - \psi(c) + (\displaystyle\min_{\pi \in \Pi} - H(\pi) + \mathbb{E}_\pi [c(s, a)]  )   - \mathbb{E}_{\pi_E} [c(s,a)]​$

여기서 $\psi(c), H(\pi)$ 는 각각 보상함수와 정책에 대한 정규화 term이다. 보상함수의 정규화 term에 대해서,(GAN에서는) log-sigmoid 함수만을 사용해야한다. 

이 때 최적 정책을 찾지 않고 바로 두 Occupancy Measure를 줄일 수 있으면 되기 때문에 보상함수가 푸는 수식을 다음처럼 쓸 수 있다. 

$\large \psi^*_{GA}*( \rho_\pi - \rho_{\pi_E}) = \displaystyle\max_{D \in (0, 1)^{S\times A}} \mathbb{E}_\pi [\log (D(s,a))] + \mathbb{E}_{\pi_E}[\log(1 - D(s,a))]$

GAN - 현재 판별자를 기준으로 가장 그럴듯한 샘플 생성 

$\large \rightarrow \displaystyle\min_G V(D, G) = \mathbb{E}_{x \sim p_{data}(x)} [\log D(x)] + \mathbb{E}_{z \sim p_{z} (z)} [\log (1 - D(G(z)))]$

GAIL - 현재 보상함수를 기준으로 가장 차이가 적은 정책 

$\large \displaystyle\min_\pi \psi^*_{GA} (\rho_\pi - \rho_{\pi_E}) - \lambda H(\pi) = D_{JS}(\rho_\pi, \rho_{\pi_E}) - \lambda H(\pi)$

$D_{JS}(P || Q) = 0.5 D(P || M) + 0.5 D(Q || M) \ \ where M = 0.5(P+Q)$ 은 Jensen-Shannon Divegence이다. 

#### Advanced GAIL

GAIL은 GAN과 밀접하기 때문에 GAN과 관련된 내용에 대해서 다룬다. 

**Instance Noise**

- Vanishing gradient problem in GAN : 판별자의 loss는 0이 될 수 있다. 따라서 생성된 샘플에 대해서 완전히 분류하기 때문에 기울기가 0 혹은 그 가까이 수렴한다. 그러면  생성자는 학습할 수 가 없어진다. 

Instance Noise는  각 실제와 가짜 샘플에게 노이즈를 추가해서 작위적으로 판별자의 loss를 높여준다. 따라서 기울기도 발생한다. 이 때 어떤 노이즈를 사용하는가는 학습 성과에 큰 영향을 미치는 요인이다. 

- 노이즈 종류

  $\large \epsilon \sim \mathcal{N}(0, \sigma^2I)$, $\large \epsilon \sim \mathcal{N}(0, \Sigma)$

  분산이 클 수록 판별자의 정확도를 낮춘다.

**Zero-centered Gradient Penelty** 

- Regularized Jensen-Shannon GAN 

  $\Omega_{JS}(\mathbb{P}, \mathbb{Q}, \phi) := \mathbb{E}_{\mathbb{p}}[(1-\phi(x))^2 || \nabla \psi (x)||^2 ] + \mathbb{E}_{\mathbb{Q}}[\phi(x)^2|| \nabla \psi(x) ||^2]$

  실제 샘플과 가짜 샘플의 각 기울기 크기에 대해서 Penelty를 준다. 

- 효과

  정규화 term이 없을 때는 판별자는 밀고 생성자는 당기기만 하는데 기울기는 진짜와 가짜 사이에서만 발생하기 때문에 mode collapse 되면서 학습이 끝나지 않는다. 정규화가 존재하면 전체적으로 기울기가 퍼지고 안정화되는 효과 발생한다. 또 이 정규화가 있으면 최적 판별자를 찾지 않고 번갈아가면서 업데이트 해도 괜찮다고 하고, 각 표본에 계산하던 정규화를 한쪽에만 계산해도 괜찮다고 한다. 

**Variational Discriminator Bottleneck**

- VAE & VDB

  VAE 

  - Autoencoding - 복구(reconstruction) 가능한 코드(latent variable distribution)을 만드는 것 

  - KL regularization - Prior distribution과 가깝게 만든다.(정규분포를 가정하므로 코드들이 평균 0, 분산이 1이 되도록 유도한다.)

    $\large \mathcal{L}(\theta, \phi ; x^{(i)}) \simeq \dfrac{1}{2} = \displaystyle\sum_{j = 1}^J (1  + \log( (\sigma_j^i)^2) - (\mu_j^i)^2 - (\sigma_j^i)^2 ) + \dfrac{1}{N} \displaystyle\sum_{l = 1}^L \log p_\theta (x^i | z^{(i,l)}) \\ \large \;\;\; where \ \  z^{(i, l)} = \mu^i + \sigma^i \odot\epsilon^l and  \  \ \epsilon^l \sim \mathcal{N}(0, I)$

  VDB

  - Discriminating - 판별 가능한 코드를 만드는 것 
  - KL regularization - prior와 가까운 코드를 만든다. (이또한 정규분포를 가정하여 평균 0, 분산이 1이 되도록 유도한다. )

  $\large \mathcal{L}(D, E, \beta) = \mathbb{E}_{x \sim p^*(x)}[\mathbb{E}_{z \sim E(z|x)} [ - \log (D(z))] ] + \mathbb{E}_{x\sim G(x)} [ \mathbb{E}_{z \sim E(z|x)} [ -\log(1 -D(z))  ] ] + \beta (\mathbb{E}_{x \sim \hat{p}(x)}[ KL[E(z|x) || r(z)]] - I_c )$

  - 판별자를 학습 시켜도 Instance Noise와 유사한 효과가 발생한다.
