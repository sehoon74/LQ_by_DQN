### LQ를 DQN으로 풀어서 DQN의 기본을 학습하기
# DQN 프로세스와 LQ 매핑


익숙한 Linear Quadratic problem을 DQN의 frame으로 플어보기.

DQN은 원래 이산 행동 공간(discrete action space)을 대상으로 하지만, LQ는 
**연속 행동 공간(linear–quadratic)** 
문제라 직접적으로 DQN을 쓰기는 맞지 않음. 
다만 
**DQN의 각 step**
을 
**LQ 문제에 비유**
하면 DQN의 구조를 이해하는 데 도움



## 1. 기본 개념

* **DQN:**

  * Q-function $Q(x,a)$를 NN으로 근사.
  * Bellman 잔차를 줄이면서 학습.
  * 행동은 $\epsilon$-greedy로 선택.
  * Replay buffer + target network 사용.

* **LQ 대응:**

  * $Q(x,u)$는 사실 정확히 **이차형(quadratic)**.
  * 행동 $u$는 연속이지만, 만약 **유한한 candidate gain $K$** 집합을 두면 이산 문제로 볼 수 있음.
  * 예: “control gain 후보 $\{K_1,K_2,\dots,K_M\}$ 중 선택” → DQN이 평가.



## 2. Step ① 경험 수집 (Exploration)

* **DQN:** 상태 $x_t$에서 행동 $a_t$를 선택 ($\epsilon$-greedy).
* **LQ:** 상태 $x_t$에서 여러 gain 후보 중 하나를 선택해 $u=-K_i x_t$. (이산화 버전)
* 또는 단순화: “왼쪽/오른쪽 토크”처럼 **이산 입력**으로 LQ의 continuous action을 coarse하게 근사.



## 3. Step ② Q-function 업데이트

* **DQN target:**

  $y_t = r_t + \gamma \max_{a'} Q_{\theta^-}(x_{t+1},a'). $
* **LQ 대응:**

  * reward: $r_t = -(x_t^\top Q x_t + u_t^\top R u_t)$.
  * next-state: $x_{t+1}=Ax_t+Bu_t$.
  * target은 결국 Riccati Bellman 식과 동일:

    $Q(x,u) = r(x,u) + \gamma \min_{u'} Q(x',u').    $
  * 여기서 $\min_{u'}$가 $\max_a$에 해당.
  * 즉 **DQN 업데이트 = Riccati 반복과 구조적으로 동일**.



## 4. Step ③ Policy Extraction

* **DQN:** 정책은 greedy:

  $\pi(x)=\arg\max_a Q(x,a).  $
* **LQ:** 정책은 최적 feedback gain:

  $u^\star(x) = \arg\min_u Q(x,u).  $
* 실제로 Q-function이 quadratic이므로 최소값은 **닫힌 형태**로 나오고, 이게 바로 Riccati 최적 제어 $u=-K^\star x$.



## 5. Step ④ Target Network

* **DQN:** $Q_{\theta^-}$를 천천히 업데이트해 안정성 확보.
* **LQ:** 이론적으로 필요 없지만, 학습적 관점에서는 “수렴을 완화시키는 보조장치”로 설명 가능.



# 📌 요약

* DQN = “이산 행동 공간에서 Bellman 반복으로 Q-function 학습”
* LQ = “연속 행동 공간에서 Quadratic Q-function을 학습 → Riccati 방정식”
* 차이는 action space 뿐이고, \*\*핵심 아이디어(다음 상태의 Q를 bootstrap으로 이용해 현재 Q를 업데이트)\*\*는 동일.
* 즉, LQ는 DQN의 구조를 **수학적으로 완전히 풀 수 있는 toy model**로 볼 수 있습니다.

---


# 1) DQN의 Q 업데이트: 어떤 네트워크들이 있으며, 누가 무엇을 하나?

DQN(기본형)은 **두 개의 신경망**만 씁니다.

1. **온라인 Q-네트워크** $Q_\theta(x,a)$

   * 현재 학습 중인 본체.
   * **행동 선택**(ε-greedy)과 **손실 계산의 예측값**에 사용.

2. **타깃 Q-네트워크** $Q_{\bar\theta}(x,a)$

   * 온라인 네트워크의 **느린 사본**(moving target 완화).
   * **TD 타깃값** 계산에만 사용.
   * 주기적 하드카피(예: 매 C 스텝) 또는 소프트 업데이트

     $\bar\theta \leftarrow \tau\,\theta + (1-\tau)\,\bar\theta$

     로 천천히 따라옴.

> DQN에는 **정책(Actor) 네트워크가 따로 없습니다.** 정책은 “온라인 Q의 greedy”로 암묵적으로 정의됩니다:
> $\pi(x)=\arg\max_a Q_\theta(x,a)$.


# 2) 한 미니배치에서의 Q 업데이트 수식(기본 DQN)

리플레이 버퍼에서 $(x_i,a_i,r_i,x'_i)$를 샘플하면:

* **TD 타깃(bootstrap):**

  $y_i = r_i + \gamma \max_{a'} Q_{\bar\theta}\big(x'_i, a'\big)$

* **예측값:**

  $\hat q_i = Q_{\theta}(x_i, a_i)$

* **손실(평균제곱오차):**

  $\mathcal{L}(\theta)=\frac{1}{N}\sum_i \big(\hat q_i - y_i\big)^2 $

* **경사하강:** $\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$

* **타깃 네트워크 갱신:** 하드카피 또는 소프트 업데이트로 $\bar\theta$ 이동.

> (선택) **Double DQN**이면 타깃만 바뀝니다(과대추정 완화):
>
> $$y_i = r_i + \gamma\, Q_{\bar\theta}\!\Big(x'_i,\ \underbrace{\arg\max_{a'} Q_\theta(x'_i,a')}_{\text{온라인으로 선택}}\Big)$$
>
> 즉, **행동 선택은 온라인**, **평가는 타깃**이 합니다.


# 3) LQ 문제에 **그대로** 적용하면 어떤 값들이 들어가나?

LQ(연속 입력)를 DQN으로 풀려면 **입력을 이산화**해서 $\mathcal{U}=\{u^{(1)},\dots,u^{(M)}\}$로 둡니다.
우리 모터 예제에서 M=9(또는 11) 같은 **균일 그리드**를 썼죠.

* **상태:** $x\equiv v$ (속도 오차)
* **전이:** $x' = A x + B u$
* **보상:** $r(x,u)= -\big(x^\top Q x + u^\top R u\big)$  ← LQ 비용의 음수
* **Q-네트 출력:** 길이 $M$ 벡터 $[Q(x,u^{(1)}),\dots,Q(x,u^{(M)})]$

이제 DQN 공식을 LQ에 대입하면:

* **TD 타깃:**

  $y = -\big(x^\top Q x + u^\top R u\big) + \gamma \max_{u'\in\mathcal{U}} Q_{\bar\theta}(x',u'),\quad x' = A x + B u.$
* **예측:** $\hat q = Q_\theta(x,u)$
* **손실:** $(\hat q - y)^2$
* **업데이트:** $\theta$를 경사하강, $\bar\theta$는 느리게 추종.

즉, **LQ의 리카티 구조**(연속최적해 $u=-Kx$)를 **격자 위의 greedy 선택**으로 근사합니다. 그리드가 촘촘할수록 $\arg\max$가 연속 $\arg\min_u$에 가까워져 $u \approx -K^\star x$를 점점 잘 따릅니다.


# 4) “네트워크는 몇 개인가?”를 코드 흐름으로 한눈에

* **온라인 Q**: `qnet(s)` → 행동 선택(ε-greedy), 예측값 $\hat q$
* **타깃 Q**: `target(sp)` → 타깃값 $y$의 **부트스트랩 항** 계산
* **하나의 옵티마이저**로 `qnet`만 업데이트
* `target`은 주기적/소프트로 천천히 `qnet`을 따라감
* (Double DQN이면) `argmax`는 `qnet`, 값 평가는 `target`

> **중요:** DQN은 **정책 네트워크가 따로 없다**는 점이 DDPG/SAC와 다릅니다.
> 정책은 항상 $\pi(x)=\arg\max_a Q_\theta(x,a)$로 “Q에서 읽어냅니다”.


# 5) LQ 이론과의 연결(왜 이게 맞는가?)

* LQ의 **진짜** $Q^\star(x,u)$는 연속 $u$에서 **정확히 이차형**:

  $Q^\star(x,u) = x^\top Q x + u^\top R u + \gamma (Ax+Bu)^\top P^\star (Ax+Bu).$
* DQN은 **격자점** $u^{(m)}$에서의 $Q^\star(x,u^{(m)})$ 값을 신경망으로 근사하고,
  $\max_m Q_\theta(x,u^{(m)})$로 greedy 선택.
* 학습이 충분하고 격자가 충분히 촘촘하면,
  $\max_m$는 연속 $\min_u$를 잘 근사 → \*\*리카티 해 $K^\star$\*\*에 수렴.

> 직관: DQN의 업데이트는 **벨만 최적성 방정식**
> $Q= r + \gamma \max Q$
> 을 데이터로 맞추는 과정이고, LQ는 그 방정식의 **닫힌 해**가 있는 특수 케이스(연속 $u$를 이산 근사).


# 6) FAQ (실무 팁)

* **왜 타깃 네트워크가 필요한가?**
  온라인 네트워크로 타깃까지 만들면 타깃이 매 스텝 흔들려 발산/진동 위험 ↑. 타깃 네트워크가 고정점 근처로 천천히 끌어가 줍니다.

* **보상이 ‘비용의 음수’인 이유?**
  DQN은 $\max$를 쓰므로, 최소비용 문제를 **최대보상**으로 바꾸기 위해 $r= -\text{cost}$.

* **Double DQN을 LQ에 쓰면?**
  연속 LQ에서도 **과대추정 편향**을 줄여 수렴을 더 안정화하는 데 도움이 됩니다.

* **격자 해상도는?**
  너무 거칠면 $u=-Kx$의 선형 정책을 잘 못 따라가고, 너무 촘촘하면 학습 난이도/시간이 증가합니다(차원↑일수록 DDPG/SAC 권장).


# 7) (선택) 수식 한 줄 요약 — 한 샘플의 그래디언트

$\nabla_\theta \mathcal{L}
= \big(Q_\theta(x,a) - [r + \gamma \max_{a'} Q_{\bar\theta}(x',a')]\big)\, \nabla_\theta Q_\theta(x,a).
$

즉, **오차(δ) × 기울기(∇Q)** 형태의 반향학습이며,
LQ에서는 $r = -(x^\top Q x + u^\top R u)$, $x'=Ax+Bu$가 대입됩니다.


