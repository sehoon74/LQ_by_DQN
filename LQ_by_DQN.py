# ==========================
# Motor velocity regulation
# LQ (Riccati) vs DQN (discretized actions)
# ==========================
import math, random
import numpy as np

# ---- Optional: DQN uses PyTorch (comment out if torch is unavailable) ----
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception as e:
    print("PyTorch not available -> will run LQ only.")
    TORCH_OK = False


# ============ 0) Problem Setup ============
# Discrete-time first-order motor velocity model:
#   v_{t+1} = a * v_t + b * u_t
# where v_t is angular velocity error (target is 0), u_t is control input (voltage/torque cmd)
dt = 0.02
tau = 0.15                  # time constant [s] (smaller => faster motor)
a = float(np.exp(-dt/tau))  # discretized first-order factor
b = (1.0 - a) * 1.0         # input gain (scaled to ~1); you can set b=dt/J as well
A = np.array([[a]], dtype=np.float64)
B = np.array([[b]], dtype=np.float64)

# Quadratic cost:  J = sum v_t^2 * q + u_t^2 * r
q = 1.0
r = 0.05
Q = np.array([[q]], dtype=np.float64)
R = np.array([[r]], dtype=np.float64)

gamma = 0.99               # discount (≤1). With stable policy, 1.0도 가능
u_max = 2.0                # actuator limit
process_noise_std = 0.00   # small if you want; here keep 0 for clarity

# Simulation utility
def step_env(v, u):
    """One-step transition and reward."""
    v_next = A @ v + B @ u + np.random.randn(*v.shape) * process_noise_std
    cost = float(v.T @ Q @ v + u.T @ R @ u)
    reward = -cost
    return v_next, reward

# Rollout utility (shared by both controllers)
def simulate(policy_fn, T=200, episode=1, v0_sigma=1.0):
    """Return average episode return and trajectories for the last episode."""
    ep_returns = []
    traj = None
    for ep in range(episode):
        v = np.array([[np.random.randn()*v0_sigma]], dtype=np.float64)
        ret = 0.0
        states, actions = [], []
        for t in range(T):
            u = np.array([[policy_fn(v)]], dtype=np.float64)
            states.append(float(v))
            actions.append(float(u))
            v, r = step_env(v, u)
            ret += (gamma**t) * r
        ep_returns.append(ret)
        traj = (states, actions)
    return float(np.mean(ep_returns)), traj


# ============ 1) LQ Solution via policy iteration (Riccati) ============
# We avoid SciPy; use standard policy iteration:
#   Initialize K0 (stabilizing). Then iterate:
#   P = Q + K^T R K + (A-BK)^T P (A-BK)   [Lyapunov]
#   K = (R + B^T P B)^{-1} B^T P A        [greedy policy improvement]
def lqr_policy_iteration(A, B, Q, R, gamma=1.0, iters=200, K0=None):
    n = A.shape[0]; m = B.shape[1]
    if K0 is None:
        # conservative initial K0 from scalar pole placement (deadbeat-ish)
        K = np.zeros((m, n))
    else:
        K = K0.copy()

    P = Q.copy()
    for _ in range(iters):
        AK = A - B @ K
        # discounted Lyapunov solve by simple fixed-point iteration (small system)
        #   P = Q + K^T R K + gamma * AK^T P AK
        for _ in range(200):
            P_new = Q + K.T @ R @ K + gamma * (AK.T @ P @ AK)
            if np.linalg.norm(P_new - P) < 1e-10:
                break
            P = P_new
        # policy improvement
        G = R + gamma * (B.T @ P @ B)
        F = gamma * (B.T @ P @ A)
        K = np.linalg.solve(G, F)  # K = G^{-1} F
    return K, P

K_lqr, P_lqr = lqr_policy_iteration(A, B, Q, R, gamma=gamma, iters=100)
def lqr_policy(v):
    # u = -K v  (note: above formula produced K for + sign; take negative)
    return float(-(K_lqr @ v)[0,0])

lqr_ret, (lqr_states, lqr_actions) = simulate(lqr_policy, T=300, episode=10)
print(f"[LQ]  K* = {K_lqr.flatten()}   AvgReturn ≈ {lqr_ret:.3f}")


# ============ 2) DQN with discretized actions ============
# Discretize the continuous action u in [-u_max, u_max]
# For 1-D motor, small grid works well.
n_actions = 11
U_grid = np.linspace(-u_max, u_max, n_actions).astype(np.float64)

if TORCH_OK:
    torch.manual_seed(0)
    device = "cpu"

    class QNet(nn.Module):
        def __init__(self, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, n_actions))
        def forward(self, x):  # x: (B,1)
            return self.net(x)

    qnet = QNet().to(device)
    target = QNet().to(device)
    target.load_state_dict(qnet.state_dict())
    opt = optim.Adam(qnet.parameters(), lr=1e-3)
    tau = 0.005  # soft target update
    epsilon_start, epsilon_end, epsilon_decay = 0.5, 0.02, 5000

    # Replay buffer
    class Replay:
        def __init__(self, N=50000):
            self.N=N; self.ptr=0; self.full=False
            self.s = np.zeros((N,1), dtype=np.float32)
            self.a = np.zeros((N,1), dtype=np.int64)
            self.r = np.zeros((N,1), dtype=np.float32)
            self.sp= np.zeros((N,1), dtype=np.float32)
        def push(self, v, a, r, v_next):
            i=self.ptr; self.s[i]=v; self.a[i]=a; self.r[i]=r; self.sp[i]=v_next
            self.ptr=(self.ptr+1)%self.N; self.full = self.full or self.ptr==0
        def sample(self, B):
            M = self.N if self.full else self.ptr
            idx = np.random.randint(0, M, size=B)
            return (torch.tensor(self.s[idx]), torch.tensor(self.a[idx]),
                    torch.tensor(self.r[idx]), torch.tensor(self.sp[idx]))

    replay = Replay()

    # DQN training loop
    def dqn_train(episodes=200, T=200, batch=256, warmup=1000):
        global qnet, target
        step_count = 0
        ep_returns = []
        for ep in range(episodes):
            v = np.array([[np.random.randn()*1.0]], dtype=np.float64)
            ret = 0.0
            for t in range(T):
                # epsilon-greedy
                eps = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-step_count/epsilon_decay)
                if random.random() < eps:
                    a = random.randrange(n_actions)
                else:
                    with torch.no_grad():
                        qvals = qnet(torch.tensor([[float(v)]], dtype=torch.float32)).cpu().numpy()[0]
                        a = int(np.argmax(qvals))
                u = U_grid[a]
                v_next, r = step_env(v, np.array([[u]], dtype=np.float64))
                replay.push(float(v), a, float(r), float(v_next))
                v = v_next
                ret += (gamma**t) * r
                step_count += 1

                # update
                if (replay.ptr > warmup or replay.full) and TORCH_OK:
                    s,a_t,rw,sp = replay.sample(batch)
                    s  = s.to(device); a_t = a_t.to(device); rw = rw.to(device); sp = sp.to(device)
                    with torch.no_grad():
                        q_next = target(sp)
                        a_max  = q_next.argmax(dim=1, keepdim=True)
                        y = rw + gamma * q_next.gather(1, a_max)
                    q = qnet(s).gather(1, a_t)
                    loss = ((q - y)**2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    # soft target update
                    with torch.no_grad():
                        for tp, p in zip(target.parameters(), qnet.parameters()):
                            tp.data.mul_(1-tau).add_(tau*p.data)
            ep_returns.append(ret)
            if (ep+1) % 20 == 0:
                print(f"[DQN] ep {ep+1:4d}  avgRet(last20)={np.mean(ep_returns[-20:]): .3f}")
        return ep_returns

    dqn_returns = dqn_train(episodes=300, T=200)

    # Extract greedy policy from Q-network
    def dqn_policy(v):
        with torch.no_grad():
            qvals = qnet(torch.tensor([[float(v)]], dtype=torch.float32)).cpu().numpy()[0]
        return float(U_grid[int(np.argmax(qvals))])

    dqn_ret, (dqn_states, dqn_actions) = simulate(dqn_policy, T=300, episode=10)
    print(f"[DQN] AvgReturn ≈ {dqn_ret:.3f}")

    # Compare policies on a few initial conditions
    test_inits = [2.0, -2.0, 1.0, -1.0, 0.5]
    print("\nCompare first control action u(v0) for several v0:")
    for v0 in test_inits:
        u_lqr = lqr_policy(np.array([[v0]]))
        u_dqn = dqn_policy(np.array([[v0]]))
        print(f"v0={v0:+.2f}  LQ u={u_lqr:+.3f}   DQN u={u_dqn:+.3f}")

else:
    # Torch unavailable -> still show LQ result and a discrete greedy baseline
    print("\n[Info] Torch not available. Showing LQ only.")
    # Discrete greedy using the same grid to approximate LQ: choose u minimizing Q-stage cost + next V
    U_grid = np.linspace(-u_max, u_max, 11)
    def discrete_greedy_policy(v):
        # one-step lookahead using LQ value P_lqr
        best_u, best_val = None, 1e9
        for u in U_grid:
            v1 = A @ v + B @ np.array([[u]])
            val = float(v.T@Q@v + [[u]]@R@[[u]] + gamma*(v1.T@P_lqr@v1))
            if val < best_val:
                best_val, best_u = val, u
        return best_u
    d_ret, _ = simulate(discrete_greedy_policy, T=300, episode=10)
    print(f"[Discrete greedy w.r.t. LQ-V] AvgReturn ≈ {d_ret:.3f}")

