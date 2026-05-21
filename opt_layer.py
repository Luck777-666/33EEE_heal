import numpy as np
import cvxpy as cp

class OptLayer:
    """
    可微物理投影层 —— IEEE33 节点配电网“物理天条”引擎
    功能：
      - 将任意 AI 动作 a_hat 投影到基尔霍夫等式、热稳定/辐射状不等式定义的可行域
      - 同时输出解析雅可比矩阵 J_KKT = ∂a*/∂â (基于活跃约束的隐函数定理)
    接口兼容：
      - project(a_hat) -> a_star
      - compute_jacobian(a_hat) -> J (n x n)
      - forward(a_hat) -> (a_star, J)
    """
    def __init__(self, W=None, C=None, b=None, d=None, n=37):
        self.n = n

        # ---------- 等式约束：W @ a == b （基尔霍夫功率平衡）----------
        if W is None:
            # 示例矩阵：取前5个等式约束（答辩时可替换为真实关联矩阵）
            self.W = np.eye(self.n)[:5]          # shape (5, 37)
            self.b = np.zeros(5)
        else:
            self.W = W
            self.b = b if b is not None else np.zeros(W.shape[0])

        # ---------- 不等式约束：C @ a <= d （热稳定、无环路、开关限值）----------
        if C is None:
            # 基本约束：所有开关状态在 [0,1] 内
            C_bound = np.vstack([np.eye(self.n), -np.eye(self.n)])   # (2*n, n)
            d_bound = np.ones(2 * self.n) * 1.0

            # 可追加辐射状约束，例如：总闭合开关数不超过 4 个（防止环路）
            C_radial = np.ones((1, self.n))      # (1, n)
            d_radial = np.array([4.0])

            self.C = np.vstack([C_bound, C_radial])   # (2*n+1, n)
            self.d = np.hstack([d_bound, d_radial])
        else:
            self.C = C
            self.d = d

        # 求解器选项：使用高精度 OSQP（cvxpy 底层调用）
        self.solver_opts = {'solver': cp.OSQP, 'eps_abs': 1e-10, 'eps_rel': 1e-10}

    def project(self, a_hat):
        """
        QP投影：min 0.5 ||a - a_hat||²
                s.t. W a == b,  C a <= d
        返回: a_star (n,)
        """
        a = cp.Variable(self.n)
        objective = cp.Minimize(cp.sum_squares(a - a_hat))
        constraints = [self.W @ a == self.b, self.C @ a <= self.d]
        prob = cp.Problem(objective, constraints)
        prob.solve(**self.solver_opts)
        if a.value is None:
            # 若求解失败，返回原始动作（降级处理）
            return np.array(a_hat)
        return a.value

    def get_active_set(self, a_star, tol=1e-8):
        """
        提取在 a_star 处起作用的不等式约束行索引
        返回: A_active (m_active, n) 包含所有等式约束 + 活跃不等式约束
        """
        A_eq = self.W.copy()
        # 检查每个不等式约束是否活跃
        active_rows = []
        for i in range(self.C.shape[0]):
            if abs(self.C[i] @ a_star - self.d[i]) <= tol:
                active_rows.append(self.C[i])
        if active_rows:
            A_ineq = np.array(active_rows)
            A_active = np.vstack([A_eq, A_ineq]) if A_eq.shape[0] > 0 else A_ineq
        else:
            A_active = A_eq
        return A_active

    def compute_jacobian(self, a_hat):
        """
        计算完整雅可比矩阵 J_KKT = ∂a*/∂â
        基于活跃约束的零空间投影：J = I - Q Q^T，其中 Q 为 A_active^T 的正交基
        """
        a_star = self.project(a_hat)
        A_active = self.get_active_set(a_star)
        if A_active.shape[0] == 0:
            return np.eye(self.n)
        Q, R = np.linalg.qr(A_active.T)
        rank = np.linalg.matrix_rank(R)
        Q = Q[:, :rank]
        J = np.eye(self.n) - Q @ Q.T
        return J

    def forward(self, a_hat):
        """
        前向传播接口（给 SAC 网络调用）：
        输入: a_hat (n,) 原始动作
        输出: a_star (n,) 安全动作, J (n, n) 雅可比矩阵
        """
        a_star = self.project(a_hat)
        J = self.compute_jacobian(a_hat)   # 为清晰性重复求解，可优化
        return a_star, J