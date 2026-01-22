import cvxpy as cp
import torch
import numpy as np


class NashMSFL():
    def __init__(
        self,
        n_tasks: int,
        max_norm: float = 1.0,
        update_weights_every: int = 1,
        optim_niter=20,
    ):
        super(NashMSFL, self).__init__(
            # n_tasks=n_tasks,
        )

        self.optim_niter = optim_niter
        self.update_weights_every = update_weights_every
        self.max_norm = max_norm
        self.n_tasks = n_tasks
        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = self.init_gtg = np.eye(self.n_tasks)
        self.step = 0.0
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)

    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (
                np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                < 1e-6
            )
        )

    def solve_optimization(self, gtg: np.array):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.n_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(
            shape=(self.n_tasks,), value=self.prvs_alpha
        )
        self.G_param = cp.Parameter(
            shape=(self.n_tasks, self.n_tasks), value=self.init_gtg
        )
        self.normalization_factor_param = cp.Parameter(
            shape=(1,), value=np.array([1.0])
        )

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.n_tasks):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )
        self.prob = cp.Problem(obj, constraint)

    def get_weighted_loss(self,losses,dummy_data,):
        """
        """

        extra_outputs = dict()
        if self.step == 0:
            self._init_optim_problem()

        if (self.step % self.update_weights_every) == 0:
            self.step += 1

            grads = {}
            for i, loss in enumerate(losses):
                g = list(
                    torch.autograd.grad(
                        loss,
                        dummy_data,
                        retain_graph=True,
                    )
                )
                grad = torch.cat([torch.flatten(grad) for grad in g])
                grads[i] = grad

            G = torch.stack(tuple(v for v in grads.values()))
            GTG = torch.mm(G, G.t())

            self.normalization_factor = (
                torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            )
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())
            alpha = torch.from_numpy(alpha)

        else:
            self.step += 1
            alpha = self.prvs_alpha

        weighted_loss = sum([losses[i] * alpha[i] for i in range(len(alpha))])
        extra_outputs["weights"] = alpha
        return weighted_loss, extra_outputs, alpha


def _stop_criteria(alpha_param, gtg, prvs_alpha_param, alpha_t):
    return (
            (alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (
                    np.linalg.norm(alpha_param.value - prvs_alpha_param.value)
                    < 1e-6
            )
    )

def solve_optimization(gtg: np.array):
    n_tasks = 2
    init_gtg = np.eye(n_tasks)
    G_param = cp.Parameter(shape=(n_tasks, n_tasks), value=init_gtg)
    normalization_factor_param = cp.Parameter(shape=(1,), value=np.array([1.0]))
    normalization_factor = np.ones((1,))
    prvs_alpha = np.ones(n_tasks, dtype=np.float32)
    alpha_param = cp.Variable(shape=(n_tasks,), nonneg=True)
    prvs_alpha_param = prvs_alpha_param = cp.Parameter(shape=(n_tasks,), value=prvs_alpha)

    G_param.value = gtg
    normalization_factor_param.value = normalization_factor
    G_alpha = G_param @ alpha_param

    def _calc_phi_alpha_linearization(G_param, prvs_alpha_param, alpha_param):
        G_prvs_alpha = G_param @ prvs_alpha_param
        prvs_phi_tag = 1 / prvs_alpha_param + (1 / G_prvs_alpha) @ G_param
        phi_alpha = prvs_phi_tag @ (alpha_param - prvs_alpha_param)
        return phi_alpha

    phi_alpha = _calc_phi_alpha_linearization(G_param, prvs_alpha_param, alpha_param)
    obj = cp.Minimize(cp.sum(G_alpha) + phi_alpha / normalization_factor_param)
    constraint = []
    for i in range(n_tasks):
        constraint.append(
            -cp.log(alpha_param[i] * normalization_factor_param)
            - cp.log(G_alpha[i])
            <= 0
        )
    prob = cp.Problem(obj, constraint)
    alpha_t = prvs_alpha
    for _ in range(20):
        alpha_param.value = alpha_t
        prvs_alpha_param.value = alpha_t

        try:
            prob.solve(solver=cp.ECOS_BB, warm_start=True, max_iters=100)
        except:
            alpha_param.value = prvs_alpha_param.value

        if _stop_criteria(alpha_param, gtg, prvs_alpha_param, alpha_t):
            break

        alpha_t = alpha_param.value

    if alpha_t is not None:
        prvs_alpha = alpha_t

    return prvs_alpha

def get_weighted_loss(losses,dummy_data):
    """
    """

    grads = {}
    for i, loss in enumerate(losses):
        g = list(
            torch.autograd.grad(
                loss,
                dummy_data,
                retain_graph=True,
            )
        )
        grad = torch.cat([torch.flatten(grad) for grad in g])
        grads[i] = grad

    G = torch.stack(tuple(v for v in grads.values()))
    GTG = torch.mm(G, G.t())
    normalization_factor = (torch.norm(GTG).detach().cpu().numpy().reshape((1,)))
    GTG = GTG / normalization_factor.item()
    alpha = solve_optimization(GTG.cpu().detach().numpy())
    alpha = torch.from_numpy(alpha)

    return alpha