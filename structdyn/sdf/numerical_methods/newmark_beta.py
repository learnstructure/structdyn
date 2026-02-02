import numpy as np


def get_beta_parameters(acc_type="average"):
    """Get Newmark-Beta method parameters."""
    if acc_type == "average":
        beta, gamma = 1 / 4, 1 / 2
    elif acc_type == "linear":
        beta, gamma = 1 / 6, 1 / 2
    else:
        raise ValueError("Invalid acceleration_type. Choose 'average' or 'linear'.")
    return beta, gamma


class NewmarkBeta:
    """NewmarkBeta Scheme for Solving Structural Dynamics Numerically"""

    def __init__(
        self,
        sdf,
        dt,
        u_init=0,
        u_dot_init=0,
        p_init=0,
        non_linear=False,
        acc_type="average",
    ):
        self.dt = dt
        self.non_linear = non_linear
        self.beta, self.gamma = get_beta_parameters(acc_type)

        self.k, m, ji, w_n = sdf.k, sdf.m, sdf.ji, sdf.w_n
        c = 2 * m * w_n * ji  # Damping constant

        self.u_init, self.u_dot_init = u_init, u_dot_init
        self.u_dot_init = u_dot_init if u_dot_init is not None else 0

        u_dot2_init = (p_init - c * self.u_dot_init - self.k * self.u_init) / m

        self.u_minus1 = self.u_init - dt * self.u_dot_init + (u_dot2_init * dt**2) / 2

        self.a1 = m / (m * dt**2) + c / (2 * dt)

        self.k_bar = m / dt**2 + c / (2 * dt)

        self.b = self.k - 2 * m / dt**2
        self.b_bar = 2 * m / dt**2

    def compute_solution(self, time_steps, load_values, fs=None):
        """Computes the displacement, velocity and resisting force of the SDF system."""
        n_steps = len(time_steps)
        u, u_dot, p_bar, fs_val = (
            np.zeros(n_steps),
            np.zeros(n_steps),
            np.zeros(n_steps),
            np.zeros(n_steps),
        )

        u[0], u_dot[0], fs_last = self.u_init, self.u_dot_init, 0

        for i in range(n_steps - 1):
            u_prev = self.u_minus1 if i == 0 else u[i - 1]

            if self.non_linear:
                fs_val[i] = fs(u[i], u_prev, fs_last)
                p_bar[i] = (
                    load_values[i] - self.a * u_prev + self.b_bar * u[i] - fs_val[i]
                )
                fs_last = fs_val[i]
            else:
                p_bar[i] = load_values[i] - self.a * u_prev - self.b * u[i]
                fs_val[i] = self.k * u[i]

            u[i + 1] = p_bar[i] / self.k_bar
            u_dot[i] = (u[i + 1] - u_prev) / (2 * self.dt)

        fs_val[-1] = fs(u[-1], u[-2], fs_last) if self.non_linear else self.k * u[-1]
        return u, u_dot, fs_val
