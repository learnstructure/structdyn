import matplotlib.pyplot as plt


def fs_elastoplastic(uy=0.02, fy=36000):
    """Get resisting force for given u.
    uy is Yield deformation and fy is yield force"""

    def get_fs_elastoplastic(
        u,
        u_last,
        fs_last=0,
    ):
        fs = fs_last + fy / uy * (u - u_last)
        fs = fs if abs(fs) < fy else fy if fs > fy else -fy
        return fs

    return get_fs_elastoplastic


def plot_displacement(time_steps, displacement, text=None):
    plt.plot(time_steps, displacement, marker=".")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement")
    plt.title(text)
    plt.show()
