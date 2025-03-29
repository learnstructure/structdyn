def fs_elastoplastic(uy=0.02, fy=36000):
    """Elasto-plastic force-displacement relation"""

    def get_fs(u):
        return fy / uy * u if u <= uy else fy

    return get_fs


def fs_hysteresis(uy=0.02, fy=36000):  # non-linear force displacement relation
    """Get resisting force for given u.
    uy is Yield deformation and fy is yield force"""

    def get_fs_hysteresis(
        u,
        u_last,
        fs_last=0,
    ):
        loading = True if u_last < u else False
        if loading:
            if u < uy:
                return fy / uy * u
            fs = fs_last + fy / uy * (u - u_last)
            fs = fs if fs < fy else fy

        else:
            fs = fs_last - (fy / uy) * (u_last - u)
            fs = fs if fs > -fy else -fy
        return fs

    return get_fs_hysteresis
