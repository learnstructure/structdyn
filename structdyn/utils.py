def fs_elastoplastic(uy=0.02, fy=36000):
    """Elasto-plastic force-displacement relation"""

    def get_fs(u):
        return fy / uy * u if u <= uy else fy

    return get_fs
