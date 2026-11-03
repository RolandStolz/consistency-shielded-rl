try:
    import tianshou  # noqa: F401
except ImportError:
    raise ImportError('tianshou is not installed. Please install acorl[tianshou]')
