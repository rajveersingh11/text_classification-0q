import os
import platform
import ctypes

if platform.system() == "Windows":
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec is not None and spec.origin is not None:
            torch_dir = os.path.dirname(spec.origin)
            c10_path = os.path.join(torch_dir, "lib", "c10.dll")
            if os.path.exists(c10_path):
                ctypes.CDLL(os.path.normpath(c10_path))
    except Exception:
        pass
