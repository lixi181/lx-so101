"""按路径加载 scripts/ 下的脚本模块。

`scripts/` 不是包，测试用 importlib 直接加载脚本文件。手动加载的模块不会自动注册到
`sys.modules`，而 `dataclasses` 解析字符串注解时要求能从 `sys.modules[cls.__module__]`
拿到模块字典，所以这里显式注册；每次加载用不同的模块名，保证测试之间互不污染全局状态。

脚本之间用 `import pick_cube` 这种平铺导入（直接 `python scripts/xxx.py` 时 scripts/
就是 sys.path[0]），因此加载前把脚本所在目录加进 sys.path，保持和实际运行一致。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path: str, name: str) -> ModuleType:
    path = ROOT / relative_path
    script_dir = str(path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
