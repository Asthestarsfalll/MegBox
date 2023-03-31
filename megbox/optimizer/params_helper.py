from typing import Dict, List, Union

from megengine import Parameter
from megengine.module import Module


def filter_params(
    model: Module, patterns: Union[List[Dict], Dict]
) -> Union[List[Parameter], List[Dict[str, Parameter]]]:
    if not patterns:
        return model.parameters()
    params = model.named_parameters()
    # 1 for other parameters
    groups_num = len(patterns) + 1
    params_list = [[] for _ in range(groups_num)]
    if isinstance(patterns, dict):
        patterns = [patterns]
    target_names = [p["name"] for p in patterns]
    for name, param in params:
        for idx, p in enumerate(target_names):
            if p in name:
                params_list[idx].append(param)
                break
        else:
            params_list[-1].append(param)

    params_out = []
    for idx, pattern in enumerate(patterns):
        pattern["params"] = params_list[idx]
        params_out.append(pattern)

    return params_out
