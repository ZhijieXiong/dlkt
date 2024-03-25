import datetime
import ast


def get_now_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def params2str_tool(param):
    if isinstance(param, set) or isinstance(param, list) or type(param) is bool:
        return str(param)
    elif type(param) in (int, float, str):
        return param
    else:
        return "not transform"


def params2str(params):
    params_json = {}
    for k, v in params.items():
        if type(v) is not dict:
            params_json[k] = params2str_tool(v)
        else:
            params_json[k] = params2str(v)
    return params_json


def is_valid_eval_string(in_str):
    try:
        ast.literal_eval(in_str)
        return True
    except (SyntaxError, ValueError):
        return False


def str_dict2params_tool(param):
    if is_valid_eval_string(param):
        return eval(param)
    else:
        return param


def str_dict2params(str_dict):
    params = {}
    for k, v in str_dict.items():
        if type(v) is not dict:
            params[k] = str_dict2params_tool(v)
        else:
            params[k] = str_dict2params(v)
    return params
