import datetime


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
