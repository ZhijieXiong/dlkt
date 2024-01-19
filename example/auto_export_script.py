import re
import os
import argparse
import platform


if __name__ == "__main__":
    # demo
    # text = 'if __name__ == \"__main__\":\n' \
    #        '\tparser = argparse.ArgumentParser()\n' \
    #        '\tparser.add_argument(\'--setting_name\', type=str, default=\"random_split_leave_multi_out_setting\", choices=(\"StepLR\", \"MultiStepLR\"))\n' \
    #        '\tparser.add_argument(\"--dataset_name\", type=str, default=\"assist2012\")\n' \
    #        '\tparser.add_argument(\"--use_early_stop\", type=str2bool, default=True)'
    # arg_str_all = re.findall(r'parser\.add_argument\((.*?)\)', text)
    # for arg_str in arg_str_all:
    #     arg_name = re.search(r'^[\'"](.*?)[\'"]', arg_str).group(1)
    #     match = re.search(r'default=(?:"(.*?)"|\'(.*?)\'|([^,]+))(?=[,"]|$)', arg_str)
    #     if match.group(1) is None:
    #         arg_default = match.group(0).split("=")[1]
    #     else:
    #         arg_default = match.group(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_python_file", type=str, default=r"F:/code/myProjects/dlkt/example/train/qdkt_matual_enhance4long_tail.py")
    parser.add_argument("--script_dir", type=str, default=r"F:/code/myProjects/dlkt/example/script_template")
    args = parser.parse_args()
    params = vars(args)

    with open(params["target_python_file"], 'r', encoding='utf-8') as f:
        text = f.read()

    script_template_path = os.path.join(
        os.path.basename(params["script_dir"]),
        os.path.basename(params["target_python_file"]).replace(".py", ".sh")
    )

    arg_str_all = re.findall(r'parser\.add_argument\(([\s\S]*?)\)', text)
    arg_parse_result = []
    for arg_str in arg_str_all:
        arg_name = re.search(r'^[\'"](.*?)[\'"]', arg_str).group(1)
        # 有些参数不好提取，就不提了，比如--multi_metrics "['AUC', 'ACC']"，这种多种引号嵌套的
        if arg_name in ["--multi_metrics"]:
            continue
        match = re.search(r'default=(?:"(.*?)"|\'(.*?)\'|([^,]+))(?=[,"]|$)', arg_str)
        if match.group(1) is None:
            arg_default = match.group(0).split("=")[1]
        else:
            arg_default = match.group(1)
        arg_parse_result.append((arg_name, arg_default))

    if platform.system() == "Windows":
        py_path = params['target_python_file'].replace("\\", "/")
    else:
        py_path = params['target_python_file']
    script_template_text = f"python {py_path} \\\n  "
    for i in range(len(arg_parse_result)):
        script_template_text += f"{arg_parse_result[i][0]} {arg_parse_result[i][1]} "
        if i != 0 and i % 5 == 0:
            script_template_text += "\\\n  "

    with open(script_template_path, "w") as f:
        f.write(script_template_text)
    print(script_template_text)

    print(arg_parse_result)
    print(len(arg_parse_result))


