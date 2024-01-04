import re
import argparse


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
    parser.add_argument("--target_python_file", type=str, default=r"F:\code\myProjects\dlkt\example\train\dimkt_instance_cl.py")
    parser.add_argument("--script_dir", type=str, default=r"F:\code\myProjects\dlkt\example\script_template")
    args = parser.parse_args()
    params = vars(args)

    with open(params["target_python_file"], 'r', encoding='utf-8') as f:
        text = f.read()

    arg_str_all = re.findall(r'parser\.add_argument\((.*?)\)', text)
    arg_parse_result = []
    for arg_str in arg_str_all:
        arg_name = re.search(r'^[\'"](.*?)[\'"]', arg_str).group(1)
        match = re.search(r'default=(?:"(.*?)"|\'(.*?)\'|([^,]+))(?=[,"]|$)', arg_str)
        if match.group(1) is None:
            arg_default = match.group(0).split("=")[1]
        else:
            arg_default = match.group(1)
        arg_parse_result.append((arg_name, arg_default))

    print(arg_parse_result)
    print(len(arg_parse_result))

