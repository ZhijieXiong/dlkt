import argparse
import re


def cal_average(metrics):
    metrics_ = list(filter(lambda x: 0 <= x <= 1, metrics))
    if len(metrics_) == 0:
        return -1.0
    else:
        return sum(metrics_) / len(metrics_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str,
                        default=r"F:\code\myProjects\dlkt\example\result_local\our_setting_new\aux_info_dct\IPS_our_setting_new_ednet-kt1_save.txt")
    parser.add_argument("--key_words", type=str, default="test performance by best valid epoch is main metric")
    parser.add_argument("--n", help="隔几个做一次平均", type=int, default=5)
    parser.add_argument("--first_num", type=int, default=1,
                        help="从第几个小数开始算指标"
                             "如test performance by best valid epoch is main metric: 0.82201  , AUC: 0.82201  , ACC: 0.82917  , RMSE: 0.34999  , MAE: 0.23693  , "
                             "该值为1，因为从第0个不算，从第1个小数开始是指标"
                             "如evaluation of CORE (allow replace), num of sample is 39196    , performance is AUC: 0.691639 , ACC: 0.639734 , RMSE: 0.222505 , MAE: 0.430633"
                             "该值为0，第0个小数就开始算指标")
    args = parser.parse_args()
    params = vars(args)

    with open(params["file_path"], "r") as f:
        log_str = f.read()
    lines = log_str.split("\n")
    results = list(filter(lambda x: params["key_words"] in x, lines))
    results = list(map(lambda res: list(map(lambda x: float(x), re.findall("-?[01][.]\d+", res))), results))
    n = params["n"]
    first_num = params["first_num"]
    AUCs = []
    ACCs = []
    RMSEs = []
    MAEs = []
    for i in range(len(results)):
        if i != 0 and i % n == 0:
            print(f"result {i // n}:")
            print(f"AUC: {cal_average(AUCs):<8.4}, "
                  f"ACC: {cal_average(ACCs):<8.4}, "
                  f"RMSE: {cal_average(RMSEs):<8.4}, "
                  f"MAE: {cal_average(MAEs):<8.4}")
            AUCs = []
            ACCs = []
            RMSEs = []
            MAEs = []
        if (first_num+0) < len(results[i]):
            AUCs.append(results[i][first_num+0])
        if (first_num + 1) < len(results[i]):
            ACCs.append(results[i][first_num+1])
        if (first_num + 2) < len(results[i]):
            RMSEs.append(results[i][first_num+2])
        if (first_num + 3) < len(results[i]):
            MAEs.append(results[i][first_num+3])
        if i == (len(results) - 1):
            m = i % n + 1
            print(f"result {len(results) // n}:")
            print(f"AUC: {cal_average(AUCs):<8.4}, "
                  f"ACC: {cal_average(ACCs):<8.4}, "
                  f"RMSE: {cal_average(RMSEs):<8.4}, "
                  f"MAE: {cal_average(MAEs):<8.4}")
