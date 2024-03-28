import argparse
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str,
                        default=r"F:\code\myProjects\dlkt\example\result_local\dimkt_our_setting_new_statics2011_save.txt")
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
    results = list(map(lambda res: list(map(lambda x: float(x), re.findall("0[.]\d+", res))), results))
    AUC_ave = 0
    ACC_ave = 0
    MAE_ave = 0
    RMSE_ave = 0
    n = params["n"]
    first_num = params["first_num"]
    for i in range(len(results)):
        if i != 0 and i % n == 0:
            print(f"result {i // n}:")
            print(f"AUC: {AUC_ave / n:<8.4}, ACC: {ACC_ave / n:<8.4}, RMSE: {RMSE_ave / n:<8.4}, MAE: {MAE_ave / n:<8.4}")
            AUC_ave = 0
            ACC_ave = 0
            MAE_ave = 0
            RMSE_ave = 0
        AUC_ave += results[i][first_num+0]
        ACC_ave += results[i][first_num+1]
        RMSE_ave += results[i][first_num+2]
        MAE_ave += results[i][first_num+3]
        if i == (len(results) - 1):
            m = i % n + 1
            print(f"result {len(results) // n}:")
            print(f"AUC: {AUC_ave / m:<8.4}, ACC: {ACC_ave / m:<8.4}, RMSE: {RMSE_ave / m:<8.4}, MAE: {MAE_ave / m:<8.4}")
