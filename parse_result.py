import argparse
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default=r"F:\code\myProjects\dlkt\example\result_local\qdkt\ood\our_setting_ood_by_school_assist2012_split_7_save.txt")
    parser.add_argument("--key_words", type=str, default="test performance by best valid epoch is main metric")
    parser.add_argument("--n", help="隔几个做一次平均", type=int, default=5)
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
    for i in range(len(results)):
        if i != 0 and i % n == 0:
            print(f"result {i // n}:")
            print(f"AUC: {AUC_ave / n:<8.4}, ACC: {ACC_ave / n:<8.4}, RMSE: {RMSE_ave / n:<8.4}, MAE: {MAE_ave / n:<8.4}")
            AUC_ave = 0
            ACC_ave = 0
            MAE_ave = 0
            RMSE_ave = 0
        AUC_ave += results[i][1]
        ACC_ave += results[i][2]
        RMSE_ave += results[i][3]
        MAE_ave += results[i][4]
        if i == (len(results) - 1):
            m = i % n + 1
            print(f"result {len(results) // n}:")
            print(f"AUC: {AUC_ave / m:<8.4}, ACC: {ACC_ave / m:<8.4}, RMSE: {RMSE_ave / m:<8.4}, MAE: {MAE_ave / m:<8.4}")
