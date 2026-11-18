"""F4 本地评分：按《赛题》第四节的公式，在你自己切出的验证集上算标准版（B）和严格版（A）得分。

用法：
  python tools/score.py --truth holdout.csv --pred pred.csv

--truth：含 pkg_no、lgs_company_nm、y_total_hours 三列的 CSV（例如从训练集切出的验证集）
--pred ：含 pkg_no、pred_hours 两列的 CSV，格式同正式提交

只用于自检。正式评分用的是主办方保管的测试集真实时效。
"""
import argparse
import sys

import numpy as np
import pandas as pd


def score(g, y, T):
    """g：服务商；y：实际时效；T：预估时效。返回 ME、两项罚分和 A、B 两个得分。"""
    d = pd.DataFrame({"g": np.asarray(g), "y": np.asarray(y, float), "T": np.asarray(T, float)})
    me = abs(d["T"].mean() - d["y"].mean())
    r = (d["y"] <= d["T"]).groupby(d["g"]).mean()                 # 准时率
    s = ((d["y"] - d["T"]) > 24).groupby(d["g"]).mean()           # 严重超时占比
    pen_on = 5 * np.maximum(0, (0.90 - r) * 100).sum()
    pen_sev = 5 * np.maximum(0, (s - 0.03) * 100).sum()
    return dict(ME=me, pen_ontime=pen_on, pen_severe=pen_sev, B=me + pen_on, A=me + pen_on + pen_sev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()
    t = pd.read_csv(a.truth, usecols=["pkg_no", "lgs_company_nm", "y_total_hours"])
    p = pd.read_csv(a.pred)
    errs = []
    if p["pkg_no"].duplicated().any():
        errs.append("pkg_no 有重复")
    if set(p["pkg_no"]) != set(t["pkg_no"]):
        errs.append("pkg_no 集合与验证集不一致（缺少或多出）")
    v = pd.to_numeric(p["pred_hours"], errors="coerce")
    if not (np.isfinite(v) & (v > 0)).all():
        errs.append("pred_hours 须为有限正数")
    if errs:
        print("提交格式有问题：", "；".join(errs)); sys.exit(1)
    m = t.merge(p, on="pkg_no")
    res = score(m["lgs_company_nm"], m["y_total_hours"], m["pred_hours"])
    for k, v in res.items():
        print(f"{k:10s} {v:,.4f}")


if __name__ == "__main__":
    main()
