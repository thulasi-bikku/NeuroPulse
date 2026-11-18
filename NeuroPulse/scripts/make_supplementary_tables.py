#!/usr/bin/env python
"""Print the LaTeX of the supplementary tables from docs/supplementary_results.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "docs" / "supplementary_results.json").read_text())


def p_fmt(p):
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def table_baselines():
    rows = []
    for name, d in R["E1_baselines"].items():
        mc = d.get("mcnemar_vs_svm")
        test = "reference" if mc is None else f"{mc['svm_only_correct']} / {mc['other_only_correct']}; {p_fmt(mc['p'])}"
        lo, hi = d["pooled_ci95"]
        rows.append(f"{name.replace('k-NN (k = 5)', 'k-NN ($k=5$)')} & {d['acc']['mean']:.2f} $\\pm$ {d['acc']['sd']:.2f} & "
                    f"{d['macro_f1']['mean']:.4f} & {lo:.2f}--{hi:.2f} & {100 * d['angry_fear_error_share']:.0f} & {test} \\\\")
    return ("\\begin{table}[!htbp]\n\\centering\\small\n\\caption{Baseline comparison on the reference generator at the default "
            "separation ($s=4$), over " + str(R["meta"]["replicates"]) + " independent replicates. Accuracy and macro-F1 are "
            "means over replicates (accuracy $\\pm$ sample SD). The 95\\% interval is an exact binomial interval for accuracy "
            "pooled over all " + f"{next(iter(R['E1_baselines'].values()))['pooled_n']:,}" + " test records. The last column gives the "
            "exact McNemar counts (records only the SVM classified correctly / records only the other model classified correctly) and "
            "the two-sided $p$-value on the pooled paired predictions.}\n\\label{tab:supp_baselines}\n"
            "\\begin{tabular}{lcccrl}\n\\toprule\nModel & Accuracy (\\%) & Macro-F1 & Pooled 95\\% interval (\\%) & "
            "ANGRY/FEAR errors (\\%) & McNemar vs.\\ SVM \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")


def table_controls():
    conds = list(next(iter(R["E3_controls"].values())))
    models = list(R["E3_controls"])
    head = " & ".join(models)
    rows = [f"{c} & " + " & ".join(f"{R['E3_controls'][m][c]['mean']:.1f} $\\pm$ {R['E3_controls'][m][c]['sd']:.1f}" for m in models)
            + " \\\\" for c in conds]
    return ("\\begin{table}[!htbp]\n\\centering\\small\n\\caption{Negative controls on the reference generator (test accuracy in \\%, "
            "mean $\\pm$ SD over " + str(R["meta"]["replicates"]) + " replicates). Chance level is 20\\%.}\n\\label{tab:supp_controls}\n"
            "\\begin{tabular}{lccc}\n\\toprule\nCondition & " + head + " \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table}")


def table_channel():
    models = list(R["E6_channel_neutralisation"]["unmodified"])
    chans = list(next(iter(R["E6_channel_neutralisation"]["neutralised"].values())))
    rows = [f"{m} & {R['E6_channel_neutralisation']['unmodified'][m]['mean']:.1f} & " +
            " & ".join(f"{R['E6_channel_neutralisation']['neutralised'][m][c]['mean']:.1f}" for c in chans) + " \\\\" for m in models]
    return ("\\begin{table}[!htbp]\n\\centering\\small\n\\caption{Test accuracy (\\%) of the trained reference-generator models when the "
            "seven features of one channel group are replaced by their training means (no retraining); mean over "
            + str(R["meta"]["replicates"]) + " replicates.}\n\\label{tab:supp_channel}\n\\begin{tabular}{lccccc}\n\\toprule\n"
            "Model & Unmodified & " + " & ".join(f"No {c}" for c in chans) + " \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table}")


if __name__ == "__main__":
    print(table_baselines(), table_controls(), table_channel(), sep="\n\n")
