"""Chạy python main.py để huấn luyện, kiểm thử và xuất các bảng, hình."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as ReferenceNB

from naive_bayes import GaussianNB

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
FEATURES = ["height", "weight"]
LABELS = ["F", "M"]


def load_data(output):
    raw = pd.read_csv(ROOT / "data" / "Davis.csv").set_index("rownames")
    raw.isna().sum().rename("so_o_thieu").to_csv(output / "gia_tri_thieu.csv")
    # Dòng 12 của bản Davis gốc có số đo mâu thuẫn với cặp tự khai 163 cm, 56 kg.
    suspect = (raw.index == 12) & raw.height.eq(57) & raw.weight.eq(166)
    valid = raw.sex.isin(LABELS) & raw[FEATURES].gt(0).all(axis=1) & ~suspect
    raw.loc[~valid].to_csv(output / "dong_loai.csv")
    clean = raw.loc[valid, FEATURES + ["sex"]]
    clean.to_csv(output / "du_lieu_sach.csv")
    return clean


def run(example):
    output = ROOT / "ket_qua"
    output.mkdir(exist_ok=True)
    data = load_data(output)
    train, test = train_test_split(data, test_size=0.30, stratify=data.sex, random_state=42)
    model = GaussianNB().fit(train[FEATURES], train.sex)
    predicted = model.predict(test[FEATURES])
    reference = ReferenceNB().fit(train[FEATURES], train.sex)
    if not np.array_equal(predicted, reference.predict(test[FEATURES])):
        raise RuntimeError("Hai bản cài đặt cho kết quả khác nhau. Hãy kiểm tra công thức.")
    train.to_csv(output / "tap_huan_luyen.csv")
    test.assign(du_doan=predicted).to_csv(output / "du_doan_kiem_thu.csv")
    report = classification_report(test.sex, predicted, labels=LABELS,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.drop(index="accuracy").to_csv(output / "chi_so_theo_lop.csv")
    baseline = train.sex.value_counts().idxmax()
    accuracy = accuracy_score(test.sex, predicted)
    summary = (f"Huấn luyện: {len(train)} | Kiểm thử: {len(test)}\n"
               f"Dự đoán đúng: {(test.sex == predicted).sum()}/{len(test)}\n"
               f"Accuracy: {accuracy:.2%} | Macro-F1: {report['macro avg']['f1-score']:.4f}\n"
               f"Mốc so sánh (luôn đoán {baseline}): {(test.sex == baseline).mean():.2%}")
    print(summary)
    (output / "tong_ket.txt").write_text(summary + "\n", encoding="utf-8")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(test.sex, predicted, labels=LABELS,
                                            colorbar=False, cmap="Blues", ax=ax)
    ax.set(xlabel="Nhãn dự đoán", ylabel="Nhãn thực tế", title="Ma trận nhầm lẫn trên tập kiểm thử")
    fig.savefig(output / "ma_tran_nham_lan.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots()
    for label, group in train.groupby("sex"):
        ax.scatter(group.height, group.weight, label=label, alpha=0.65)
    ax.set(xlabel="Chiều cao đo (cm)", ylabel="Cân nặng đo (kg)", title="Phân bố tập huấn luyện")
    ax.legend(title="Nhãn")
    fig.savefig(output / "phan_bo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    if example:
        print("Nhãn mô hình dự đoán:", model.predict([example])[0])
    print("Thư mục kết quả:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", nargs=2, type=float, metavar=("CM", "KG"),
                        help="Dự đoán thử từ chiều cao (cm) và cân nặng (kg)")
    args = parser.parse_args()
    if args.predict and not all(np.isfinite(x) and x > 0 for x in args.predict):
        parser.error("Chiều cao và cân nặng phải là số dương hữu hạn.")
    run(args.predict)
