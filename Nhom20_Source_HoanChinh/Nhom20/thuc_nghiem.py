"""Tái lập kiểm tra chéo, chọn thuộc tính, McNemar và đối chiếu Weka."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB as ReferenceNB

from main import FEATURES, LABELS, ROOT
from naive_bayes import GaussianNB


def read_split():
    output = ROOT / "ket_qua"
    clean = pd.read_csv(output / "du_lieu_sach.csv", index_col="rownames")
    train = pd.read_csv(output / "tap_huan_luyen.csv", index_col="rownames")
    test = pd.read_csv(output / "du_doan_kiem_thu.csv", index_col="rownames")
    columns = FEATURES + ["sex"]
    combined = pd.concat([train[columns], test[columns]])
    if not combined.index.is_unique or not combined.sort_index().equals(clean.sort_index()):
        raise ValueError("Tập học và tập kiểm thử phải tách biệt và bao phủ đúng dữ liệu sạch.")
    model = GaussianNB().fit(train[FEATURES], train.sex)
    if not np.array_equal(model.predict(test[FEATURES]), test.du_doan):
        raise ValueError("Dự đoán đã lưu khác lần tính lại. Hãy chạy main.py trước.")
    return clean, train, test


def arff_text(data):
    header = "@relation davis\n@attribute height numeric\n@attribute weight numeric\n"
    header += "@attribute sex {F,M}\n@data\n"
    return header + data[FEATURES + ["sex"]].to_csv(index=False, header=False, lineterminator="\n")


def read_predictions(text, test):
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith("inst#,actual,predicted,")), None)
    if header is None:
        raise ValueError("Đầu ra Weka cần ở định dạng CSV có cột inst#, actual, predicted.")
    rows = pd.read_csv(io.StringIO("\n".join(lines[header:])))
    if rows["inst#"].tolist() != list(range(1, len(test) + 1)):
        raise ValueError("Số lượng hoặc thứ tự dự đoán Weka không khớp tập kiểm thử.")
    actual = rows.actual.str.split(":").str[-1].to_numpy()
    predicted = rows.predicted.str.split(":").str[-1].to_numpy()
    if not np.array_equal(actual, test.sex) or not np.isin(predicted, LABELS).all():
        raise ValueError("Nhãn hoặc thứ tự bản ghi trong đầu ra Weka không hợp lệ.")
    return predicted


def evaluate_weka(train, test, jar, java):
    folder = ROOT / "weka"
    files = {f"train_{len(train)}.arff": arff_text(train),
             f"test_{len(test)}.arff": arff_text(test)}
    hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
              for name, text in files.items()}
    if jar is None:
        metadata = json.loads((folder / "cau_hinh.json").read_text(encoding="utf-8"))
        if hashes != metadata["sha256"]:
            raise ValueError("Dữ liệu đã khác kết quả Weka lưu sẵn. Chạy lại với --weka-jar.")
        for name, digest in hashes.items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Tệp ARFF đã thay đổi. Chạy lại với --weka-jar.")
        predictions = (folder / "weka_predictions.txt").read_text(encoding="utf-8")
    else:
        jar = jar.resolve()
        if not jar.is_file():
            raise FileNotFoundError(f"Không tìm thấy Weka tại {jar}")
        classpath = os.pathsep.join([str(jar), str(jar.parent / "*"), str(jar.parent / "lib" / "*")])
        command = [java, "-cp", classpath]
        with tempfile.TemporaryDirectory(prefix="nhom20_weka_") as temp:
            work = Path(temp)
            for name, text in files.items():
                (work / name).write_text(text, encoding="utf-8", newline="\n")
            def execute(arguments):
                result = subprocess.run(command + arguments, cwd=work, text=True,
                                        encoding="utf-8", errors="replace", capture_output=True,
                                        check=True, timeout=120)
                return result.stdout
            version = execute(["weka.core.Version"]).strip()
            arguments = ["weka.classifiers.bayes.NaiveBayes", "-t", f"train_{len(train)}.arff",
                         "-T", f"test_{len(test)}.arff", "-c", "last"]
            predictions = execute(arguments + ["-classifications", "weka.classifiers.evaluation.output.prediction.CSV"])
            evaluation = execute(arguments + ["-v"])
        # Xác nhận đầu ra trước khi thay các tệp kết quả của lần chạy trước.
        read_predictions(predictions, test)
        metadata = {"weka_version": version.splitlines()[0], "classifier": arguments[0],
                    "evaluation": "Supplied test set", "sha256": hashes}
        folder.mkdir(exist_ok=True)
        files.update({"weka_predictions.txt": predictions, "weka_evaluation.txt": evaluation,
                      "cau_hinh.json": json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"})
        for name, text in files.items():
            (folder / name).write_text(text, encoding="utf-8", newline="\n")
    predicted = read_predictions(predictions, test)
    comparison = test.reset_index()
    comparison.insert(0, "inst", range(1, len(test) + 1))
    comparison["weka"] = predicted
    comparison["trung_nhau"] = comparison.du_doan.eq(comparison.weka)
    comparison.to_csv(folder / "doi_chieu.csv", index=False)
    return comparison, metadata["weka_version"]


def run(jar=None, java="java"):
    clean, train, test = read_split()
    comparison, version = evaluate_weka(train, test, jar, java)
    output = ROOT / "ket_qua"
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(ReferenceNB(), clean[FEATURES], clean.sex, cv=folds)
    sizes = [len(indices) for _, indices in folds.split(clean[FEATURES], clean.sex)]
    pd.DataFrame({"luot": range(1, 6), "so_mau": sizes,
                  "so_dung": np.rint(scores * sizes).astype(int), "accuracy": scores
                  }).to_csv(output / "kiem_tra_cheo.csv", index=False)
    ablation = []
    for columns in [["height"], ["weight"], FEATURES]:
        predicted = GaussianNB().fit(train[columns], train.sex).predict(test[columns])
        correct = test.sex.eq(predicted)
        ablation.append({"thuoc_tinh": "+".join(columns), "so_dung": int(correct.sum()),
                         "accuracy": float(correct.mean()),
                         "F_dung": int((correct & test.sex.eq("F")).sum()),
                         "M_dung": int((correct & test.sex.eq("M")).sum())})
    pd.DataFrame(ablation).to_csv(output / "chon_thuoc_tinh.csv", index=False)
    model_ok = test.du_doan.eq(test.sex)
    baseline = train.sex.mode().iat[0]
    base_ok = test.sex.eq(baseline)
    b = int((model_ok & ~base_ok).sum())
    c = int((~model_ok & base_ok).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    counts = [[int((model_ok & base_ok).sum()), b], [c, int((~model_ok & ~base_ok).sum())]]
    pd.DataFrame(counts, index=["NB_dung", "NB_sai"], columns=["moc_dung", "moc_sai"]
                 ).to_csv(output / "mcnemar.csv")
    errors = comparison.loc[comparison.sex.ne(comparison.weka), "rownames"].tolist()
    source = "vừa chạy bằng Java" if jar else "đã lưu và kiểm tra khớp dữ liệu ARFF"
    summary = (
        f"Kiểm tra chéo 5 phần: {scores.tolist()}\n"
        f"Accuracy trung bình: {scores.mean():.6%}\n"
        f"Đúng khi dùng height, weight, cả hai: {[row['so_dung'] for row in ablation]}\n"
        f"Mốc so sánh luôn đoán {baseline}. McNemar hai phía: b={b}, c={c}, p={p:.16g}\n"
        f"Weka {version} ({source})\n"
        f"Nhãn Python và Weka trùng nhau: {comparison.trung_nhau.sum()}/{len(test)}\n"
        f"Weka đoán đúng: {comparison.sex.eq(comparison.weka).sum()}/{len(test)}\n"
        f"Mã bản ghi Weka đoán sai: {errors}\n"
    )
    (output / "thuc_nghiem_bo_sung.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weka-jar", type=Path, help="Đường dẫn weka.jar để chạy lại Weka bằng Java")
    parser.add_argument("--java", default="java", help="Tên lệnh hoặc đường dẫn java.exe")
    args = parser.parse_args()
    try:
        run(args.weka_jar, args.java)
    except subprocess.CalledProcessError as error:
        parser.exit(1, "Weka chạy lỗi. Kiểm tra Java và đường dẫn weka.jar.\n" + error.stderr)
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Không hoàn tất thực nghiệm: {error}\n")
