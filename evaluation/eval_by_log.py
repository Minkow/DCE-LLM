import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

with open('output.jsonl', 'r') as f:
    log_data_list = [json.loads(line) for line in f]

with open('../dataset/test.json', 'r') as f:
    ground_truth_data_list = [json.loads(line) for line in f]

python_y_true = []
python_y_pred = []
java_y_true = []
java_y_pred = []

python_line_y_true = []
python_line_y_pred = []
java_line_y_true = []
java_line_y_pred = []

overall_y_true = []
overall_y_pred = []

overall_line_y_true = []
overall_line_y_pred = []

def multilabel_accuracy(y_true, y_pred):
    correct = 0
    for true_labels, pred_labels in zip(y_true, y_pred):
        if set(true_labels) == set(pred_labels):
            correct += 1
    return correct / len(y_true)

mlb = MultiLabelBinarizer(classes=["clean", "unused", "unreachable"])

for log_data, ground_truth_data in zip(log_data_list, ground_truth_data_list):
    file_suffix = ground_truth_data["file"].split(".")[-1]
    if file_suffix == "py":
        y_true = python_y_true
        y_pred = python_y_pred
        line_y_true = python_line_y_true
        line_y_pred = python_line_y_pred
    elif file_suffix == "java":
        y_true = java_y_true
        y_pred = java_y_pred
        line_y_true = java_line_y_true
        line_y_pred = java_line_y_pred
    else:
        continue

    log_labels = set([label.lower() for label in log_data["label"]]) if log_data["label"] else ['clean']
    ground_truth_labels = set([label.lower() for label in ground_truth_data["label"]])

    y_true.append(list(ground_truth_labels))
    y_pred.append(list(log_labels))

    overall_y_true.append(list(ground_truth_labels))
    overall_y_pred.append(list(log_labels))


    if "clean" not in ground_truth_labels:
        log_line_numbers = set(log_data["Line Number"])
        ground_truth_line_numbers = set(map(int, ground_truth_data["dead_lineno"].keys()))

        line_y_true.append(list(ground_truth_line_numbers))
        line_y_pred.append(list(log_line_numbers))

        overall_line_y_true.append(list(ground_truth_line_numbers))
        overall_line_y_pred.append(list(log_line_numbers))

python_y_true_bin = mlb.fit_transform(python_y_true)
python_y_pred_bin = mlb.transform(python_y_pred)

java_y_true_bin = mlb.fit_transform(java_y_true)
java_y_pred_bin = mlb.transform(java_y_pred)

overall_y_true_bin = mlb.fit_transform(overall_y_true)
overall_y_pred_bin = mlb.transform(overall_y_pred)

python_report = classification_report(python_y_true_bin, python_y_pred_bin, zero_division=1, target_names=mlb.classes_,
                                      output_dict=True)
java_report = classification_report(java_y_true_bin, java_y_pred_bin, zero_division=1, target_names=mlb.classes_,
                                    output_dict=True)
overall_report = classification_report(overall_y_true_bin, overall_y_pred_bin, zero_division=1,
                                       target_names=mlb.classes_, output_dict=True)

def compute_line_metrics(y_true_list, y_pred_list):
    all_y_true = []
    all_y_pred = []
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        for line in y_true:
            if line in y_pred:
                all_y_true.append(1)
                all_y_pred.append(1)
            else:
                all_y_true.append(1)
                all_y_pred.append(0)
        for line in y_pred:
            if line not in y_true:
                all_y_true.append(0)
                all_y_pred.append(1)

    precision = precision_score(all_y_true, all_y_pred, zero_division=1)
    recall = recall_score(all_y_true, all_y_pred, zero_division=1)
    f1 = f1_score(all_y_true, all_y_pred, zero_division=1)

    return {
        "precision": precision,
        "recall": recall,
        "f1-score": f1
    }


python_line_report = compute_line_metrics(python_line_y_true, python_line_y_pred)
java_line_report = compute_line_metrics(java_line_y_true, java_line_y_pred)
overall_line_report = compute_line_metrics(overall_line_y_true, overall_line_y_pred)

python_accuracy = multilabel_accuracy(python_y_true, python_y_pred)
java_accuracy = multilabel_accuracy(java_y_true, java_y_pred)
overall_accuracy = multilabel_accuracy(overall_y_true, overall_y_pred)

import pprint

pp = pprint.PrettyPrinter(indent=4)

print("Python Metrics:")
pp.pprint(python_report)
print("Python Accuracy:", python_accuracy)

print("Java Metrics:")
pp.pprint(java_report)
print("Java Accuracy:", java_accuracy)

print("Overall Metrics:")
pp.pprint(overall_report)
print("Overall Accuracy:", overall_accuracy)
