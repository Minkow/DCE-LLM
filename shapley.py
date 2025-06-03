import argparse
import json

import torch
from tqdm import tqdm
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
)

from organized.mask_deadcode_python import mask_remove_deadcode


def load_data(json_file):
    data = []
    with open(json_file, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def predict(args, model, tokenizer, dataset, id2label):
    model.eval()
    count = {"gold": 0, "predict": 0, "correct": 0, "gold_unused": 0, "predict_unused": 0, "correct_unused": 0, "gold_unreachable": 0, "predict_unreachable": 0, "correct_unreachable": 0}
    cnt = 0
    fout = open("test_shapleyed.json", "w")
    for line in tqdm(dataset):
        cnt += 1
        code = line['code'].rstrip("\n")
        col = code.count('\n') - code.count('\\n') + 1

        if ".py" in line["file"]:
            language = "python"
        else:
            language = "java"
        line_number = [x + 1 for x in range(col)]
        batch = [code]
        try:
            batch.extend(mask_remove_deadcode(code, line_number, language, "shapley"))
        except:
            continue
        inputs = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors="pt").to(args.device)

        with torch.no_grad():
            logits = model(**inputs).logits

        original_logits = logits[0]
        pred_o = (torch.sigmoid(logits[0]) >= 0.5).int()
        logits_diff = torch.sigmoid(logits) - torch.sigmoid(original_logits)

        dead_lineno_uns = []
        dead_lineno_unr = []
        dead_lineno = []
        gold_lineno = [int(lineno) for lineno in line["dead_lineno"].keys()]
        print(pred_o)
        line["prediction"] = [id2label[i] for i, val in enumerate(pred_o) if val == 1]
        if pred_o[0]==0:
            print(line['code'])
            for i in range(len(logits_diff)):
                diff = logits_diff[i]
                if diff[1] < 0:
                    dead_lineno_uns.append((i, diff[1]))
                if diff[2] < 0:
                    dead_lineno_unr.append((i, diff[2]))
            dead_lineno_uns.sort(key=lambda x: x[1])
            dead_lineno_unr.sort(key=lambda x: x[1])
            print(dead_lineno_uns)
            print(dead_lineno_unr)
            dead_uns = [l[0] for l in dead_lineno_uns if l[1] <= dead_lineno_uns[0][1] - (dead_lineno_uns[0][1] - dead_lineno_uns[-1][1])/1]
            dead_unr = [l[0] for l in dead_lineno_unr if l[1] <= dead_lineno_unr[0][1] - (dead_lineno_unr[0][1] - dead_lineno_unr[-1][1])/5]
            if pred_o[1] == 1:
                dead_lineno = list(set(dead_lineno + dead_uns))
            if pred_o[2] == 1:
                dead_lineno = list(set(dead_lineno + dead_unr))
            print(line["id"])
            print("predict label: ", pred_o)
            print("gold label: ", line["label"])
            print("diff: ", logits_diff)
            print("predict lineno", dead_lineno)
            print("gold lineno", gold_lineno)
            line["suspect_lineno"] = dead_lineno

            count["gold"] += len(gold_lineno)
            if 'unused' in line["label"]:
                count["gold_unused"] += len(gold_lineno)
            if "unreachable" in line["label"]:
                count["gold_unreachable"] += len(gold_lineno)

            miss_lineno = []
            for lineno in gold_lineno:
                if lineno in dead_lineno:
                    count["correct"] += 1
                    if pred_o[1] == 1:
                        count["correct_unused"] += 1
                    if pred_o[2] == 1:
                        count["correct_unreachable"] += 1
                else:
                    miss_lineno.append(lineno)
            print("miss lineno: ", miss_lineno)

        fout.write(json.dumps(line))
        fout.write("\n")

    print("recall: ", count["correct"] / count["gold"])
    print("recall_unused: ", count["correct_unused"] / count["gold_unused"])
    print("recall_unreachable: ", count["correct_unreachable"] / count["gold_unreachable"])


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default="", type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--dataset_path", default="dataset", type=str)
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_train_epochs', type=int, default=42,
                        help="num_train_epochs")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    label2id = {
        'clean': 0,
        'unused': 1,
        'unreachable': 2
    }
    id2label = {v: k for k, v in label2id.items()}

    model = RobertaForSequenceClassification.from_pretrained("codebert-base", num_labels=3).to(device)
    tokenizer = RobertaTokenizerFast.from_pretrained("codebert-base")
    model.load_state_dict(torch.load("output/best_model.bin"))
    test_dataset = load_data("dataset/test.json")
    predict(args, model, tokenizer, test_dataset, id2label)


if __name__ == "__main__":
    main()

