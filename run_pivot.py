import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from torch.utils.data import DataLoader, Dataset

def load_data(json_file):
    data = []
    with open(json_file, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


class CodeDataset(Dataset):
    def __init__(self, encodings, labels, clean_encodings=None):
        self.encodings = encodings
        self.labels = labels
        self.clean_encodings = clean_encodings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)

        if self.clean_encodings:
            item['contrastive_code'] = {key: val.clone().detach() for key, val in self.clean_encodings[idx].items()}

        return item


def process_data(data, tokenizer, label2id):
    codes = [item['code'] for item in data]
    contrastive_codes = [item.get('contrastive_codes', None) for item in data]

    encodings = tokenizer(codes, truncation=True, padding="max_length", max_length=512, return_tensors='pt')

    contrastive_encodings = []
    for code in contrastive_codes:
        contrastive_encodings.append(tokenizer([code if code else ""], truncation=True, padding="max_length", max_length=512, return_tensors='pt'))

    labels = [[label2id[label] for label in item['label']] for item in data]
    multi_labels = []
    for label in labels:
        multi_label = [0] * len(label2id)
        for l in label:
            multi_label[l] = 1
        multi_labels.append(multi_label)

    return encodings, multi_labels, contrastive_encodings


def is_empty(tensor, tokenizer):
    empty = tokenizer("", truncation=True, padding="max_length", max_length=512, return_tensors='pt')
    return torch.equal(tensor, empty["input_ids"])


def weighted_bce_with_logits_loss(logits, labels, weight=None):
    if weight is None:
        weight = torch.tensor([1.0, 3.0, 10.0])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight.to(logits.device))
    return loss_fn(logits, labels)


def contrastive_loss(anchor, negative, label, margin=1.0):
    euclidean_distance = torch.nn.functional.pairwise_distance(anchor, negative)
    loss = torch.mean(
        (1 - label) * torch.pow(euclidean_distance, 2) +
        label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
    )
    return loss


def train(args, model, tokenizer, train_dataloader, dev_dataloader):
    device = args.device

    def train_epoch(dataloader, model, optimizer, scheduler):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits

            for i, contrastive_input_ids in enumerate(batch['contrastive_code']['input_ids']):
                if not is_empty(contrastive_input_ids, tokenizer):
                    clean_input_ids = contrastive_input_ids.to(device)
                    clean_attention_mask = batch['contrastive_code']['attention_mask'][i].to(device)
                    negative = model(input_ids=clean_input_ids,
                                            attention_mask=clean_attention_mask).logits
                    logits = torch.cat((logits, negative), dim=0)
                    labels = torch.cat((labels, torch.tensor([[1, 0, 0]], dtype=torch.float).to(device)), dim=0)

            loss = weighted_bce_with_logits_loss(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate_epoch(dataloader, model):
        model.eval()
        total_loss = 0

        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            for i, contrastive_input_ids in enumerate(batch['contrastive_code']['input_ids']):
                anchor = logits[i]
                if not is_empty(contrastive_input_ids, tokenizer):
                    clean_input_ids = contrastive_input_ids.to(device)
                    clean_attention_mask = batch['contrastive_code']['attention_mask'][i].to(device)
                    negative = model(input_ids=clean_input_ids,
                                     attention_mask=clean_attention_mask).logits
                    logits = torch.cat((logits, negative), dim=0)
                    labels = torch.cat((labels, torch.tensor([[1, 0, 0]], dtype=torch.float).to(device)), dim=0)

            loss = weighted_bce_with_logits_loss(logits, labels)
            total_loss += loss.item()

        return total_loss / len(dataloader)

    num_epochs = args.num_train_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    max_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1,
                                                num_training_steps=max_steps)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_epoch(train_dataloader, model, optimizer, scheduler)
        val_loss = evaluate_epoch(dev_dataloader, model)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_dir = 'output_wly'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.bin'))
            print(f"New best model saved with validation loss {val_loss:.4f}")
    
        if best_val_loss < 0.05:
            return


def test(args, model, tokenizer, dataset, id2label, label2id):
    model.eval()
    f = open(os.path.join("output", "predictions.txt"), 'w')
    correct = 0
    tp = [0, 0, 0]
    fn = [0, 0, 0]
    fp = [0, 0, 0]

    for batch in tqdm(dataset):
        inputs = tokenizer(batch["code"], max_length=512, padding="max_length", truncation=True,
                           return_tensors="pt").to(
            args.device)

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_classes = (torch.sigmoid(logits) > 0.5).int().squeeze().tolist()
        predicted_labels = [id2label[i] for i, val in enumerate(predicted_classes) if val == 1]

        gold_labels = batch["label"]

        if set(predicted_labels) == set(gold_labels):
            correct += 1
        for val in gold_labels:
            if val not in predicted_labels:
                fn[label2id[val]] += 1
        for val in predicted_labels:
            if val in gold_labels:
                tp[label2id[val]] += 1
            else:
                fp[label2id[val]] += 1

        f.write(json.dumps({"id": batch["id"],
                            "answer": set(predicted_labels) == set(gold_labels),
                            "predict": predicted_labels,
                            "gold": gold_labels,
                            "file": batch["file"]}))
        f.write("\n")

    accuracy = correct / len(dataset)
    precisions = [tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0 for i in range(len(tp))]
    recalls = [tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0 for i in range(len(tp))]
    f1s = [2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i]) if (precisions[i] + recalls[i]) > 0 else 0 for
           i in range(len(tp))]

    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precisions: {precisions}\n")
    f.write(f"Recalls: {recalls}\n")
    f.write(f"F1 scores: {f1s}\n")
    f.close()


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
    parser.add_argument('--num_train_epochs', type=int, default=6,
                        help="num_train_epochs")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    model = RobertaForSequenceClassification.from_pretrained("codebert-base", num_labels=3).to(device)
    tokenizer = RobertaTokenizerFast.from_pretrained('codebert-base')

    label2id = {
        'clean': 0,
        'unused': 1,
        'unreachable': 2
    }
    id2label = {v: k for k, v in label2id.items()}

    train_data = load_data('dataset/train.json')
    encodings, labels, clean_encodings = process_data(train_data, tokenizer, label2id)
    train_dataset = CodeDataset(encodings, labels, clean_encodings)
    dev_data = load_data('dataset/dev.json')
    encodings, labels, clean_encodings = process_data(dev_data, tokenizer, label2id)
    dev_dataset = CodeDataset(encodings, labels, clean_encodings)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

    # Training
    train(args, model, tokenizer, train_dataloader, dev_dataloader)

    from datasets import load_dataset
    model.load_state_dict(torch.load("output/best_model.bin"))
    test_dataset = load_dataset('json', data_files={"test": 'dataset/test.json'})["test"]
    test(args, model, tokenizer, test_dataset, id2label, label2id)


if __name__ == "__main__":
    main()
