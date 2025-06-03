import json
import random

with open('dataset/dev_anno.jsonl', 'r', encoding='utf-8') as f:
    dev_anno = [json.loads(line) for line in f.readlines()]

with open('dataset/dev_shapleyed.json', 'r', encoding='utf-8') as f:
    dev_shapleyed = [json.loads(line) for line in f]

with open('dataset/train_anno.jsonl', 'r', encoding='utf-8') as f:
    train_anno = [json.loads(line) for line in f.readlines()]

with open('dataset/train.json') as f:
    train_data = [json.loads(line) for line in f.readlines()]


output_yes = []
output_no = []
output_train = []

for shapleyed_item in dev_shapleyed:
    item_id = shapleyed_item['id']
    anno_item = next((item for item in dev_anno if item['id'] == item_id), None)

    instruction = f"{shapleyed_item['code']}\n"
    if anno_item and anno_item.get('Line Number') is not None:
        instruction += f"Suspect lines: {anno_item['Line Number']}"

    output_item = {
        "system": "'''\nYou are an experienced programmer expert in detecting dead code such as unused code and unreachable code. Given the following code and suspect lines (if any), respond in the following format:\nDead code: <Yes or No>\n\n(If it is dead code, do the following, otherwise, skip)\n(For each dead code)\nLine Number: <Line number>\nType: <Unused or Unreachable>\nExplanation: <Explanation for prediction>\n\n(Finally, fix the code if it is dead code, otherwise skip)\nFixed Code: <Fixed code>\n'''",
        "instruction": instruction,
        "output": ""
    }

    if 'clean' in shapleyed_item['label']:
        output_item["output"] = "Dead code: No\n"
        output_no.append(output_item)
    elif anno_item is not None:
        output_item["output"] = "Dead code: Yes\n"
        for line_num, label, explanation in zip(anno_item['Line Number'], anno_item['label'], anno_item['Explanation']):
            output_item["output"] += f"Line Number: {line_num}\nType: {label}\nExplanation: {explanation}\n"
        # output_item["output"] += f"Fixed Code: {anno_item['Fixed Code']}"
        if shapleyed_item['label'] == 'unreachable':
            output_item["output"] += f"Fixed Code: {anno_item['Fixed Code']}"
        else:
            print(anno_item)
            output_item["output"] += f"Fixed Code: {anno_item['Fixed Code']}"
        output_train.append(output_item)

for train_item in train_anno:

    item_id = train_item['id']
    print(item_id)
    train_ref = next((item for item in train_data if item['id'] == item_id), None)

    instruction = f"{train_ref['code']}\n"

    output_item = {
        "system": "'''\nYou are an experienced programmer expert in detecting dead code such as unused code and unreachable code. Given the following code and suspect lines (if any), respond in the following format:\nDead code: <Yes or No>\n\n(If it is dead code, do the following, otherwise, skip)\n(For each dead code)\nLine Number: <Line number>\nType: <Unused or Unreachable>\nExplanation: <Explanation for prediction>\n\n(Finally, fix the code if it is dead code, otherwise skip)\nFixed Code: <Fixed code>\n'''",
        "instruction": instruction,
        "output": ""
    }


    if train_item is not None:
        output_item["output"] = "Dead code: Yes\n"
        for line_num, label, explanation in zip(train_item['Line Number'], train_item['label'], train_item['Explanation']):
            output_item["output"] += f"Line Number: {line_num}\nType: {label}\nExplanation: {explanation}\n"
        output_item["output"] += f"Fixed Code: {train_item['Fixed Code']}"
        output_yes.append(output_item)

min_length = min(len(output_yes), len(output_no))

output_yes = output_yes[:min_length]
output_no = output_no[:min_length]

output = output_yes + output_no + output_train
random.shuffle(output)

with open('dataset/sft_train.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)