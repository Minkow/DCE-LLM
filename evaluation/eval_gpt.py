from openai import OpenAI

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer


system_prompt = '''
You are an experienced programmer expert in detecting dead code such as unused code and unreachable code. Given the following code, respond in the following format:
Dead code: <Yes or No>

(If is dead code, do the following, otherwise, skip)
(For each dead code)
Line Number: <Line number>
Type: <Unused or Unreachable>
Explanation: <Explanation for prediction>

(Finally, fix the code if it is dead code, otherwise skip)
Fixed Code: <Fixed code>
'''

code = '''
def fill_str(Data):
    s1 = input()
    s2 = s1 + '<PAD>'
    s3 = s1 + '<EOS>'     
    if len(s2) == 0:
        print('Empty string')
        Data.pad_str = None
        Data.eos_str = None
    else:
        Data.pad_str = s2
        Data.eos_str = 's3'
'''

client = OpenAI(api_key="")

import re


def parse_output(input_string):
    lines = input_string.strip().split("\n")
    current_entry = {"pred": "", "id": "", "dead": "", "label": [], "Line Number": [], "Explanation": [],
                     "Fixed Code": ""}
    is_fixed_code = False
    fixed_code_lines = []
    for line in lines:
        if line.startswith("Dead code:"):
            if "No" in line or "no" in line:
                current_entry = {"id": "", "dead": "No", "label": [], "Line Number": [], "Explanation": [],
                                 "Fixed Code": ""}
                return current_entry
            else:
                current_entry["dead"] = "Yes"

        if line.startswith("Line Number:"):
            try:
                line_number = int(line.split(":")[1].strip())
                current_entry["Line Number"].append(line_number)
            except Exception as e:
                print(f"Invalid Line Number format: {line}")
                match = re.match(r"Line Number:\s*(\d+)", line)
                if match:
                    line_number = int(match.group(1))
                    current_entry["Line Number"].append(line_number)
                    print(f"Number {line_number} accetped: {line}")
                else:
                    print(f"Cannot parse: {line}")
                    current_entry["Line Number"].append(line)
        elif line.startswith("Type:"):
            current_entry["label"].append(line.split(":")[1].strip())

        elif line.startswith("Explanation:"):
            try:
                current_entry["Explanation"].append(line.split(": ", 1)[1].strip())
            except Exception as e:
                print(f"Invalid Explanation format: {line}")
                current_entry["Explanation"].append([])
        elif line.startswith("Fixed Code:"):
            is_fixed_code = True
            continue
        elif is_fixed_code:
            if line.strip() == "```":
                is_fixed_code = False
            else:
                fixed_code_lines.append(line)

    if fixed_code_lines:
        try:
            fixed_code = "\n".join(fixed_code_lines).strip()
            fixed_code = re.sub(r'^```[\w]*', '', fixed_code, flags=re.MULTILINE).strip()
            fixed_code = re.sub(r'```$', '', fixed_code, flags=re.MULTILINE).strip()
            current_entry["Fixed Code"] = fixed_code
        except Exception as e:
            current_entry["Fixed Code"] = []

    return current_entry

true_labels = []
pred_labels = []

import json
f = open('../dataset/test.json', 'r')
rec = open('output.jsonl', 'w')

for idx, line in enumerate(f.readlines()):
    line = json.loads(line)
    true_labels.append(line['label'])
    print(idx+1, 'Truth:', line['label'])

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": line["code"]}
        ],
        temperature=0.0,
    )
    res = parse_output(completion.choices[0].message.content)
    res['id'] = line['id']

    pred_res = []
    if res['dead'] == 'Yes':
        dead_code_types = set(res['label'])
        if 'unused' in dead_code_types or 'Unused' in dead_code_types:
            pred_res.append('unused')
        if 'unreachable' in dead_code_types or 'Unreachable' in dead_code_types:
            pred_res.append('unreachable')
        pred_labels.append(pred_res)
    else:
        pred_labels.append(['clean'])
    print('Prediction: ', pred_res)
    res['pred'] = pred_res
    print(res)
    print('--------------------')
    rec.write(json.dumps(res) + '\n')
    rec.flush()

mlb = MultiLabelBinarizer()
y_true = mlb.fit_transform(true_labels)
y_pred = mlb.transform(pred_labels)

report = classification_report(y_true, y_pred, target_names=mlb.classes_,
                               output_dict=True)
print(report)

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")

