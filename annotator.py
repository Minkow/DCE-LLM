from openai import OpenAI
import re
import json

system_prompt = '''
You are an experienced programmer expert in fixing dead code such as unused code and unreachable code. 
Given the following code and a report locating where dead code is, you are asked to generate proper explanations for dead code, and fix the issues properly. 
First, mark line numbers for code. Do not delete even a empty line!
Believe in the dead code report! They are always correct! Explain ALL LINES mentioned in the report and DO NOT ADD ANY NEW LINE!



Respond in the following format in PLAIN TEXT,
(For each line in the report)
Line Number: <Line number>
Type: <Unused or Unreachable>
Explanation: <Explanation for prediction>

(Finally, fix the code according to the report)
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


def parse_output(input_string):
    lines = input_string.strip().split("\n")
    current_entry = {"id": "", "label": "", "Line Number": [], "Explanation": [], "Fixed Code": ""}
    is_fixed_code = False
    fixed_code_lines = []
    for line in lines:
        line.strip('*').strip('#')
        if line.startswith("Line Number:") or "Line Number:" in line:
            try:
                line_number = int(line.split(":")[1].strip())
                current_entry["Line Number"].append(line_number)
            except ValueError:
                print(f"Invalid Line Number format: {line}")
                match = re.match(r"Line Number:\s*(\d+)", line)
                if match:
                    print(f"But received Line Number format: {line}")
                    line_number = int(match.group(1))
                    current_entry["Line Number"].append(line_number)
                else:
                    print(f"Cannot parse: {line}")
                    return None
        elif line.startswith("Explanation:") or "Explanation:" in line:
            current_entry["Explanation"].append(line.split(": ", 1)[1].strip())
        elif line.startswith("Fixed Code:") or "Fixed Code:" in line:
            is_fixed_code = True
            continue
        elif is_fixed_code:
            if line.strip() == "```":
                is_fixed_code = False
            else:
                fixed_code_lines.append(line)

    if fixed_code_lines:
        fixed_code = "\n".join(fixed_code_lines).strip()
        fixed_code = re.sub(r'^```[\w]*', '', fixed_code, flags=re.MULTILINE).strip()
        fixed_code = re.sub(r'```$', '', fixed_code, flags=re.MULTILINE).strip()
        current_entry["Fixed Code"] = fixed_code

    return current_entry


api_key = ""
client = OpenAI(api_key=api_key)

def annotate(id, code, report, label):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": code + '\n' + report}
        ],
        temperature=0.0,
    )

    response = completion.choices[0].message.content
    res = parse_output(response)
    res["id"] = id
    res["label"] = label
    return res, response


f = open('dataset/train_anno.jsonl', 'w')
cnt = 0
for line in open("dataset/train.json", 'r').readlines():
    cnt += 1
    if cnt % 100 == 0:
        print('Current: ', cnt, ' Finished, previous id: ', id)
    data = json.loads(line)
    if data['id'] is not None:
        id = data["id"]
        code = data["code"]
        label = data["label"]
        dead_lineno = data["dead_lineno"]
        if "clean" not in label:
            if len(label) == 1:
                report = '\n\n'.join([f"Dead code type: {label}\nLine: {lineno} \nReason: {dead_lineno[lineno]}" for lineno in dead_lineno])
            else:
                temp = []
                for lineno in dead_lineno:
                    t_label = 'unreachable' if "unreachable" in dead_lineno[lineno] else 'unused'
                    temp.append(f"Dead code type: {t_label}\nLine: {lineno}\nReason: {dead_lineno[lineno]}")
                    report = '\n\n'.join(temp)

        else:
            report = None

        if report:
            try:
                result, response = annotate(id, code, report, label)
                if result:
                    print(response)
                    print(result)
                    f.write(json.dumps(result) + '\n')
                    f.flush()
            except Exception as e:
                print(response)
