import re

# 表格字符串
table_str = """
|    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge|      1|none  |     0|acc     |↑  |0.4104|±  |0.0144|
|             |       |none  |     0|acc_norm|↑  |0.4224|±  |0.0144|
|arc_easy     |      1|none  |     0|acc     |↑  |0.7210|±  |0.0092|
|             |       |none  |     0|acc_norm|↑  |0.6848|±  |0.0095|
|boolq        |      2|none  |     0|acc     |↑  |0.7746|±  |0.0073|
|hellaswag    |      1|none  |     0|acc     |↑  |0.5546|±  |0.0050|
|             |       |none  |     0|acc_norm|↑  |0.7342|±  |0.0044|
|openbookqa   |      1|none  |     0|acc     |↑  |0.2940|±  |0.0204|
|             |       |none  |     0|acc_norm|↑  |0.4080|±  |0.0220|
|rte          |      1|none  |     0|acc     |↑  |0.6895|±  |0.0279|
|winogrande   |      1|none  |     0|acc     |↑  |0.6709|±  |0.0132|
"""

# 定义一个函数来解析表格字符串
def parse_table(table_str: str):
    # 使用正则表达式分割字符串，去除多余的空白字符
    lines = [re.split(r'\s*\|\s*', line.strip()) for line in table_str.strip().split('\n')]
    return lines

# 解析表格字符串
parsed_table = parse_table(table_str)

# print(parsed_table)

# 提取所有 "acc" 的值
acc_values = [row[7] for row in parsed_table if row[5] == 'acc']

print(acc_values)

acc_values = list(map(float, acc_values))
print(sum(acc_values) / len(acc_values))
