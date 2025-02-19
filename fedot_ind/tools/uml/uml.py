import os
import re

from fedot_ind.tools.serialisation.path_lib import DEFAULT_PATH_MODELS

with open(DEFAULT_PATH_MODELS, 'r') as file:
    file_content = [string for string in file.read().split(
        "\n") if not string.startswith("from")]
file_content = file_content[:next(
    (i for i, line in enumerate(file_content) if line.startswith("def")), None)]
file_content = file_content[next(
    (i for i, line in enumerate(file_content) if line == '}'), None) + 1:]

pattern = re.compile(r'\b[A-Z_]+\b')

substrings = []
for s in file_content:
    matches = pattern.findall(s)
    substrings.extend(matches)
substrings = list(set(substrings))

res = ["@startuml", "'https://plantuml.com/sequence-diagram",
       "", "class AtomizedModel {", "    Enum", "}", ""]
for ind in substrings:
    res += ["abstract " + ind]
res += ["", ""]

res += ["INDUSTRIAL_CLF_PREPROC_MODEL --> AtomizedModel",
        "AtomizedModel <-- SKLEARN_CLF_MODELS",
        "FEDOT_PREPROC_MODEL --> AtomizedModel",
        "AtomizedModel <- INDUSTRIAL_PREPROC_MODEL",
        "SKLEARN_REG_MODELS --> AtomizedModel",
        "AtomizedModel <-- FORECASTING_MODELS",
        "FORECASTING_PREPROC -> AtomizedModel",
        "NEURAL_MODEL -> AtomizedModel", ""]

for ind in substrings:
    start_index = next((i for i, line in enumerate(
        file_content) if line.endswith(ind + " = {")), None)
    end_index = next((i for i, line in enumerate(
        file_content[start_index:]) if line.endswith("}")), None) + 1 + start_index
    list_of_strings = file_content[start_index:end_index]
    pattern = re.compile(r"': ([^']*)")
    formatted_strings = [pattern.search(string).group(
        1) if pattern.search(string) else "" for string in list_of_strings]
    res = res + ["abstract " + ind + " {"] + [string for string in formatted_strings if
                                              string.strip() and string.strip() != "{"] + ["}", ""]

with open('your_file.puml', 'w') as f:
    for line in res + ["@enduml"]:
        f.write(f"{line}\n")

os.system('python -m plantuml your_file.puml')
