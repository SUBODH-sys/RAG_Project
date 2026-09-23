import json

dataset = 'D:/RAG_Project/data/json_files/bliss_corpus.json'

with open(dataset, "r", encoding="utf-8") as file:
    dataset = json.load(file)

unique_flags = set()

for item in dataset:
    flags_list = item.get("metadata", {}).get('flags',[])

    unique_flags.update(flags_list)

result = list(unique_flags)
print(result)