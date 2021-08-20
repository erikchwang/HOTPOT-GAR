from utility import *


def get_answer_facts(dataset_record):
    return [] if dataset_record["answer"] == "yes" or dataset_record["answer"] == "no" else list(
        [title, index]
        for title, content in dataset_record["context"]
        for index, source in enumerate(content)
        if dataset_record["answer"] in source
    )


train_dataset = load_file(train_dataset_path, "json")
develop_dataset = load_file(develop_dataset_path, "json")

for record in train_dataset:
    record["supporting_facts"] = list(fact for fact in get_answer_facts(record) if fact in record["supporting_facts"])

for record in develop_dataset:
    record["supporting_facts"] = list(fact for fact in get_answer_facts(record) if fact in record["supporting_facts"])

dump_file(train_dataset, train_dataset_path, "json")
dump_file(develop_dataset, develop_dataset_path, "json")
