from utility import *

begin_time = datetime.datetime.now()
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument(dest="step_index", type=int)

if argument_parser.parse_args().step_index == 0:
    enwiki_lookup = {"registry": {}, "catalog": []}
    element_number = {}

    for path in glob.glob("{}/*/*".format(enwiki_path)):
        for record in load_file(path, "bz2"):
            enwiki_lookup["registry"][record["title"]] = []

            for source in record["text_with_links"]:
                enwiki_lookup["registry"][record["title"]].append([])

                for entity, mention in set(
                        (urllib.parse.unquote(a.get("href")).lower(), a.string.lower())
                        for a in bs4.BeautifulSoup(source).find_all("a")
                        if a.get("href") is not None and
                           len(a.get("href")) != 0 and
                           a.string is not None and
                           len(a.string) != 0
                ).union([(record["title"].lower(), record["title"].lower())]):
                    if entity not in element_number:
                        element_number[entity] = len(enwiki_lookup["catalog"])
                        enwiki_lookup["catalog"].append(entity)

                    if mention not in element_number:
                        element_number[mention] = len(enwiki_lookup["catalog"])
                        enwiki_lookup["catalog"].append(mention)

                    enwiki_lookup["registry"][record["title"]][-1].append(
                        [
                            element_number[entity],
                            element_number[mention]
                        ]
                    )

    dump_file(enwiki_lookup, enwiki_lookup_path, "json")

elif argument_parser.parse_args().step_index == 1:
    word_tokenizer = spacy.load(name="en_core_web_sm", disable=["tagger", "parser", "ner"])
    glove_vocabulary = {}
    glove_embedding = []
    element_count = {}
    element_glove = {}

    for record in load_file(train_dataset_path, "json") + load_file(develop_dataset_path, "json"):
        for _, content in record["context"]:
            for source in content:
                for token in word_tokenizer(" ".join(source.split())):
                    element_count[token.text] = element_count[token.text] + 1 if token.text in element_count else 1

        for token in word_tokenizer(" ".join(record["question"].split())):
            element_count[token.text] = element_count[token.text] + 1 if token.text in element_count else 1

    for path in glob.glob("{}/*".format(glove_path)):
        for record in load_file(path, "text"):
            glove_items = record.split(" ")

            if glove_items[0] in element_count and glove_items[0] not in element_glove:
                element_glove[glove_items[0]] = list(float(item) for item in glove_items[1:])

    for element in sorted(element_count, key=element_count.get, reverse=True):
        if element in element_glove and element not in glove_vocabulary:
            glove_vocabulary[element] = len(glove_vocabulary)
            glove_embedding.append(element_glove[element])

    glove_vocabulary[""] = len(glove_vocabulary)
    glove_embedding.append(list(0.0 for _ in glove_embedding[-1]))
    dump_file(glove_vocabulary, glove_vocabulary_path, "json")
    dump_file(glove_embedding, glove_embedding_path, "pickle")

elif argument_parser.parse_args().step_index == 2:
    with multiprocessing.Pool(psutil.cpu_count(False)) as pool:
        train_dataset = load_file(train_dataset_path, "json")
        enwiki_lookup = load_file(enwiki_lookup_path, "json")
        glove_vocabulary = load_file(glove_vocabulary_path, "json")
        word_tokenizer = spacy.load(name="en_core_web_sm", disable=["tagger", "parser", "ner"])

        train_composite = list(
            record
            for record in pool.map(
                func=functools.partial(
                    convert_dataset,
                    word_tokenizer=word_tokenizer,
                    enwiki_lookup=enwiki_lookup,
                    glove_vocabulary=glove_vocabulary,
                    require_label=True
                ),
                iterable=train_dataset
            )
            if record is not None
        )

        dump_file(train_composite, train_composite_path, "pickle")

elif argument_parser.parse_args().step_index == 3:
    with multiprocessing.Pool(psutil.cpu_count(False)) as pool:
        develop_dataset = load_file(develop_dataset_path, "json")
        enwiki_lookup = load_file(enwiki_lookup_path, "json")
        glove_vocabulary = load_file(glove_vocabulary_path, "json")
        word_tokenizer = spacy.load(name="en_core_web_sm", disable=["tagger", "parser", "ner"])

        develop_composite = pool.map(
            func=functools.partial(
                convert_dataset,
                word_tokenizer=word_tokenizer,
                enwiki_lookup=enwiki_lookup,
                glove_vocabulary=glove_vocabulary,
                require_label=False
            ),
            iterable=develop_dataset
        )

        dump_file(develop_composite, develop_composite_path, "pickle")

else:
    raise Exception("invalid step index: {}".format(argument_parser.parse_args().step_index))

print(
    "preprocess step {}: cost {} seconds".format(
        argument_parser.parse_args().step_index,
        int((datetime.datetime.now() - begin_time).total_seconds())
    )
)
