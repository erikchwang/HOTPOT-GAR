import logging, os, warnings

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ZEROMQ_SOCK_TMP_DIR"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), "other")
warnings.filterwarnings("ignore")

import argparse, bs4, bz2, datetime, functools, glob, itertools, json, multiprocessing, numpy
import pickle, psutil, random, spacy, subprocess, sys, urllib
import tensorflow as tf, tensorflow_hub as tfh
from bert_serving import client, server

gpu_count = 4
batch_size = 32
bert_size = 4096
token_limit = 510
head_count = 8
block_count = 3
layer_size = 256
inner_size = 1024
dropout_rate = 0.1
reasoning_graph_hop_count = 2
paragraph_proof_loss_weight = 1.0
sentence_proof_loss_weight = 5.0
answer_class_loss_weight = 1.0
fact_proof_probability_exponent = 2.0
fact_proof_probability_threshold = 0.4
exponential_moving_average_decay = 0.9995
early_stopping_round_limit = 4
weight_decay_annealing_schedule = lambda input: 0.0001 * 0.5 ** input
learning_rate_annealing_schedule = lambda input: 0.0003 * 0.5 ** input
weight_decay_skip_terms = ["bias", "norm"]
elmo_url = "https://tfhub.dev/google/elmo/3"
enwiki_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "enwiki")
glove_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "glove")
bert_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bert")
train_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "train_dataset")
develop_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "develop_dataset")
evaluate_script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "evaluate_script")
enwiki_lookup_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "enwiki_lookup")
glove_vocabulary_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "glove_vocabulary")
glove_embedding_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "glove_embedding")
train_composite_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "train_composite")
develop_composite_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "develop_composite")
develop_solution_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "develop_solution")
model_design_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model", "model_design")
model_storage_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model", "model_storage")
model_progress_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model", "model_progress")


def load_file(file_path, file_type):
    if file_type == "pickle":
        with open(file_path, "rb") as stream:
            return pickle.load(stream)

    elif file_type == "json":
        with open(file_path, "rt") as stream:
            return json.load(stream)

    elif file_type == "text":
        with open(file_path, "rt") as stream:
            return stream.read().splitlines()

    elif file_type == "bz2":
        with bz2.open(filename=file_path, mode="rt") as stream:
            return list(json.loads(item) for item in stream.read().splitlines())

    else:
        raise Exception("invalid file type: {}".format(file_type))


def dump_file(file_items, file_path, file_type):
    if file_type == "pickle":
        with open(file_path, "wb") as stream:
            pickle.dump(obj=file_items, file=stream)

    elif file_type == "json":
        with open(file_path, "wt") as stream:
            json.dump(obj=file_items, fp=stream)

    elif file_type == "text":
        with open(file_path, "wt") as stream:
            stream.write("\n".join(file_items))

    elif file_type == "bz2":
        with bz2.open(filename=file_path, mode="wt") as stream:
            stream.write("\n".join(json.dumps(item) for item in file_items))

    else:
        raise Exception("invalid file type: {}".format(file_type))


def convert_dataset(dataset_record, word_tokenizer, enwiki_lookup, glove_vocabulary, require_label):
    def get_text_symbols(text_tokens):
        return list(token.text for token in text_tokens)

    def get_text_numbers(text_tokens):
        return list(
            glove_vocabulary[token.text] if token.text in glove_vocabulary else glove_vocabulary[""]
            for token in text_tokens
        )

    def get_text_spans(text_tokens, span_string, span_offset):
        return list(
            [span_offset + start_index, span_offset + end_index]
            for start_index in range(len(text_tokens))
            if 0 <= text_tokens[start_index:].text.find(span_string) < len(text_tokens[start_index])
            for end_index in range(start_index, len(text_tokens))
            if span_string not in text_tokens[start_index:end_index].text and
            span_string in text_tokens[start_index:end_index + 1].text
        )

    paragraph_title_array = []
    paragraph_entities_array = []
    fact_location_array = []
    sentence_source_array = []
    sentence_entities_array = []
    sentence_string_array = []
    sentence_tokens_array = []
    context_symbols = []
    context_numbers = []
    context_paragraphs = []
    context_sentences = []
    question_symbols = []
    question_numbers = []
    question_paragraphs = []
    question_sentences = []
    sentence_paragraph = []
    entity_mentions = {}

    for title, content in dataset_record["context"]:
        if len("".join(content).strip()) != 0:
            paragraph_title_array.append(title)
            paragraph_entities_array.append([])
            token_count = 0

            for index, source in enumerate(content):
                if token_count < token_limit and len(source.strip()) != 0:
                    fact_location_array.append([title, index])
                    sentence_source_array.append(source)
                    sentence_entities_array.append([])
                    sentence_string_array.append(" ".join(source.split()))
                    sentence_tokens_array.append(word_tokenizer(sentence_string_array[-1])[:token_limit - token_count])
                    context_symbols.extend(get_text_symbols(sentence_tokens_array[-1]))
                    context_numbers.extend(get_text_numbers(sentence_tokens_array[-1]))
                    context_paragraphs.extend(list(len(paragraph_title_array) - 1 for _ in sentence_tokens_array[-1]))
                    context_sentences.extend(list(len(sentence_source_array) - 1 for _ in sentence_tokens_array[-1]))
                    sentence_paragraph.append(len(paragraph_title_array) - 1)

                    if title in enwiki_lookup["registry"] and index < len(enwiki_lookup["registry"][title]):
                        for entity, mention in set(
                                (enwiki_lookup["catalog"][item[0]], enwiki_lookup["catalog"][item[1]])
                                for item in enwiki_lookup["registry"][title][index]
                        ):
                            if entity not in paragraph_entities_array[-1]:
                                paragraph_entities_array[-1].append(entity)

                            if entity not in sentence_entities_array[-1]:
                                sentence_entities_array[-1].append(entity)

                            if entity not in entity_mentions:
                                entity_mentions[entity] = []

                            if mention not in entity_mentions[entity]:
                                entity_mentions[entity].append(mention)

                    token_count += len(sentence_tokens_array[-1])

    paragraph_title_array.append("")

    paragraph_entities_array.append(
        list(
            paragraph_title_array[index].lower()
            for index, entities in enumerate(paragraph_entities_array)
            if any(
                mention in dataset_record["question"].lower()
                for entity in entities
                for mention in entity_mentions[entity]
            )
        )
    )

    sentence_source_array.append(dataset_record["question"])

    sentence_entities_array.append(
        list(
            entity
            for entity in set(itertools.chain.from_iterable(sentence_entities_array))
            if any(mention in dataset_record["question"].lower() for mention in entity_mentions[entity])
        )
    )

    sentence_string_array.append(" ".join(dataset_record["question"].split()))
    sentence_tokens_array.append(word_tokenizer(sentence_string_array[-1])[:token_limit])
    question_symbols.extend(get_text_symbols(sentence_tokens_array[-1]))
    question_numbers.extend(get_text_numbers(sentence_tokens_array[-1]))
    question_paragraphs.extend(list(len(paragraph_title_array) - 1 for _ in sentence_tokens_array[-1]))
    question_sentences.extend(list(len(sentence_source_array) - 1 for _ in sentence_tokens_array[-1]))
    sentence_paragraph.append(len(paragraph_title_array) - 1)

    composite_record = {
        "paragraph_title_array": paragraph_title_array,
        "fact_location_array": fact_location_array,
        "sentence_source_array": sentence_source_array,
        "sentence_string_array": sentence_string_array,
        "context_symbols": context_symbols,
        "context_numbers": context_numbers,
        "context_paragraphs": context_paragraphs,
        "context_sentences": context_sentences,
        "question_symbols": question_symbols,
        "question_numbers": question_numbers,
        "question_paragraphs": question_paragraphs,
        "question_sentences": question_sentences,
        "sentence_paragraph": sentence_paragraph
    }

    paragraph_links = numpy.asarray(
        a=list(
            [subject_index, object_index]
            for subject_index, subject_entities in enumerate(paragraph_entities_array)
            for object_index, object_entities in enumerate(paragraph_entities_array)
            if subject_index != object_index and
            (
                    paragraph_title_array[subject_index].lower() in object_entities or
                    paragraph_title_array[object_index].lower() in subject_entities
            )
        ) or numpy.empty(shape=[0, 2], dtype=numpy.int32),
        dtype=numpy.int32
    )

    sentence_links = numpy.asarray(
        a=list(
            [subject_index, object_index]
            for subject_index, subject_entities in enumerate(sentence_entities_array)
            for object_index, object_entities in enumerate(sentence_entities_array)
            if [sentence_paragraph[subject_index], sentence_paragraph[object_index]] in paragraph_links.tolist() and
            len(set(subject_entities).intersection(object_entities)) != 0
        ) or numpy.empty(shape=[0, 2], dtype=numpy.int32),
        dtype=numpy.int32
    )

    composite_record["paragraph_links"] = paragraph_links
    composite_record["sentence_links"] = sentence_links

    if require_label:
        paragraph_proof = list(
            1 if any(
                [title, index] in dataset_record["supporting_facts"]
                for index in range(len(content))
                if [title, index] in fact_location_array
            ) else 0
            for title, content in dataset_record["context"]
            if title in set(title for title, _ in fact_location_array)
        )

        sentence_proof = list(
            1 if [title, index] in dataset_record["supporting_facts"] else 0
            for title, content in dataset_record["context"]
            for index in range(len(content))
            if [title, index] in fact_location_array
        )

        composite_record["paragraph_proof"] = paragraph_proof
        composite_record["sentence_proof"] = sentence_proof

        if dataset_record["answer"] != "yes" and dataset_record["answer"] != "no":
            answer_spans = list(
                itertools.chain.from_iterable(
                    get_text_spans(
                        sentence_tokens_array[index],
                        " ".join(dataset_record["answer"].split()),
                        sum(len(tokens) for tokens in sentence_tokens_array[:index])
                    )
                    for index in range(len(fact_location_array))
                    if sentence_proof[index] == 1
                )
            )

            composite_record["answer_spans"] = answer_spans

            if len(answer_spans) != 0:
                answer_class = 0
                composite_record["answer_class"] = answer_class

                return composite_record

        else:
            answer_spans = [list(sum(len(tokens) for tokens in sentence_tokens_array[:-1]) for _ in range(2))]
            answer_class = 1 if dataset_record["answer"] == "yes" else 2
            composite_record["answer_spans"] = answer_spans
            composite_record["answer_class"] = answer_class

            return composite_record

    else:
        question_id = dataset_record["_id"]
        composite_record["question_id"] = question_id

        return composite_record


def enrich_composite(composite_records, bert_client):
    composite_records = list(record.copy() for record in composite_records)

    if bert_client is not None:
        bert_inputs = []
        bert_bounds = []

        for record in composite_records:
            record_offset = 0 if len(bert_bounds) == 0 else bert_bounds[-1][1]
            record_length = len(record["paragraph_title_array"])

            for index in range(record_length - 1):
                paragraph_offset = record["context_paragraphs"].index(index)
                paragraph_length = record["context_paragraphs"].count(index)

                bert_inputs.append(
                    list(
                        symbol.lower()
                        for symbol in record["context_symbols"][paragraph_offset:paragraph_offset + paragraph_length]
                    )
                )

            bert_inputs.append(list(symbol.lower() for symbol in record["question_symbols"]))
            bert_bounds.append([record_offset, record_offset + record_length])

        bert_outputs = list(
            output[1:len(input) + 1]
            for input, output in zip(bert_inputs, bert_client.encode(texts=bert_inputs, is_tokenized=True))
        )

        for record, bound in zip(composite_records, bert_bounds):
            context_berts = list(itertools.chain.from_iterable(bert_outputs[bound[0]:bound[1] - 1]))
            question_berts = bert_outputs[bound[1] - 1]
            record["context_berts"] = context_berts
            record["question_berts"] = question_berts

    else:
        for record in composite_records:
            context_berts = list([] for _ in record["context_symbols"])
            question_berts = list([] for _ in record["question_symbols"])
            record["context_berts"] = context_berts
            record["question_berts"] = question_berts

    return composite_records


def feed_forward(
        ELMO_MODULE, GLOVE_EMBEDDING,
        CONTEXT_SYMBOLS, QUESTION_SYMBOLS,
        CONTEXT_NUMBERS, QUESTION_NUMBERS,
        CONTEXT_BERTS, QUESTION_BERTS,
        CONTEXT_PARAGRAPHS, QUESTION_PARAGRAPHS,
        CONTEXT_SENTENCES, QUESTION_SENTENCES,
        SENTENCE_PARAGRAPH, PARAGRAPH_LINKS, SENTENCE_LINKS,
        require_backward
):
    def get_transformer_outputs(TARGET_INPUTS, TARGET_SEGMENTS):
        def get_encoder_outputs(ENCODER_INPUTS):
            ENCODER_OUTPUTS = ENCODER_INPUTS

            for _ in range(block_count):
                ENCODER_OUTPUTS = tf.contrib.layers.layer_norm(
                    tf.math.add(
                        x=ENCODER_OUTPUTS,
                        y=tf.layers.dropout(
                            inputs=tf.layers.dense(
                                inputs=tf.concat(
                                    values=tf.unstack(
                                        tf.linalg.matmul(
                                            a=tf.layers.dropout(
                                                inputs=tf.nn.softmax(
                                                    tf.math.divide(
                                                        x=tf.linalg.matmul(
                                                            a=tf.stack(
                                                                tf.split(
                                                                    value=tf.layers.dense(
                                                                        inputs=ENCODER_OUTPUTS,
                                                                        units=ENCODER_INPUTS.shape.as_list()[1],
                                                                        use_bias=False
                                                                    ),
                                                                    num_or_size_splits=head_count,
                                                                    axis=1
                                                                )
                                                            ),
                                                            b=tf.stack(
                                                                tf.split(
                                                                    value=tf.layers.dense(
                                                                        inputs=ENCODER_OUTPUTS,
                                                                        units=ENCODER_INPUTS.shape.as_list()[1],
                                                                        use_bias=False
                                                                    ),
                                                                    num_or_size_splits=head_count,
                                                                    axis=1
                                                                )
                                                            ),
                                                            transpose_b=True
                                                        ),
                                                        y=(ENCODER_INPUTS.shape.as_list()[1] / head_count) ** 0.5
                                                    )
                                                ),
                                                rate=dropout_rate,
                                                training=require_backward
                                            ),
                                            b=tf.stack(
                                                tf.split(
                                                    value=tf.layers.dense(
                                                        inputs=ENCODER_OUTPUTS,
                                                        units=ENCODER_INPUTS.shape.as_list()[1],
                                                        use_bias=False
                                                    ),
                                                    num_or_size_splits=head_count,
                                                    axis=1
                                                )
                                            )
                                        )
                                    ),
                                    axis=1
                                ),
                                units=ENCODER_INPUTS.shape.as_list()[1],
                                use_bias=False
                            ),
                            rate=dropout_rate,
                            training=require_backward
                        )
                    )
                )

                ENCODER_OUTPUTS = tf.contrib.layers.layer_norm(
                    tf.math.add(
                        x=ENCODER_OUTPUTS,
                        y=tf.layers.dropout(
                            inputs=tf.layers.dense(
                                inputs=tf.layers.dropout(
                                    inputs=tf.layers.dense(
                                        inputs=ENCODER_OUTPUTS,
                                        units=inner_size,
                                        activation=lambda INPUTS: tf.math.multiply(
                                            x=tf.math.divide(x=INPUTS, y=2.0),
                                            y=tf.math.add(x=tf.math.erf(tf.math.divide(x=INPUTS, y=2.0 ** 0.5)), y=1.0)
                                        )
                                    ),
                                    rate=dropout_rate,
                                    training=require_backward
                                ),
                                units=ENCODER_INPUTS.shape.as_list()[1]
                            ),
                            rate=dropout_rate,
                            training=require_backward
                        )
                    )
                )

            return ENCODER_OUTPUTS

        VALID_LENGTHS = tf.unique_with_counts(TARGET_SEGMENTS)[2]
        VALID_INDICES = tf.where(tf.sequence_mask(VALID_LENGTHS))

        return tf.gather_nd(
            params=tf.map_fn(
                fn=lambda INPUTS: tf.scatter_nd(
                    indices=tf.expand_dims(input=tf.range(INPUTS[1]), axis=1),
                    updates=get_encoder_outputs(INPUTS[0][:INPUTS[1]]),
                    shape=tf.shape(INPUTS[0])
                ),
                elems=[
                    tf.scatter_nd(
                        indices=VALID_INDICES,
                        updates=TARGET_INPUTS,
                        shape=[
                            tf.size(VALID_LENGTHS),
                            tf.math.reduce_max(VALID_LENGTHS),
                            TARGET_INPUTS.shape.as_list()[1]
                        ]
                    ),
                    VALID_LENGTHS
                ],
                dtype=tf.float32
            ),
            indices=VALID_INDICES
        )

    def get_foundation_outputs(TARGET_INPUTS, TARGET_SEGMENTS):
        VALID_LENGTHS = tf.unique_with_counts(TARGET_SEGMENTS)[2]
        VALID_INDICES = tf.where(tf.sequence_mask(VALID_LENGTHS))

        return tf.layers.dropout(
            inputs=tf.math.add(
                x=tf.layers.dense(
                    inputs=TARGET_INPUTS,
                    units=layer_size,
                    activation=lambda INPUTS: tf.math.multiply(
                        x=tf.math.divide(x=INPUTS, y=2.0),
                        y=tf.math.add(x=tf.math.erf(tf.math.divide(x=INPUTS, y=2.0 ** 0.5)), y=1.0)
                    )
                ),
                y=tf.gather(
                    params=tf.get_variable(name="POSITION_EMBEDDING", shape=[token_limit, layer_size]),
                    indices=tf.gather_nd(
                        params=tf.broadcast_to(
                            input=tf.expand_dims(input=tf.range(tf.math.reduce_max(VALID_LENGTHS)), axis=0),
                            shape=[tf.size(VALID_LENGTHS), tf.math.reduce_max(VALID_LENGTHS)]
                        ),
                        indices=VALID_INDICES
                    )
                )
            ),
            rate=dropout_rate,
            training=require_backward
        )

    def get_aggregation_outputs(TARGET_INPUTS, TARGET_SEGMENTS):
        VALID_LENGTHS = tf.unique_with_counts(TARGET_SEGMENTS)[2]
        VALID_INDICES = tf.where(tf.sequence_mask(VALID_LENGTHS))

        return tf.math.reduce_sum(
            input_tensor=tf.math.multiply(
                x=tf.expand_dims(
                    input=tf.sparse.to_dense(
                        tf.sparse.softmax(
                            tf.sparse.SparseTensor(
                                indices=VALID_INDICES,
                                values=tf.squeeze(
                                    input=tf.layers.dense(inputs=TARGET_INPUTS, units=1, use_bias=False),
                                    axis=[1]
                                ),
                                dense_shape=[tf.size(VALID_LENGTHS), tf.math.reduce_max(VALID_LENGTHS)]
                            )
                        )
                    ),
                    axis=2
                ),
                y=tf.scatter_nd(
                    indices=VALID_INDICES,
                    updates=TARGET_INPUTS,
                    shape=[tf.size(VALID_LENGTHS), tf.math.reduce_max(VALID_LENGTHS), TARGET_INPUTS.shape.as_list()[1]]
                )
            ),
            axis=1
        )

    def get_combination_outputs(SUBJECT_INPUTS, OBJECT_INPUTS):
        TRANSFORM_GATES = tf.math.multiply(
            x=tf.layers.dense(
                inputs=tf.concat(
                    values=[
                        SUBJECT_INPUTS,
                        OBJECT_INPUTS,
                        tf.math.multiply(x=SUBJECT_INPUTS, y=OBJECT_INPUTS),
                        tf.math.subtract(x=SUBJECT_INPUTS, y=OBJECT_INPUTS)
                    ],
                    axis=1
                ),
                units=1,
                activation=tf.math.sigmoid
            ),
            y=tf.dtypes.cast(
                x=tf.math.reduce_any(
                    input_tensor=tf.math.not_equal(x=OBJECT_INPUTS, y=tf.zeros_like(OBJECT_INPUTS)),
                    axis=1,
                    keepdims=True
                ),
                dtype=tf.float32
            )
        )

        TRANSFORM_INFOS = tf.layers.dense(
            inputs=tf.concat(
                values=[
                    SUBJECT_INPUTS,
                    OBJECT_INPUTS,
                    tf.math.multiply(x=SUBJECT_INPUTS, y=OBJECT_INPUTS),
                    tf.math.subtract(x=SUBJECT_INPUTS, y=OBJECT_INPUTS)
                ],
                axis=1
            ),
            units=SUBJECT_INPUTS.shape.as_list()[1],
            activation=lambda INPUTS: tf.math.multiply(
                x=tf.math.divide(x=INPUTS, y=2.0),
                y=tf.math.add(x=tf.math.erf(tf.math.divide(x=INPUTS, y=2.0 ** 0.5)), y=1.0)
            )
        )

        return tf.math.add(
            x=tf.math.multiply(x=TRANSFORM_GATES, y=TRANSFORM_INFOS),
            y=tf.math.multiply(x=tf.math.subtract(x=1.0, y=TRANSFORM_GATES), y=SUBJECT_INPUTS)
        )

    def get_similarity_outputs(SUBJECT_INPUTS, OBJECT_INPUTS):
        SUBJECT_WEIGHT = tf.get_variable(name="SUBJECT_WEIGHT", shape=[1, SUBJECT_INPUTS.shape.as_list()[1]])
        OBJECT_WEIGHT = tf.get_variable(name="OBJECT_WEIGHT", shape=[1, SUBJECT_INPUTS.shape.as_list()[1]])
        PRODUCT_WEIGHT = tf.get_variable(name="PRODUCT_WEIGHT", shape=[1, SUBJECT_INPUTS.shape.as_list()[1]])
        DIFFERENCE_WEIGHT = tf.get_variable(name="DIFFERENCE_WEIGHT", shape=[1, SUBJECT_INPUTS.shape.as_list()[1]])

        return tf.math.add_n(
            [
                tf.broadcast_to(
                    input=tf.linalg.matmul(a=SUBJECT_INPUTS, b=SUBJECT_WEIGHT, transpose_b=True),
                    shape=[tf.shape(SUBJECT_INPUTS)[0], tf.shape(OBJECT_INPUTS)[0]]
                ),
                tf.broadcast_to(
                    input=tf.linalg.matmul(a=OBJECT_WEIGHT, b=OBJECT_INPUTS, transpose_b=True),
                    shape=[tf.shape(SUBJECT_INPUTS)[0], tf.shape(OBJECT_INPUTS)[0]]
                ),
                tf.linalg.matmul(
                    a=tf.math.multiply(x=SUBJECT_INPUTS, y=PRODUCT_WEIGHT),
                    b=OBJECT_INPUTS,
                    transpose_b=True
                ),
                tf.math.subtract(
                    x=tf.linalg.matmul(a=SUBJECT_INPUTS, b=DIFFERENCE_WEIGHT, transpose_b=True),
                    y=tf.linalg.matmul(a=DIFFERENCE_WEIGHT, b=OBJECT_INPUTS, transpose_b=True)
                )
            ]
        )

    def get_distribution_outputs(SUBJECT_INPUT, OBJECT_INPUTS):
        return tf.linalg.matmul(
            a=tf.layers.dense(inputs=SUBJECT_INPUT, units=OBJECT_INPUTS.shape.as_list()[1], use_bias=False),
            b=OBJECT_INPUTS,
            transpose_b=True
        )

    with tf.variable_scope("FEATURE"):
        with tf.variable_scope(name_or_scope="FEATURE", reuse=None):
            CONTEXT_FEATURE_CODES = get_transformer_outputs(
                get_foundation_outputs(
                    tf.concat(
                        values=[
                            tf.squeeze(
                                input=ELMO_MODULE(
                                    inputs={
                                        "tokens": tf.expand_dims(input=CONTEXT_SYMBOLS, axis=0),
                                        "sequence_len": tf.expand_dims(input=tf.size(CONTEXT_SYMBOLS), axis=0)
                                    },
                                    signature="tokens",
                                    as_dict=True
                                )["word_emb"],
                                axis=[0]
                            ),
                            tf.gather(params=GLOVE_EMBEDDING, indices=CONTEXT_NUMBERS),
                            CONTEXT_BERTS
                        ],
                        axis=1
                    ),
                    CONTEXT_PARAGRAPHS
                ),
                CONTEXT_PARAGRAPHS
            )

        with tf.variable_scope(name_or_scope="FEATURE", reuse=True):
            QUESTION_FEATURE_CODES = get_transformer_outputs(
                get_foundation_outputs(
                    tf.concat(
                        values=[
                            tf.squeeze(
                                input=ELMO_MODULE(
                                    inputs={
                                        "tokens": tf.expand_dims(input=QUESTION_SYMBOLS, axis=0),
                                        "sequence_len": tf.expand_dims(input=tf.size(QUESTION_SYMBOLS), axis=0)
                                    },
                                    signature="tokens",
                                    as_dict=True
                                )["word_emb"],
                                axis=[0]
                            ),
                            tf.gather(params=GLOVE_EMBEDDING, indices=QUESTION_NUMBERS),
                            QUESTION_BERTS
                        ],
                        axis=1
                    ),
                    QUESTION_PARAGRAPHS
                ),
                QUESTION_PARAGRAPHS
            )

    with tf.variable_scope("MEMORY"):
        with tf.variable_scope("COARSE"):
            MUTUAL_SIMILARITY_MATRIX = get_similarity_outputs(CONTEXT_FEATURE_CODES, QUESTION_FEATURE_CODES)

            with tf.variable_scope(name_or_scope="COARSE", reuse=None):
                CONTEXT_MEMORY_CODES = get_combination_outputs(
                    CONTEXT_FEATURE_CODES,
                    tf.linalg.matmul(a=tf.nn.softmax(MUTUAL_SIMILARITY_MATRIX), b=QUESTION_FEATURE_CODES)
                )

            with tf.variable_scope(name_or_scope="COARSE", reuse=True):
                QUESTION_MEMORY_CODES = get_combination_outputs(
                    QUESTION_FEATURE_CODES,
                    tf.linalg.matmul(a=tf.nn.softmax(tf.transpose(MUTUAL_SIMILARITY_MATRIX)), b=CONTEXT_FEATURE_CODES)
                )

            CONTEXT_MEMORY_CODES = get_transformer_outputs(CONTEXT_MEMORY_CODES, CONTEXT_PARAGRAPHS)
            QUESTION_MEMORY_CODES = get_transformer_outputs(QUESTION_MEMORY_CODES, QUESTION_PARAGRAPHS)

        with tf.variable_scope("REFINED"):
            CONTEXT_MEMORY_CODES = get_transformer_outputs(
                get_combination_outputs(
                    CONTEXT_MEMORY_CODES,
                    tf.linalg.matmul(
                        a=tf.nn.softmax(
                            tf.math.add(
                                x=get_similarity_outputs(CONTEXT_MEMORY_CODES, CONTEXT_MEMORY_CODES),
                                y=tf.math.log(
                                    tf.dtypes.cast(
                                        x=tf.math.not_equal(
                                            x=tf.expand_dims(input=CONTEXT_PARAGRAPHS, axis=1),
                                            y=tf.expand_dims(input=CONTEXT_PARAGRAPHS, axis=0)
                                        ),
                                        dtype=tf.float32
                                    )
                                )
                            )
                        ),
                        b=CONTEXT_MEMORY_CODES
                    )
                ),
                CONTEXT_PARAGRAPHS
            )

    with tf.variable_scope("REASONING"):
        PARAGRAPH_REASONING_CODES = get_aggregation_outputs(
            tf.concat(values=[CONTEXT_MEMORY_CODES, QUESTION_MEMORY_CODES], axis=0),
            tf.concat(values=[CONTEXT_PARAGRAPHS, QUESTION_PARAGRAPHS], axis=0)
        )

        SENTENCE_REASONING_CODES = get_aggregation_outputs(
            tf.concat(values=[CONTEXT_MEMORY_CODES, QUESTION_MEMORY_CODES], axis=0),
            tf.concat(values=[CONTEXT_SENTENCES, QUESTION_SENTENCES], axis=0)
        )

        for index in range(reasoning_graph_hop_count):
            with tf.variable_scope("PARAGRAPH_PARAGRAPH_{}".format(index)):
                PARAGRAPH_REASONING_CODES = get_combination_outputs(
                    PARAGRAPH_REASONING_CODES,
                    tf.linalg.matmul(
                        a=tf.sparse.to_dense(
                            tf.sparse.softmax(
                                tf.sparse.SparseTensor(
                                    indices=tf.dtypes.cast(x=PARAGRAPH_LINKS, dtype=tf.int64),
                                    values=tf.gather_nd(
                                        params=get_similarity_outputs(
                                            PARAGRAPH_REASONING_CODES,
                                            PARAGRAPH_REASONING_CODES
                                        ),
                                        indices=PARAGRAPH_LINKS
                                    ),
                                    dense_shape=[
                                        tf.shape(PARAGRAPH_REASONING_CODES)[0],
                                        tf.shape(PARAGRAPH_REASONING_CODES)[0]
                                    ]
                                )
                            )
                        ),
                        b=PARAGRAPH_REASONING_CODES
                    )
                )

            with tf.variable_scope("PARAGRAPH_SENTENCE_{}".format(index)):
                SENTENCE_REASONING_CODES = get_combination_outputs(
                    SENTENCE_REASONING_CODES,
                    tf.gather(params=PARAGRAPH_REASONING_CODES, indices=SENTENCE_PARAGRAPH)
                )

            with tf.variable_scope("SENTENCE_SENTENCE_{}".format(index)):
                SENTENCE_REASONING_CODES = get_combination_outputs(
                    SENTENCE_REASONING_CODES,
                    tf.linalg.matmul(
                        a=tf.sparse.to_dense(
                            tf.sparse.softmax(
                                tf.sparse.SparseTensor(
                                    indices=tf.dtypes.cast(x=SENTENCE_LINKS, dtype=tf.int64),
                                    values=tf.gather_nd(
                                        params=get_similarity_outputs(
                                            SENTENCE_REASONING_CODES,
                                            SENTENCE_REASONING_CODES
                                        ),
                                        indices=SENTENCE_LINKS
                                    ),
                                    dense_shape=[
                                        tf.shape(SENTENCE_REASONING_CODES)[0],
                                        tf.shape(SENTENCE_REASONING_CODES)[0]
                                    ]
                                )
                            )
                        ),
                        b=SENTENCE_REASONING_CODES
                    )
                )

            with tf.variable_scope("SENTENCE_PARAGRAPH_{}".format(index)):
                PARAGRAPH_REASONING_CODES = get_combination_outputs(
                    PARAGRAPH_REASONING_CODES,
                    tf.linalg.matmul(
                        a=tf.nn.softmax(
                            tf.math.add(
                                x=get_similarity_outputs(PARAGRAPH_REASONING_CODES, SENTENCE_REASONING_CODES),
                                y=tf.math.log(
                                    tf.dtypes.cast(
                                        x=tf.math.equal(
                                            x=tf.expand_dims(
                                                input=tf.range(tf.shape(PARAGRAPH_REASONING_CODES)[0]),
                                                axis=1
                                            ),
                                            y=tf.expand_dims(input=SENTENCE_PARAGRAPH, axis=0)
                                        ),
                                        dtype=tf.float32
                                    )
                                )
                            )
                        ),
                        b=SENTENCE_REASONING_CODES
                    )
                )

    with tf.variable_scope("SUMMARY"):
        CONTEXT_SUMMARY_CODES = tf.concat(
            values=[
                CONTEXT_MEMORY_CODES,
                tf.get_variable(name="DUMMY_CODE", shape=[1, CONTEXT_MEMORY_CODES.shape.as_list()[1]])
            ],
            axis=0
        )

        QUESTION_SUMMARY_CODE = get_aggregation_outputs(QUESTION_MEMORY_CODES, QUESTION_PARAGRAPHS)

    with tf.variable_scope("OUTPUT"):
        PARAGRAPH_PROOF_DISTRIBUTION = tf.squeeze(
            input=get_distribution_outputs(PARAGRAPH_REASONING_CODES[-1:], PARAGRAPH_REASONING_CODES[:-1]),
            axis=[0]
        )

        SENTENCE_PROOF_DISTRIBUTION = tf.squeeze(
            input=get_distribution_outputs(SENTENCE_REASONING_CODES[-1:], SENTENCE_REASONING_CODES[:-1]),
            axis=[0]
        )

        ANSWER_SPAN_DISTRIBUTION = tf.concat(
            values=[
                get_distribution_outputs(QUESTION_SUMMARY_CODE, CONTEXT_SUMMARY_CODES),
                get_distribution_outputs(QUESTION_SUMMARY_CODE, CONTEXT_SUMMARY_CODES)
            ],
            axis=0
        )

        ANSWER_CLASS_DISTRIBUTION = tf.squeeze(
            input=tf.layers.dense(
                inputs=tf.concat(
                    values=[
                        tf.linalg.matmul(
                            a=tf.nn.softmax(ANSWER_SPAN_DISTRIBUTION[:1, :-1]),
                            b=CONTEXT_SUMMARY_CODES[:-1]
                        ),
                        tf.linalg.matmul(
                            a=tf.nn.softmax(ANSWER_SPAN_DISTRIBUTION[1:, :-1]),
                            b=CONTEXT_SUMMARY_CODES[:-1]
                        ),
                        CONTEXT_SUMMARY_CODES[-1:],
                        QUESTION_SUMMARY_CODE
                    ],
                    axis=1
                ),
                units=3,
                use_bias=False
            ),
            axis=[0]
        )

        return PARAGRAPH_PROOF_DISTRIBUTION, SENTENCE_PROOF_DISTRIBUTION, ANSWER_SPAN_DISTRIBUTION, ANSWER_CLASS_DISTRIBUTION


def build_update(
        ELMO_MODULE, GLOVE_EMBEDDING,
        CONTEXT_SYMBOLS_BATCH, QUESTION_SYMBOLS_BATCH,
        CONTEXT_NUMBERS_BATCH, QUESTION_NUMBERS_BATCH,
        CONTEXT_BERTS_BATCH, QUESTION_BERTS_BATCH,
        CONTEXT_PARAGRAPHS_BATCH, QUESTION_PARAGRAPHS_BATCH,
        CONTEXT_SENTENCES_BATCH, QUESTION_SENTENCES_BATCH,
        SENTENCE_PARAGRAPH_BATCH, PARAGRAPH_LINKS_BATCH, SENTENCE_LINKS_BATCH,
        PARAGRAPH_PROOF_BATCH, SENTENCE_PROOF_BATCH, ANSWER_SPANS_BATCH, ANSWER_CLASS_BATCH,
        WEIGHT_DECAY, LEARNING_RATE, EMA_MANAGER
):
    GRADIENTS_BATCH = []

    for index in range(batch_size):
        with tf.device("/GPU:{}".format(index % gpu_count)):
            with tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse=None if index == 0 else True):
                PARAGRAPH_PROOF_DISTRIBUTION, SENTENCE_PROOF_DISTRIBUTION, ANSWER_SPAN_DISTRIBUTION, ANSWER_CLASS_DISTRIBUTION = feed_forward(
                    ELMO_MODULE, GLOVE_EMBEDDING,
                    CONTEXT_SYMBOLS_BATCH[index], QUESTION_SYMBOLS_BATCH[index],
                    CONTEXT_NUMBERS_BATCH[index], QUESTION_NUMBERS_BATCH[index],
                    CONTEXT_BERTS_BATCH[index], QUESTION_BERTS_BATCH[index],
                    CONTEXT_PARAGRAPHS_BATCH[index], QUESTION_PARAGRAPHS_BATCH[index],
                    CONTEXT_SENTENCES_BATCH[index], QUESTION_SENTENCES_BATCH[index],
                    SENTENCE_PARAGRAPH_BATCH[index], PARAGRAPH_LINKS_BATCH[index], SENTENCE_LINKS_BATCH[index],
                    True
                )

                GRADIENTS_BATCH.append(
                    tf.gradients(
                        ys=[
                            tf.losses.sigmoid_cross_entropy(
                                multi_class_labels=PARAGRAPH_PROOF_BATCH[index],
                                logits=PARAGRAPH_PROOF_DISTRIBUTION,
                                weights=paragraph_proof_loss_weight
                            ),
                            tf.losses.sigmoid_cross_entropy(
                                multi_class_labels=SENTENCE_PROOF_BATCH[index],
                                logits=SENTENCE_PROOF_DISTRIBUTION,
                                weights=sentence_proof_loss_weight
                            ),
                            tf.math.reduce_mean(
                                tf.math.reduce_sum(
                                    input_tensor=tf.losses.sparse_softmax_cross_entropy(
                                        labels=ANSWER_SPANS_BATCH[index],
                                        logits=tf.broadcast_to(
                                            input=tf.expand_dims(input=ANSWER_SPAN_DISTRIBUTION, axis=0),
                                            shape=[
                                                tf.shape(ANSWER_SPANS_BATCH[index])[0],
                                                tf.shape(ANSWER_SPAN_DISTRIBUTION)[0],
                                                tf.shape(ANSWER_SPAN_DISTRIBUTION)[1]
                                            ]
                                        ),
                                        reduction=tf.losses.Reduction.NONE
                                    ),
                                    axis=1
                                )
                            ),
                            tf.losses.sparse_softmax_cross_entropy(
                                labels=ANSWER_CLASS_BATCH[index],
                                logits=ANSWER_CLASS_DISTRIBUTION,
                                weights=answer_class_loss_weight
                            )
                        ],
                        xs=tf.trainable_variables()
                    )
                )

    with tf.device("/CPU:0"):
        VARIABLES_UPDATE = tf.contrib.opt.AdamWOptimizer(
            weight_decay=WEIGHT_DECAY,
            learning_rate=LEARNING_RATE
        ).apply_gradients(
            grads_and_vars=zip(
                list(tf.math.reduce_mean(input_tensor=tf.stack(BATCH), axis=0) for BATCH in zip(*GRADIENTS_BATCH)),
                tf.trainable_variables()
            ),
            decay_var_list=list(
                VARIABLE
                for VARIABLE in tf.trainable_variables()
                if all(term not in VARIABLE.name.lower() for term in weight_decay_skip_terms)
            )
        )

        with tf.control_dependencies([VARIABLES_UPDATE]):
            MODEL_UPDATE = EMA_MANAGER.apply(tf.trainable_variables())

            return MODEL_UPDATE


def build_predict(
        ELMO_MODULE, GLOVE_EMBEDDING,
        CONTEXT_SYMBOLS_BATCH, QUESTION_SYMBOLS_BATCH,
        CONTEXT_NUMBERS_BATCH, QUESTION_NUMBERS_BATCH,
        CONTEXT_BERTS_BATCH, QUESTION_BERTS_BATCH,
        CONTEXT_PARAGRAPHS_BATCH, QUESTION_PARAGRAPHS_BATCH,
        CONTEXT_SENTENCES_BATCH, QUESTION_SENTENCES_BATCH,
        SENTENCE_PARAGRAPH_BATCH, PARAGRAPH_LINKS_BATCH, SENTENCE_LINKS_BATCH
):
    MODEL_PREDICT_BATCH = []

    for index in range(batch_size):
        with tf.device("/GPU:{}".format(index % gpu_count)):
            with tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse=True):
                PARAGRAPH_PROOF_DISTRIBUTION, SENTENCE_PROOF_DISTRIBUTION, ANSWER_SPAN_DISTRIBUTION, ANSWER_CLASS_DISTRIBUTION = feed_forward(
                    ELMO_MODULE, GLOVE_EMBEDDING,
                    CONTEXT_SYMBOLS_BATCH[index], QUESTION_SYMBOLS_BATCH[index],
                    CONTEXT_NUMBERS_BATCH[index], QUESTION_NUMBERS_BATCH[index],
                    CONTEXT_BERTS_BATCH[index], QUESTION_BERTS_BATCH[index],
                    CONTEXT_PARAGRAPHS_BATCH[index], QUESTION_PARAGRAPHS_BATCH[index],
                    CONTEXT_SENTENCES_BATCH[index], QUESTION_SENTENCES_BATCH[index],
                    SENTENCE_PARAGRAPH_BATCH[index], PARAGRAPH_LINKS_BATCH[index], SENTENCE_LINKS_BATCH[index],
                    False
                )

                VALID_LENGTHS = tf.unique_with_counts(CONTEXT_SENTENCES_BATCH[index])[2]
                VALID_INDICES = tf.where(tf.sequence_mask(VALID_LENGTHS))

                FACT_PROOF_PROBABILITY = tf.math.multiply(
                    x=tf.gather(
                        params=tf.math.sigmoid(PARAGRAPH_PROOF_DISTRIBUTION),
                        indices=SENTENCE_PARAGRAPH_BATCH[index][:-1]
                    ),
                    y=tf.math.sigmoid(SENTENCE_PROOF_DISTRIBUTION)
                )

                ANSWER_CLASS_PREDICT = tf.expand_dims(
                    input=tf.dtypes.cast(x=tf.math.argmax(ANSWER_CLASS_DISTRIBUTION), dtype=tf.int32),
                    axis=0
                )

                ANSWER_SPAN_PREDICT = tf.unravel_index(
                    indices=tf.dtypes.cast(
                        x=tf.math.argmax(
                            tf.reshape(
                                tensor=tf.linalg.band_part(
                                    input=tf.math.multiply(
                                        x=tf.linalg.matmul(
                                            a=tf.scatter_nd(
                                                indices=VALID_INDICES,
                                                updates=tf.transpose(tf.nn.softmax(ANSWER_SPAN_DISTRIBUTION[:1, :-1])),
                                                shape=[tf.size(VALID_LENGTHS), tf.math.reduce_max(VALID_LENGTHS), 1]
                                            ),
                                            b=tf.scatter_nd(
                                                indices=VALID_INDICES,
                                                updates=tf.transpose(tf.nn.softmax(ANSWER_SPAN_DISTRIBUTION[1:, :-1])),
                                                shape=[tf.size(VALID_LENGTHS), tf.math.reduce_max(VALID_LENGTHS), 1]
                                            ),
                                            transpose_b=True
                                        ),
                                        y=tf.expand_dims(
                                            input=tf.expand_dims(
                                                input=tf.math.pow(
                                                    x=FACT_PROOF_PROBABILITY,
                                                    y=fact_proof_probability_exponent
                                                ),
                                                axis=1
                                            ),
                                            axis=2
                                        )
                                    ),
                                    num_lower=0,
                                    num_upper=-1
                                ),
                                shape=[-1]
                            )
                        ),
                        dtype=tf.int32
                    ),
                    dims=[tf.size(VALID_LENGTHS), tf.math.reduce_max(VALID_LENGTHS), tf.math.reduce_max(VALID_LENGTHS)]
                )

                ANSWER_SUPPORT_PREDICT = tf.dtypes.cast(
                    x=tf.math.greater(x=FACT_PROOF_PROBABILITY, y=fact_proof_probability_threshold),
                    dtype=tf.int32
                )

                MODEL_PREDICT_BATCH.append(
                    tf.concat(
                        values=[ANSWER_CLASS_PREDICT, ANSWER_SPAN_PREDICT, ANSWER_SUPPORT_PREDICT],
                        axis=0
                    )
                )

    return MODEL_PREDICT_BATCH


def run_model(
        SESSION, MODEL_PREDICT_BATCH,
        CONTEXT_SYMBOLS_BATCH, QUESTION_SYMBOLS_BATCH,
        CONTEXT_NUMBERS_BATCH, QUESTION_NUMBERS_BATCH,
        CONTEXT_BERTS_BATCH, QUESTION_BERTS_BATCH,
        CONTEXT_PARAGRAPHS_BATCH, QUESTION_PARAGRAPHS_BATCH,
        CONTEXT_SENTENCES_BATCH, QUESTION_SENTENCES_BATCH,
        SENTENCE_PARAGRAPH_BATCH, PARAGRAPH_LINKS_BATCH, SENTENCE_LINKS_BATCH,
        target_composite, bert_client, word_tokenizer, require_parallel
):
    target_solution = {"answer": {}, "sp": {}}
    feed_size = batch_size if require_parallel else 1

    for offset in range(0, len(target_composite), feed_size):
        feed_records = enrich_composite(target_composite[offset: offset + feed_size], bert_client)
        feed_dict = {}

        for index, record in enumerate(feed_records):
            feed_dict[CONTEXT_SYMBOLS_BATCH[index]] = record["context_symbols"]
            feed_dict[CONTEXT_NUMBERS_BATCH[index]] = record["context_numbers"]
            feed_dict[CONTEXT_BERTS_BATCH[index]] = record["context_berts"]
            feed_dict[CONTEXT_PARAGRAPHS_BATCH[index]] = record["context_paragraphs"]
            feed_dict[CONTEXT_SENTENCES_BATCH[index]] = record["context_sentences"]
            feed_dict[QUESTION_SYMBOLS_BATCH[index]] = record["question_symbols"]
            feed_dict[QUESTION_NUMBERS_BATCH[index]] = record["question_numbers"]
            feed_dict[QUESTION_BERTS_BATCH[index]] = record["question_berts"]
            feed_dict[QUESTION_PARAGRAPHS_BATCH[index]] = record["question_paragraphs"]
            feed_dict[QUESTION_SENTENCES_BATCH[index]] = record["question_sentences"]
            feed_dict[SENTENCE_PARAGRAPH_BATCH[index]] = record["sentence_paragraph"]
            feed_dict[PARAGRAPH_LINKS_BATCH[index]] = record["paragraph_links"]
            feed_dict[SENTENCE_LINKS_BATCH[index]] = record["sentence_links"]

        model_predicts = SESSION.run(fetches=MODEL_PREDICT_BATCH[:len(feed_records)], feed_dict=feed_dict)

        for record, predict in zip(feed_records, model_predicts):
            if predict[0] == 0:
                target_solution["answer"][record["question_id"]] = word_tokenizer(
                    record["sentence_string_array"][predict[1]]
                )[predict[2]:predict[3] + 1].text

            else:
                target_solution["answer"][record["question_id"]] = "yes" if predict[0] == 1 else "no"

            target_solution["sp"][record["question_id"]] = list(
                location
                for index, location in enumerate(record["fact_location_array"])
                if predict[index + 4] == 1
            )

    return target_solution
