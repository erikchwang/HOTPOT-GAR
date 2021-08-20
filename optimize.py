from utility import *

if bert_size != 0:
    bert_server = server.BertServer(
        server.get_args_parser().parse_args(
            [
                "-max_seq_len", "NONE",
                "-pooling_strategy", "NONE",
                "-pooling_layer", "-1", "-2", "-3", "-4",
                "-gpu_memory_fraction", "0.2",
                "-model_dir", bert_path,
                "-num_worker", "{}".format(gpu_count),
                "-max_batch_size", "{}".format(batch_size // gpu_count)
            ]
        )
    )

    bert_server.start()
    bert_client = client.BertClient(output_fmt="list")

else:
    bert_server = None
    bert_client = None

train_composite = load_file(train_composite_path, "pickle")
develop_composite = load_file(develop_composite_path, "pickle")
word_tokenizer = spacy.load(name="en_core_web_sm", disable=["tagger", "parser", "ner"])
SAVER = tf.train.import_meta_graph(model_design_path)
CONTEXT_SYMBOLS_BATCH = tf.get_collection("CONTEXT_SYMBOLS_BATCH")
CONTEXT_NUMBERS_BATCH = tf.get_collection("CONTEXT_NUMBERS_BATCH")
CONTEXT_BERTS_BATCH = tf.get_collection("CONTEXT_BERTS_BATCH")
CONTEXT_PARAGRAPHS_BATCH = tf.get_collection("CONTEXT_PARAGRAPHS_BATCH")
CONTEXT_SENTENCES_BATCH = tf.get_collection("CONTEXT_SENTENCES_BATCH")
QUESTION_SYMBOLS_BATCH = tf.get_collection("QUESTION_SYMBOLS_BATCH")
QUESTION_NUMBERS_BATCH = tf.get_collection("QUESTION_NUMBERS_BATCH")
QUESTION_BERTS_BATCH = tf.get_collection("QUESTION_BERTS_BATCH")
QUESTION_PARAGRAPHS_BATCH = tf.get_collection("QUESTION_PARAGRAPHS_BATCH")
QUESTION_SENTENCES_BATCH = tf.get_collection("QUESTION_SENTENCES_BATCH")
SENTENCE_PARAGRAPH_BATCH = tf.get_collection("SENTENCE_PARAGRAPH_BATCH")
PARAGRAPH_LINKS_BATCH = tf.get_collection("PARAGRAPH_LINKS_BATCH")
SENTENCE_LINKS_BATCH = tf.get_collection("SENTENCE_LINKS_BATCH")
PARAGRAPH_PROOF_BATCH = tf.get_collection("PARAGRAPH_PROOF_BATCH")
SENTENCE_PROOF_BATCH = tf.get_collection("SENTENCE_PROOF_BATCH")
ANSWER_SPANS_BATCH = tf.get_collection("ANSWER_SPANS_BATCH")
ANSWER_CLASS_BATCH = tf.get_collection("ANSWER_CLASS_BATCH")
MODEL_PREDICT_BATCH = tf.get_collection("MODEL_PREDICT_BATCH")
WEIGHT_DECAY = tf.get_collection("WEIGHT_DECAY")[0]
LEARNING_RATE = tf.get_collection("LEARNING_RATE")[0]
MODEL_UPDATE = tf.get_collection("MODEL_UPDATE")[0]

with tf.Session(
        config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )
) as SESSION:
    SAVER.restore(sess=SESSION, save_path=model_storage_path)
    model_progress = load_file(model_progress_path, "json")

    while True:
        early_stopping_round_count = sum(
            model_progress[index]["joint_f1"] != max(item["joint_f1"] for item in model_progress[:index + 1])
            for index in range(len(model_progress))
        )

        if early_stopping_round_count > early_stopping_round_limit:
            break

        begin_time = datetime.datetime.now()
        weight_decay = weight_decay_annealing_schedule(early_stopping_round_count)
        learning_rate = learning_rate_annealing_schedule(early_stopping_round_count)
        random.shuffle(train_composite)

        for offset in range(0, len(train_composite) // batch_size * batch_size, batch_size):
            feed_records = enrich_composite(train_composite[offset: offset + batch_size], bert_client)
            feed_dict = {WEIGHT_DECAY: weight_decay, LEARNING_RATE: learning_rate}

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
                feed_dict[PARAGRAPH_PROOF_BATCH[index]] = record["paragraph_proof"]
                feed_dict[SENTENCE_PROOF_BATCH[index]] = record["sentence_proof"]
                feed_dict[ANSWER_SPANS_BATCH[index]] = record["answer_spans"]
                feed_dict[ANSWER_CLASS_BATCH[index]] = record["answer_class"]

            SESSION.run(fetches=MODEL_UPDATE, feed_dict=feed_dict)

        develop_solution = run_model(
            SESSION, MODEL_PREDICT_BATCH,
            CONTEXT_SYMBOLS_BATCH, QUESTION_SYMBOLS_BATCH,
            CONTEXT_NUMBERS_BATCH, QUESTION_NUMBERS_BATCH,
            CONTEXT_BERTS_BATCH, QUESTION_BERTS_BATCH,
            CONTEXT_PARAGRAPHS_BATCH, QUESTION_PARAGRAPHS_BATCH,
            CONTEXT_SENTENCES_BATCH, QUESTION_SENTENCES_BATCH,
            SENTENCE_PARAGRAPH_BATCH, PARAGRAPH_LINKS_BATCH, SENTENCE_LINKS_BATCH,
            develop_composite, bert_client, word_tokenizer, True
        )

        dump_file(develop_solution, develop_solution_path, "json")

        model_progress.append(
            json.loads(
                subprocess.check_output(
                    [
                        sys.executable,
                        evaluate_script_path,
                        develop_solution_path,
                        develop_dataset_path
                    ]
                ).decode().replace("\'", "\"")
            )
        )

        dump_file(model_progress, model_progress_path, "json")

        print(
            "optimize epoch {}: cost {} seconds".format(
                len(model_progress),
                int((datetime.datetime.now() - begin_time).total_seconds())
            )
        )

        if model_progress[-1]["joint_f1"] == max(item["joint_f1"] for item in model_progress):
            SAVER.save(sess=SESSION, save_path=model_storage_path, write_meta_graph=False, write_state=False)
            print("accept {}".format(json.dumps(model_progress[-1])))

        else:
            SAVER.restore(sess=SESSION, save_path=model_storage_path)
            print("reject {}".format(json.dumps(model_progress[-1])))

if bert_client is not None:
    bert_client.close()

if bert_server is not None:
    bert_server.close()
