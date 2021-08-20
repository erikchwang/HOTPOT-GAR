from utility import *

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("dataset_path")
argument_parser.add_argument("solution_path")

if bert_size != 0:
    bert_server = server.BertServer(
        server.get_args_parser().parse_args(
            [
                "-max_seq_len", "NONE",
                "-pooling_strategy", "NONE",
                "-pooling_layer", "-1", "-2", "-3", "-4",
                "-model_dir", bert_path,
                "-max_batch_size", "{}".format(batch_size)
            ]
        )
    )

    bert_server.start()
    bert_client = client.BertClient(output_fmt="list")

else:
    bert_server = None
    bert_client = None

with multiprocessing.Pool(psutil.cpu_count(False)) as pool:
    target_dataset = load_file(argument_parser.parse_args().dataset_path, "json")
    enwiki_lookup = load_file(enwiki_lookup_path, "json")
    glove_vocabulary = load_file(glove_vocabulary_path, "json")
    word_tokenizer = spacy.load(name="en_core_web_sm", disable=["tagger", "parser", "ner"])

    target_composite = pool.map(
        func=functools.partial(
            convert_dataset,
            word_tokenizer=word_tokenizer,
            enwiki_lookup=enwiki_lookup,
            glove_vocabulary=glove_vocabulary,
            require_label=False
        ),
        iterable=target_dataset
    )

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
    MODEL_PREDICT_BATCH = tf.get_collection("MODEL_PREDICT_BATCH")
    MODEL_SMOOTH = tf.get_collection("MODEL_SMOOTH")[0]

    with tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=tf.GPUOptions(allow_growth=True)
            )
    ) as SESSION:
        SAVER.restore(sess=SESSION, save_path=model_storage_path)
        SESSION.run(MODEL_SMOOTH)

        target_solution = run_model(
            SESSION, MODEL_PREDICT_BATCH,
            CONTEXT_SYMBOLS_BATCH, QUESTION_SYMBOLS_BATCH,
            CONTEXT_NUMBERS_BATCH, QUESTION_NUMBERS_BATCH,
            CONTEXT_BERTS_BATCH, QUESTION_BERTS_BATCH,
            CONTEXT_PARAGRAPHS_BATCH, QUESTION_PARAGRAPHS_BATCH,
            CONTEXT_SENTENCES_BATCH, QUESTION_SENTENCES_BATCH,
            SENTENCE_PARAGRAPH_BATCH, PARAGRAPH_LINKS_BATCH, SENTENCE_LINKS_BATCH,
            target_composite, bert_client, word_tokenizer, False
        )

        dump_file(target_solution, argument_parser.parse_args().solution_path, "json")

if bert_client is not None:
    bert_client.close()

if bert_server is not None:
    bert_server.close()
