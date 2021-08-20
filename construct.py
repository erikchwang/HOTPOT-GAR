from utility import *

begin_time = datetime.datetime.now()
ELMO_MODULE = tfh.Module(spec=elmo_url, trainable=False)
GLOVE_EMBEDDING = tf.Variable(initial_value=load_file(glove_embedding_path, "pickle"), trainable=False)
CONTEXT_SYMBOLS_BATCH = list(tf.placeholder(dtype=tf.string, shape=[None]) for _ in range(batch_size))
CONTEXT_NUMBERS_BATCH = list(tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size))
CONTEXT_BERTS_BATCH = list(tf.placeholder(dtype=tf.float32, shape=[None, bert_size]) for _ in range(batch_size))
CONTEXT_PARAGRAPHS_BATCH = list(tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size))
CONTEXT_SENTENCES_BATCH = list(tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size))
QUESTION_SYMBOLS_BATCH = list(tf.placeholder(dtype=tf.string, shape=[None]) for _ in range(batch_size))
QUESTION_NUMBERS_BATCH = list(tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size))
QUESTION_BERTS_BATCH = list(tf.placeholder(dtype=tf.float32, shape=[None, bert_size]) for _ in range(batch_size))
QUESTION_PARAGRAPHS_BATCH = list(tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size))
QUESTION_SENTENCES_BATCH = list(tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size))
SENTENCE_PARAGRAPH_BATCH = list(tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size))
PARAGRAPH_LINKS_BATCH = list(tf.placeholder(dtype=tf.int32, shape=[None, 2]) for _ in range(batch_size))
SENTENCE_LINKS_BATCH = list(tf.placeholder(dtype=tf.int32, shape=[None, 2]) for _ in range(batch_size))
PARAGRAPH_PROOF_BATCH = list(tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size))
SENTENCE_PROOF_BATCH = list(tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size))
ANSWER_SPANS_BATCH = list(tf.placeholder(dtype=tf.int32, shape=[None, 2]) for _ in range(batch_size))
ANSWER_CLASS_BATCH = list(tf.placeholder(dtype=tf.int32, shape=[]) for _ in range(batch_size))
WEIGHT_DECAY = tf.placeholder(dtype=tf.float32, shape=[])
LEARNING_RATE = tf.placeholder(dtype=tf.float32, shape=[])
EMA_MANAGER = tf.train.ExponentialMovingAverage(exponential_moving_average_decay)

MODEL_UPDATE = build_update(
    ELMO_MODULE, GLOVE_EMBEDDING,
    CONTEXT_SYMBOLS_BATCH, QUESTION_SYMBOLS_BATCH,
    CONTEXT_NUMBERS_BATCH, QUESTION_NUMBERS_BATCH,
    CONTEXT_BERTS_BATCH, QUESTION_BERTS_BATCH,
    CONTEXT_PARAGRAPHS_BATCH, QUESTION_PARAGRAPHS_BATCH,
    CONTEXT_SENTENCES_BATCH, QUESTION_SENTENCES_BATCH,
    SENTENCE_PARAGRAPH_BATCH, PARAGRAPH_LINKS_BATCH, SENTENCE_LINKS_BATCH,
    PARAGRAPH_PROOF_BATCH, SENTENCE_PROOF_BATCH, ANSWER_SPANS_BATCH, ANSWER_CLASS_BATCH,
    WEIGHT_DECAY, LEARNING_RATE, EMA_MANAGER
)

MODEL_PREDICT_BATCH = build_predict(
    ELMO_MODULE, GLOVE_EMBEDDING,
    CONTEXT_SYMBOLS_BATCH, QUESTION_SYMBOLS_BATCH,
    CONTEXT_NUMBERS_BATCH, QUESTION_NUMBERS_BATCH,
    CONTEXT_BERTS_BATCH, QUESTION_BERTS_BATCH,
    CONTEXT_PARAGRAPHS_BATCH, QUESTION_PARAGRAPHS_BATCH,
    CONTEXT_SENTENCES_BATCH, QUESTION_SENTENCES_BATCH,
    SENTENCE_PARAGRAPH_BATCH, PARAGRAPH_LINKS_BATCH, SENTENCE_LINKS_BATCH
)

MODEL_SMOOTH = tf.group(
    *list(
        tf.assign(ref=VARIABLE, value=EMA_MANAGER.average(VARIABLE))
        for VARIABLE in tf.trainable_variables()
    )
)

for index in range(batch_size):
    tf.add_to_collection(name="CONTEXT_SYMBOLS_BATCH", value=CONTEXT_SYMBOLS_BATCH[index])
    tf.add_to_collection(name="CONTEXT_NUMBERS_BATCH", value=CONTEXT_NUMBERS_BATCH[index])
    tf.add_to_collection(name="CONTEXT_BERTS_BATCH", value=CONTEXT_BERTS_BATCH[index])
    tf.add_to_collection(name="CONTEXT_PARAGRAPHS_BATCH", value=CONTEXT_PARAGRAPHS_BATCH[index])
    tf.add_to_collection(name="CONTEXT_SENTENCES_BATCH", value=CONTEXT_SENTENCES_BATCH[index])
    tf.add_to_collection(name="QUESTION_SYMBOLS_BATCH", value=QUESTION_SYMBOLS_BATCH[index])
    tf.add_to_collection(name="QUESTION_NUMBERS_BATCH", value=QUESTION_NUMBERS_BATCH[index])
    tf.add_to_collection(name="QUESTION_BERTS_BATCH", value=QUESTION_BERTS_BATCH[index])
    tf.add_to_collection(name="QUESTION_PARAGRAPHS_BATCH", value=QUESTION_PARAGRAPHS_BATCH[index])
    tf.add_to_collection(name="QUESTION_SENTENCES_BATCH", value=QUESTION_SENTENCES_BATCH[index])
    tf.add_to_collection(name="SENTENCE_PARAGRAPH_BATCH", value=SENTENCE_PARAGRAPH_BATCH[index])
    tf.add_to_collection(name="PARAGRAPH_LINKS_BATCH", value=PARAGRAPH_LINKS_BATCH[index])
    tf.add_to_collection(name="SENTENCE_LINKS_BATCH", value=SENTENCE_LINKS_BATCH[index])
    tf.add_to_collection(name="PARAGRAPH_PROOF_BATCH", value=PARAGRAPH_PROOF_BATCH[index])
    tf.add_to_collection(name="SENTENCE_PROOF_BATCH", value=SENTENCE_PROOF_BATCH[index])
    tf.add_to_collection(name="ANSWER_SPANS_BATCH", value=ANSWER_SPANS_BATCH[index])
    tf.add_to_collection(name="ANSWER_CLASS_BATCH", value=ANSWER_CLASS_BATCH[index])
    tf.add_to_collection(name="MODEL_PREDICT_BATCH", value=MODEL_PREDICT_BATCH[index])

tf.add_to_collection(name="WEIGHT_DECAY", value=WEIGHT_DECAY)
tf.add_to_collection(name="LEARNING_RATE", value=LEARNING_RATE)
tf.add_to_collection(name="MODEL_UPDATE", value=MODEL_UPDATE)
tf.add_to_collection(name="MODEL_SMOOTH", value=MODEL_SMOOTH)
SAVER = tf.train.Saver()
SAVER.export_meta_graph(model_design_path)

with tf.Session(
        config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )
) as SESSION:
    SESSION.run(tf.initializers.global_variables())
    SAVER.save(sess=SESSION, save_path=model_storage_path, write_meta_graph=False, write_state=False)
    dump_file([], model_progress_path, "json")

print("construct: cost {} seconds".format(int((datetime.datetime.now() - begin_time).total_seconds())))
