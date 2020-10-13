import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
ACCUMULATION = 2
BERT_PATH = "/Users/gsingh/Documents/Personnal/Projects/JigSawMultiLingual_BERT/input/bert_base_multilingual_uncased/"
MODEL_PATH = "trained_bert"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
