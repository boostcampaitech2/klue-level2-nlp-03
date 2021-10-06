from load_data import *
from train import *

MODEL_NAME = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir='bert_ckpt')
test = [    0,  1389, 18491,  5102, 13791,  2343,  7992,     2,    37, 17665,
            2,  4840,  2440,    27,  2429,  1187,  2119,  2067,  2225,  2259,
           37,  2309,     8, 16184,    16,  5058,  2079, 10093,  2071,  2200,
            7,    37, 17665,     7,     8,  1389, 18491,  5102, 13791,  2343,
         7992,     8,  1453,    23,  2440,  4123,  2069,  1057,  4007,    16,
        25913,  2200,  6236,  2371,  2062,    18,     2,]

print(tokenizer.decode(test))
print(tokenizer('@@ ##'))