from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import load_dataset
from paddle.io import DataLoader, BatchSampler

from tqdm import tqdm

from prophetnet.tokenizer import ProphetNetTokenizer


def read(data_path):
    data_path_src = data_path[0]
    data_path_tgt = data_path[1]
    with open(data_path_src, 'r', encoding='utf-8') as f_d_s:
        src_lines_length = len(f_d_s.readlines())
    with open(data_path_tgt, 'r', encoding='utf-8') as f_d_t:
        tgt_lines_length = len(f_d_t.readlines())
    assert src_lines_length == tgt_lines_length
    with open(data_path_src, 'r', encoding='utf-8') as f_d_s:
        with open(data_path_tgt, 'r', encoding='utf-8') as f_d_t:
            for row_d_s, row_d_t in tqdm(zip(f_d_s, f_d_t), total=src_lines_length):
                yield {'article': row_d_s, 'highlights': row_d_t}


train_dataset = load_dataset(read, data_path=['data/cnndm/uncased_tok_data/train.src',
                                              'data/cnndm/uncased_tok_data/train.tgt'], lazy=False)

dev_dataset = load_dataset(read,
                           data_path=['data/cnndm/uncased_tok_data/dev.src', 'data/cnndm/uncased_tok_data/dev.tgt'],
                           lazy=False)

test_dataset = load_dataset(read,
                            data_path=['data/cnndm/uncased_tok_data/test.src', 'data/cnndm/uncased_tok_data/test.tgt'],
                            lazy=False)

t = ProphetNetTokenizer(vocab_file="../prophetnet.tokenizer")


def convert_example(is_test=False):
    def warpper(example):
        """convert an example into necessary features"""
        tokens = example['article']
        labels = example['highlights']
        src_ids, src_attention_mask_ids = tokens.split("$1$")
        src_ids = [int(i) for i in src_ids.split(" ")]
        src_attention_mask_ids = [int(i) for i in src_attention_mask_ids.split(" ")]

        if not is_test:
            labels, decoder_input_attention_mask_ids = labels.split("$1$")
            labels = [int(i) for i in labels.split(" ")]
            decoder_input_attention_mask_ids = [int(i) for i in decoder_input_attention_mask_ids.split(" ")]
            decoder_input_ids = [labels[-1]] + labels[:-1]

            return src_ids, src_attention_mask_ids, decoder_input_ids, decoder_input_attention_mask_ids, labels

        else:
            return src_ids, src_attention_mask_ids

    return warpper


trunc = convert_example()

train_dataset = train_dataset.map(trunc)

batchify_fn = lambda samples, fn=Tuple(Pad(axis=0, pad_val=t.pad_token_id),  # src_ids
                                       Pad(axis=0, pad_val=0),  # src_pids
                                       Pad(axis=0, pad_val=t.pad_token_id),  # tgt_ids
                                       Pad(axis=0, pad_val=0),  # tgt_pids
                                       Pad(axis=0, pad_val=t.pad_token_id)  # label
                                       ): fn(samples)

train_batch_sampler = BatchSampler(train_dataset, batch_size=2, shuffle=True)

train_data_loader = DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler, collate_fn=batchify_fn)

data = next(iter(train_data_loader))

print(data)
