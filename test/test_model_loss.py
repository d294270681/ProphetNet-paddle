from prophetnet.modeling import ProphetNetForConditionalGeneration, ProphetNetModel
from transformers import ProphetNetTokenizer

from transformers import ProphetNetForConditionalGeneration as torchProphetNetForConditionalGeneration

import paddle
import torch
import numpy as np


def compare(out_torch, out_paddle):
    assert out_torch.shape == out_paddle.shape
    abs_dif = np.abs(out_torch - out_paddle)
    mean_dif = np.mean(abs_dif)
    print("mean_dif:{}".format(mean_dif))


t = ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")

input_str = "the us state department said wednesday it had received no formal word from bolivia that it was expelling the us ambassador there but said the charges made against him are `` baseless ."
target_str = "us rejects charges against its ambassador in bolivia"

input_ids = t(input_str).input_ids
labels = t(target_str).input_ids

p_input_ids = paddle.to_tensor(input_ids, dtype=paddle.int32).reshape([1, -1])
p_labels = paddle.to_tensor(labels, dtype=paddle.int32).reshape([1, -1])

t_input_ids = torch.tensor(input_ids, dtype=torch.int32).reshape(1, -1)
t_labels = torch.tensor(labels, dtype=torch.int64).reshape(1, -1)

p = ProphetNetModel(vocab_size=30522)
pg = ProphetNetForConditionalGeneration(p)
pg.eval()

tp = torchProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased")

pg.load_dict(paddle.load("../model_state.pdparams"))

p_loss = pg(input_ids=p_input_ids, labels=p_labels)[0]
t_loss = tp(input_ids=t_input_ids, labels=t_labels).loss

print("paddle loss:", p_loss.item())
print("torch loss:", t_loss.item())
