from collections import OrderedDict

import numpy as np
import paddle
import torch


def convert_pytorch_checkpoint_to_paddle(
        pytorch_checkpoint_path="pytorch_model.bin",
        paddle_dump_path="model_state.pdparams"):
    pytorch_state_dict = torch.load(
        pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        is_transpose = False
        if k[-7:] == ".weight":
            # embeddings.weight and layer_norm.weight do not transpose
            if ("embeddings" not in k or "relative_pos_embeddings" in k) and "layer_norm" not in k:
                if v.ndim == 2:
                    v = v.transpose(0, 1)
                    is_transpose = True
        oldk = k

        print(f"Converting: {oldk} => {k} | is_transpose {is_transpose}")
        paddle_state_dict[k] = v.data.numpy()

    paddle.save(paddle_state_dict, paddle_dump_path)


if __name__ == "__main__":
    convert_pytorch_checkpoint_to_paddle(
        "pytorch_model.bin",
        "model_state.pdparams")
