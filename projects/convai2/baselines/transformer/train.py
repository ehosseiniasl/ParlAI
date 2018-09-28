# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Train model for ppl metric with pre-selected parameters.
These parameters have some variance in their final perplexity, but they were
used to achieve the pre-trained model.
"""

from parlai.scripts.train_model import setup_args, TrainLoop


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task='convai2:self',
        model='transformer',
        model_file='/tmp/convai2_self_transformer_model',
        dict_lower=True,
        dict_include_valid=False,
        dict_maxexs=-1,
        datatype='train',
        batchsize=64,
        hiddensize=1024,
        embeddingsize=256,
        attention='general',
        numlayers=2,
        rnn_class='lstm',
        learningrate=3,
        dropout=0.1,
        gradient_clip=0.1,
        lookuptable='enc_dec',
        optimizer='sgd',
        embedding_type='glove',
        momentum=0.9,
        bidirectional=False,
        context_length=-1,
        validation_every_n_secs=90,
        validation_metric='ppl',
        validation_metric_mode='min',
        validation_patience=12,
        log_every_n_secs=10,
        dict_tokenizer='split',
        tensorboard_log=True,
        save_every_n_sec=1800,
        max_token_seq_len=1000,
        d_word_vec=256,
        d_model = 256,
        d_inner = 2048,
        n_layers = 2,
        n_head = 2,
        d_k = 64,
        d_v = 64,
        #dropout = 0.1,
        tgt_emb_prj_weight_sharing = True,
        emb_src_tgt_weight_sharing = True,
        label_smoothing=False
        )
    TrainLoop(parser).train()
