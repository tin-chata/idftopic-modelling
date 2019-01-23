"""
Created on 2019-01-15
@author: duytinvo
"""
import os
import torch
from model import Autoencoder_model
from other_utils import SaveloadHP
from data_utils import Txtfile, seqPAD, PADt, Data2tensor, Embeddings

use_cuda=False
model_args = "extracted_data/tpmd.args"

margs = SaveloadHP.load(model_args)
margs.use_cuda = use_cuda

model_filename = os.path.join(margs.model_dir, margs.model_file)
print("Load Model from file: %s" % model_filename)
topic_encoder = Autoencoder_model(margs)
topic_encoder.model.load_state_dict(torch.load(model_filename))
topic_encoder.model.to(topic_encoder.device)

word_emb, enc_emb, dec_emb = topic_encoder.model.get_embs()
id2topic = {}
for i in range(enc_emb.shape[0]):
    id2topic[i] = "topic_%d" % i
Embeddings.save_embs(id2topic, dec_emb.transpose(), os.path.join(margs.model_dir, margs.dtopic_emb_file))
Embeddings.save_embs(id2topic, enc_emb, os.path.join(margs.model_dir, margs.etopic_emb_file))
Embeddings.save_embs(margs.vocab.i2w, word_emb, os.path.join(margs.model_dir, margs.tuned_word_emb_file))

if __name__ == '__main__':

    rv = "back in 2005 2007 this place was my favorite thai place ever id go here alllll the time i never had any " \
         "complaints once they started to get more known and got busy their service started to suck and their portion " \
         "sizes got cut in half i have a huge problem with paying more for way less food the last time i went there i "

    label_prob, label_pred = topic_encoder.predict(rv)


