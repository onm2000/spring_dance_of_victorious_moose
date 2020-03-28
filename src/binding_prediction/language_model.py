import torch
import tensorflow as tf
from binding_prediction.protein import ProteinSequence
from binding_prediction.utils import onehot


class LanguageModel(object):

    def __init__(self, path):
        super(LanguageModel, self).__init__()
        self.path = path

    def extract(self, x):
        pass


class Elmo(LanguageModel):
    # requires a GPU in order to test
    def __init__(self, path, trainable=False):
        super(Elmo, self).__init__(path)
        m = tf.keras.models.load_model(path)
        layer = 'LSTM2'
        self.model = tf.keras.models.Model(inputs=[m.input],
                                           outputs=[m.get_layer(layer).output],
                                           trainable=trainable)

    def extract(self, x):
        prot = ProteinSequence(x)
<<<<<<< HEAD
        embed = self.model.predict(prot.onehot).squeeze()
        return torch.Tensor(embed)
=======
        return self.model.predict(prot.onehot).squeeze()


class OneHot(LanguageModel):
    def __init__(self, path):
        super(OneHot, self).__init__()
        self.tla_codes = ["A", "R", "N", "D", "B", "C", "E", "Q", "Z", "G",
                          "H", "I", "L", "K", "M", "F", "P", "S", "T", "W",
                          "Y", "V"]
        self.num_words = len(self.tla_codes)

    def extract(self, x):
        embedding = []
        for x_i in x:
            emb_i = [onehot(self.tla_codes.index(w_i), self.num_words) for w_i in x]
            embedding.append(torch.Tensor(emb_i).float())
        return embedding
>>>>>>> 94fc8f35fdca323e49f4e72faa97a4534fb47e24
