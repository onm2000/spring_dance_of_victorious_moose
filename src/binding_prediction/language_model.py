import torch
import warnings
from binding_prediction.protein import ProteinSequence
from binding_prediction.utils import onehot
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf


class LanguageModel(object):

    def __init__(self, path, device='cuda'):
        super(LanguageModel, self).__init__()
        self.path = path
        self.device = device

    def __call__(self, x):
        pass


class Elmo(LanguageModel):
    # requires a GPU in order to test
    def __init__(self, path, trainable=False, device='cuda'):
        super(Elmo, self).__init__(path, device)
        m = tf.keras.models.load_model(path)
        layer = 'LSTM2'
        self.model = tf.keras.models.Model(inputs=[m.input],
                                           outputs=[m.get_layer(layer).output],
                                           trainable=trainable)

    def __call__(self, x):
        prot = ProteinSequence(x)
        embed = self.model.predict(prot.onehot).squeeze()


class OneHot(LanguageModel):
    def __init__(self, path, device='cuda'):
        super(OneHot, self).__init__(path, device)
        self.tla_codes = ["A", "R", "N", "D", "B", "C", "E", "Q", "Z", "G",
                          "H", "I", "L", "K", "M", "F", "P", "S", "T", "W",
                          "Y", "V"]
        self.num_words = len(self.tla_codes)

    def __call__(self, x):
        emb_i = [onehot(self.tla_codes.index(w_i), self.num_words) for w_i in x]
        return torch.Tensor(emb_i).to(self.device)
