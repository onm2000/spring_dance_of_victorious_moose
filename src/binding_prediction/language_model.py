import torch
import tensorflow as tf
from binding_prediction.protein import ProteinSequence


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
        return self.model.predict(prot.onehot).squeeze()
