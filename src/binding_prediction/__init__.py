__version__ = '0.0.0'

import binding_prediction.layers
from binding_prediction.language_model import Elmo, OneHot
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_model(path):
    return os.path.join(_ROOT, 'models', path)


pretrained_language_models = {
    'elmo': (Elmo, get_model('lstm_lm.hdf5')),
    'onehot': (OneHot, None)
}
