__version__ = '0.0.0'

import binding_prediction.layers
from binding_prediction.language_model import Elmo


_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)

language_models = {
    'elmo': (Elmo, get_data('models/lstm_lm.hdf5'))
}
