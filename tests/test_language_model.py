import torch
import pytest
from binding_prediction.language_model import Elmo
from binding_prediction import language_models
import numpy.testing as npt


class TestElmo(object):
    # Requires a GPU to test
    # to test on rusty
    # module load cuda/10.0.130_410.48
    # module load cudnn/v7.6.2-cuda-10.0
    def test_elmo(self):
        cls, path = language_models['elmo']
        model = cls(path)
        s = 'ACTATACTCTCTATTPPPP'
        res = model.extract(s)
        npt.assert_allclose(res.shape, [len(s), 512])
