# -*- coding: utf-8 -*-

from .train import main as softmax_train
from .predict import predict as softmax_predict

__all__ = ['softmax_train', 'softmax_predict']
