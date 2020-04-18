# -*- coding: utf-8 -*-

from .train import main as conv2d_train
from .predict import predict as conv2d_predict

__all__ = ['conv2d_train', 'conv2d_predict']
