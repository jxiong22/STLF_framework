import collections
import typing
import json

import numpy as np
import torch


class TrainConfig(typing.NamedTuple):
    past_T: int
    forecast_T: int
    train_size: int
    batch_size: int
    hidden_size: int
    feature_size: int
    out_feats: int
    lr: float
    logname: str
    subName: str
    epochs: int
    l1: float
    l2: float
    patience: int
    data: str
    numFeature: int

    @classmethod
    def from_dict(cls,dikt):
        past_T = dikt['past_T']
        forecast_T = dikt['forecast_T']
        train_size = dikt['train_size']
        batch_size = dikt['batch_size']
        hidden_size = dikt['hidden_size']
        feature_size = dikt['feature_size']
        out_feats = dikt['out_feats']
        lr = dikt['lr']
        logname = dikt["logname"]
        subName = dikt["subName"]
        epochs = dikt["epochs"]
        l1 = dikt["l1"]
        l2 = dikt["l2"]
        patience = dikt["patience"]
        data = dikt["data"]
        numFeature = dikt["numFeature"]

        return cls(past_T, forecast_T, train_size,batch_size,hidden_size,feature_size,out_feats,lr,logname,subName,epochs,l1,l2,patience,data,numFeature)


class TrainData(typing.NamedTuple):
    feats: np.ndarray
    targs: np.ndarray

class TestData(typing.NamedTuple):
    feats: np.ndarray
    targs: np.ndarray


# ANLF = collections.namedtuple("ANLF", ["encoder", "decoder", "feature", "enc_opt", "dec_opt", "fea_opt"])
ANLF = collections.namedtuple("ANLF", ["encoder", "decoder", "feature", "opt"])
