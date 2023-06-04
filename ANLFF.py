
import os
import typing
import utils
import argparse
import numpy as np
import matplotlib.pyplot as plt
import joblib
import copy

from numpy import save
from datetime import  datetime

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR

from custom_types import ANLF, TrainConfig

from modules import Encoder, Decoder, FeatureLayer
from sklearn import metrics
from utils import EarlyStopping

from torch.utils.data import DataLoader
from utils import Dataset_ISO, testSampler, Dataset_Utility, Dataset_Update
from sklearn.model_selection import train_test_split


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--D", type=int, default=7, help="Past day step")
    parser.add_argument("--f_T", type=int, default=24, help="forecast time step")
    parser.add_argument("--HpD", type=int, default=24, help="Hour per Day")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--hidden", type=int, default=128, help="hidden size")
    parser.add_argument("--epochs", type=int, default=60, help="training epochs")
    parser.add_argument("--save_plots", action="store_true", help="save plots when true, otherwise show")
    parser.add_argument("--logname", action="store", default='root', help="name for log")
    parser.add_argument("--test_parnn", action="store_true", help="test for parnn")
    parser.add_argument("--subName", action="store", type=str, default='test', help="name of the directory of current run")
    parser.add_argument("--l1", type=float, default=0.01, help="L1 norm weight")
    parser.add_argument("--l2", type=float, default=0, help="variance weight")
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--data', action="store",  type=str, default='ISONE', help='data')
    parser.add_argument("--final_run", action="store_true", help="test for parnn")
    parser.add_argument("--test_size", type=float, default=0.5, help="test size")
    parser.add_argument("--updateCkpName", action="store", type=str, default='checkpoint_update1',
                        help="name of checkpoint update name")
    parser.add_argument('--tryNumber', type=int, default=5, help='try n times')

    return parser.parse_args()


def pa_rnn(config):

    enc_kwargs = {"input_size": config.feature_size+1, "hidden_size": config.hidden_size, "T": config.past_T}

    encoder = Encoder(**enc_kwargs).to(device)

    dec_kwargs = {
                  "hidden_size": config.hidden_size,
                  "T": config.past_T,
                  "f_T": config.forecast_T,
                  "m_features": config.feature_size,
                  "out_feats": config.out_feats}

    decoder = Decoder(**dec_kwargs).to(device)
    feature = FeatureLayer(config.feature_size).to(device)

    optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad] + [p for p in decoder.parameters() if p.requires_grad] + [p for p in feature.parameters() if p.requires_grad],
        lr=config.lr)
    pa_rnn_net = ANLF(encoder, decoder, feature, optimizer)

    return pa_rnn_net


def train(net, train_Dataloader, vali_Dataloader, t_cfg, criterion, modelDir):

    iter_loss = []
    iter_loss_all = []
    epoch_loss = []
    vali_loss = []
    early_stopping = EarlyStopping(logger, patience=t_cfg.patience, verbose=True)

    y_vali = vali_Dataloader.dataset.targs[t_cfg.past_T:]
    y_vali = torch.from_numpy(y_vali).type(torch.FloatTensor)
    scheduler = StepLR(net.opt, step_size=30, gamma=0.1)

    checkpointBest = "checkpoint_best.ckpt"
    path = os.path.join(modelDir, checkpointBest)
    n_iter = 0
    for e_i in range(t_cfg.epochs):

        logger.info(f"# of epoches: {e_i}")
        for t_i, (feats, y_history, y_target, target_feats) in enumerate(train_Dataloader):

            loss = train_iteration(net, criterion, feats, y_history, y_target, target_feats,t_cfg.l1,t_cfg.l2)
            iter_loss.append(loss)
            n_iter += 1

        epoch_losses = np.average(iter_loss)
        iter_loss = np.array(iter_loss).reshape(-1)
        iter_loss_all.append(iter_loss)
        epoch_loss.append(epoch_losses)
        iter_loss = []
        y_vali_pred = predict(net, t_cfg, vali_Dataloader)
        val_loss = criterion(y_vali_pred, y_vali)
        logger.info(f"Epoch {e_i:d}, train loss: {epoch_losses:3.3f}, val loss: {val_loss:3.3f}.")
        vali_loss.append(val_loss)
        logger.info(f"parnn: ")
        y_predict = train_Dataloader.dataset.inverse_transform(y_vali_pred)
        mapeScore = evaluate_score(vali_Dataloader.dataset.targs_ori[t_cfg.past_T:], y_predict)
        if e_i % 10 == 0:
            checkpointName = "checkpoint_" + str(e_i) + '.ckpt'
            torch.save({
                'encoder_state_dict': net.encoder.state_dict(),
                'decoder_state_dict': net.decoder.state_dict(),
                'feature_state_dict': net.feature.state_dict(),
                'optimizer_state_dict': net.opt.state_dict(),
            }, os.path.join(modelDir, checkpointName))


        early_stopping(mapeScore, net, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step()

    checkpoint = torch.load(os.path.join(modelDir, checkpointBest), map_location=device)
    net.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    net.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    net.feature.load_state_dict(checkpoint['feature_state_dict'])
    net.opt.load_state_dict(checkpoint['optimizer_state_dict'])
    net.encoder.eval()
    net.decoder.eval()
    net.feature.eval()
    iter_loss_all = np.array(iter_loss_all).reshape(-1)
    train_loss = np.array(epoch_loss).reshape(-1)
    vali_loss = np.array(vali_loss).reshape(-1)
    save(os.path.join(modelDir, 'iter_loss.npy'), iter_loss_all)
    save(os.path.join(modelDir, 'train_loss.npy'), train_loss)
    save(os.path.join(modelDir, 'vali_loss.npy'), vali_loss)




def train_iteration(t_net, loss_func, X, y_history, y_target, target_feats, l1, l2):

    t_net.opt.zero_grad()

    X = X.type(torch.FloatTensor).to(device)
    y_history = y_history.type(torch.FloatTensor).to(device)
    y_target = y_target.type(torch.FloatTensor).to(device)
    target_feats = target_feats.type(torch.FloatTensor).to(device)

    attn_weights, weighted_history_input, weighted_future_input = t_net.feature(X, target_feats)
    input_encoded, hidden,cell = t_net.encoder(weighted_history_input,y_history)
    y_pred = t_net.decoder(input_encoded, weighted_history_input, y_history, weighted_future_input,hidden,cell)

    l1_norm = torch.norm(attn_weights, 1)
    var = torch.var(attn_weights) # optional
    loss = loss_func(y_pred, y_target) + l1 * l1_norm - l2 * var
    loss.backward()
    t_net.opt.step()

    return loss.item()

def train_update(net, train_Dataloader, vali_Dataloader, t_cfg, criterion, modelDir,
                 train_Dataloader_ori, ckpName):
    iter_loss = []
    iter_loss_all = []
    epoch_loss = []
    vali_loss = []
    early_stopping = EarlyStopping(logger, patience=10, verbose=True)

    checkpointBest = ckpName + '.ckpt'
    path = os.path.join(modelDir, checkpointBest)
    n_iter = 0
    for e_i in range(20):

        logger.info(f"# of epoches: {e_i}")
        for t_i, (feats, err, y_target, target_feats, err_target, targs_pred, y_target_ori, y_history) in enumerate(
                train_Dataloader):
            loss = train_iteration_update(net, criterion, feats, err, y_target, target_feats, t_cfg.l1, t_cfg.l2,
                                          targs_pred, err_target)
            iter_loss.append(loss)
            n_iter += 1

        epoch_losses = np.average(iter_loss)
        iter_loss = np.array(iter_loss).reshape(-1)
        iter_loss_all.append(iter_loss)
        epoch_loss.append(epoch_losses)
        iter_loss = []
        y_vali_pred, y_vali, y_vali_ori, _ = predict_update(net, t_cfg, vali_Dataloader)

        val_loss = criterion(y_vali, y_vali_pred)
        logger.info(f"Epoch {e_i:d}, train loss: {epoch_losses:3.3f}, val loss: {val_loss:3.3f}.")
        vali_loss.append(val_loss)
        logger.info(f"parnn: ")
        y_predict = train_Dataloader_ori.dataset.inverse_transform(y_vali_pred)
        mapeScore = evaluate_score(y_vali_ori.numpy(), y_predict)

        early_stopping(mapeScore, net, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    checkpoint = torch.load(os.path.join(modelDir, checkpointBest), map_location=device)
    net.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    net.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    net.feature.load_state_dict(checkpoint['feature_state_dict'])
    net.encoder.eval()
    net.decoder.eval()
    net.feature.eval()
    iter_loss_all = np.array(iter_loss_all).reshape(-1)
    train_loss = np.array(epoch_loss).reshape(-1)
    vali_loss = np.array(vali_loss).reshape(-1)
    save(os.path.join(modelDir, 'iter_loss.npy'), iter_loss_all)
    save(os.path.join(modelDir, 'train_loss.npy'), train_loss)
    save(os.path.join(modelDir, 'vali_loss.npy'), vali_loss)
    return net, early_stopping.val_loss_min


def train_iteration_update(t_net, loss_func, X, err, y_target, target_feats, l1, l2, targs_pred, err_target):

    t_net.opt.zero_grad()

    X = X.type(torch.FloatTensor).to(device)
    err = err.type(torch.FloatTensor).to(device)
    target_feats = target_feats.type(torch.FloatTensor).to(device)
    err_target = err_target.type(torch.FloatTensor).to(device)
    attn_weights, weighted_history_input, weighted_future_input = t_net.feature(X, target_feats)
    input_encoded, hidden, cell = t_net.encoder(weighted_history_input, err)
    out = t_net.decoder(input_encoded, weighted_history_input, err, weighted_future_input, hidden, cell)

    loss = loss_func(out, err_target)
    loss.backward()

    t_net.opt.step()

    return loss.item()

def predict(t_net: ANLF, t_cfg: TrainConfig, vali_Dataloader):

    y_pred = []

    with torch.no_grad():
        for _, (X, y_history, y_target, target_feats) in enumerate(vali_Dataloader):
            X = X.type(torch.FloatTensor).to(device)
            y_history = y_history.type(torch.FloatTensor).to(device)
            target_feats = target_feats.type(torch.FloatTensor).to(device)
            attn_weights, weighted_history_input, weighted_future_input = t_net.feature(X, target_feats)
            input_encoded, hidden, cell = t_net.encoder(weighted_history_input, y_history)
            output = t_net.decoder(input_encoded, weighted_history_input, y_history, weighted_future_input, hidden, cell).view(-1)
            y_pred.append(output)
        out = torch.stack(y_pred, 0).reshape(-1, 1)
    return out.cpu()


def predict_check(t_net: ANLF, t_cfg: TrainConfig, vali_Dataloader):

    y_pred = []
    y_true_ori = []

    with torch.no_grad():
        for _, (X, err, y_target, target_feats, err_target, targs_pred, y_target_ori, y_history) in enumerate(
                vali_Dataloader):
            X = X.type(torch.FloatTensor).to(device)
            y_history = y_history.type(torch.FloatTensor).to(device)
            target_feats = target_feats.type(torch.FloatTensor).to(device)
            y_target_ori = y_target_ori.type(torch.FloatTensor)
            attn_weights, weighted_history_input, weighted_future_input = t_net.feature(X, target_feats)
            input_encoded, hidden, cell = t_net.encoder(weighted_history_input, y_history)
            output = t_net.decoder(input_encoded, weighted_history_input, y_history, weighted_future_input, hidden,
                                   cell).view(-1)
            y_pred.append(output)
            y_true_ori.append(y_target_ori)

        out = torch.stack(y_pred, 0).reshape(-1, 1)
        y_true_ori = torch.stack(y_true_ori, 0).reshape(-1, 1)

    return out.cpu(), y_true_ori


def predict_update(t_net: ANLF, t_cfg: TrainConfig, vali_Dataloader):

    y_pred = []
    y_true = []
    y_true_ori = []
    y_pred_old = []

    with torch.no_grad():
        for _, (X, err, y_target, target_feats, err_target, targs_pred, y_target_ori, _) in enumerate(vali_Dataloader):
            X = X.type(torch.FloatTensor).to(device)
            err = err.type(torch.FloatTensor).to(device)
            y_target = y_target.type(torch.FloatTensor)
            y_target_ori = y_target_ori.type(torch.FloatTensor)
            targs_pred = targs_pred.type(torch.FloatTensor).to(device)
            target_feats = target_feats.type(torch.FloatTensor).to(device)
            attn_weights, weighted_history_input, weighted_future_input = t_net.feature(X, target_feats)
            input_encoded, hidden, cell = t_net.encoder(weighted_history_input, err)
            output = t_net.decoder(input_encoded, weighted_history_input, err, weighted_future_input, hidden,
                                   cell).view(-1)
            y_pred.append(output + targs_pred.view(-1))
            y_true.append(y_target)
            y_true_ori.append(y_target_ori)
            y_pred_old.append(targs_pred)

        out = torch.stack(y_pred, 0).reshape(-1, 1)
        y_true = torch.stack(y_true, 0).reshape(-1, 1)
        y_true_ori = torch.stack(y_true_ori, 0).reshape(-1, 1)
        y_pred_old = torch.stack(y_pred_old, 0).reshape(-1, 1)

    return out.cpu(), y_true, y_true_ori, y_pred_old

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mape(y_true, y_pred):

    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

def evaluate_score(y_real, y_predict):
    # MAE
    logger.info(f"MAE: {metrics.mean_absolute_error(y_real, y_predict)}")

    # RMS
    logger.info(f"RMS: {np.sqrt(metrics.mean_squared_error(y_real, y_predict))}")

    # MAPE
    mapeScore = mape(y_real, y_predict)
    logger.info(f"MAPE: {mapeScore}")

    # NRMSE
    logger.info(f"NRMSE: {np.sqrt(metrics.mean_squared_error(y_real, y_predict))/np.mean(y_real)*100}")
    return mapeScore

def save_or_show_plot(file_nm: str, save: bool):
    if save:
        plt.savefig(os.path.join(os.path.dirname(__file__), "plots", file_nm))
    else:
        plt.show()



def run_parnn(config, train_Dataloader, vali_Dataloader, modelDir):


    model = pa_rnn(config)
    criterion = nn.MSELoss()

    joblib.dump(config, os.path.join(modelDir, "config.pkl"))


    logger.info("Training start")
    train(model, train_Dataloader, vali_Dataloader, config, criterion, modelDir)
    logger.info("Training end")

    logger.info("Validation start")
    final_y_pred = predict(model, config, vali_Dataloader)

    y_predict = vali_Dataloader.dataset.inverse_transform(final_y_pred)
    y_real = vali_Dataloader.dataset.targs_ori[config.past_T:]
    _ = evaluate_score(y_real, y_predict)

    logger.info("Validation end")

    log_name = '_' + config.logname + datetime.now().strftime('log_%Y_%m_%d')
    y_parnn_name = "y_parnn_iso_" + "D_" + str(int(config.past_T/24)) + "_batch_" + str(config.batch_size) + "_ed_" + str(config.hidden_size) + "_epochs_" + str(config.epochs) + log_name + '.npy'

    save(os.path.join(modelDir, y_parnn_name), y_predict)

    return model

def update_model(config, train_Dataloader, vali_Dataloader, modelDir, model_fix, train_Dataloader_ori,
                 updateCkpName, tryNumber):
    criterion = nn.MSELoss()
    best_score = None
    checkpointBest = updateCkpName + '.ckpt'
    path = os.path.join(modelDir, checkpointBest)
    logger.info("Update Training start")
    for i in range(tryNumber):
        model_update = copy.deepcopy(model_fix)
        model_update.encoder.train()
        model_update.decoder.train()
        for param in model_update.feature.parameters():
            param.requires_grad = False
        logger.info(f"Try {i}")
        uCkpName = updateCkpName + str(i)
        net, score = train_update(model_update, train_Dataloader, vali_Dataloader, config, criterion, modelDir,
                                  train_Dataloader_ori, uCkpName)
        logger.info(f"------------MAPE: {score}")
        if best_score is None:
            best_score = score
            torch.save({
                'encoder_state_dict': net.encoder.state_dict(),
                'decoder_state_dict': net.decoder.state_dict(),
                'feature_state_dict': net.feature.state_dict(),
                'optimizer_state_dict': net.opt.state_dict(),
            }, path)
        elif score < best_score:
            logger.info(f'----------MAPE decreased ({best_score:.6f} --> {score:.6f}).  Saving model ...')
            best_score = score
            torch.save({
                'encoder_state_dict': net.encoder.state_dict(),
                'decoder_state_dict': net.decoder.state_dict(),
                'feature_state_dict': net.feature.state_dict(),
                'optimizer_state_dict': net.opt.state_dict(),
            }, path)

    logger.info("Training end")

    checkpoint = torch.load(path, map_location='cpu')

    model_update.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model_update.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model_update.feature.load_state_dict(checkpoint['feature_state_dict'])

    model_update.encoder.eval()
    model_update.decoder.eval()
    model_update.feature.eval()
    return model_update


if __name__ == '__main__':

    args = get_args()
    utils.mkdir("log/" + args.subName)
    logger = utils.setup_log(args.subName, args.logname)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using computation device: {device}")
    logger.info(args)
    save_plots = args.save_plots

    data_dict = {
        'ISONE': Dataset_ISO,
        'Utility': Dataset_Utility
    }
    Data = data_dict[args.data]
    trainData = Data(
        logger=logger,
        flag='train',
        past_T=args.D * args.HpD,
        future_T=args.f_T,
        data_path=args.data,
        debug=args.debug
    )

    valiData = Data(
        logger=logger,
        flag='val',
        past_T=args.D * args.HpD,
        future_T=args.f_T,
        data_path=args.data,
        scalerX=trainData.scalerX,
        scalerY=trainData.scalerY,
        debug=args.debug

    )

    testData = Data(
        logger=logger,
        flag='test',
        past_T=args.D * args.HpD,
        future_T=args.f_T,
        data_path=args.data,
        scalerX=trainData.scalerX,
        scalerY=trainData.scalerY,
        debug=args.debug
    )


    train_Dataloader = DataLoader(
        trainData,
        batch_size=args.batch,
        shuffle=True)

    samplerVali = testSampler(valiData.__len__())
    samplerTest = testSampler(testData.__len__())

    vali_Dataloader = DataLoader(
        valiData,
        batch_size=1,
        sampler=samplerVali
    )

    test_Dataloader = DataLoader(
        testData,
        batch_size=1,
        sampler=samplerTest
    )

    train_Dataloader_for_predict = DataLoader(
        trainData,
        batch_size=1,
        sampler=testSampler(trainData.__len__()))

    config_dict = {
        "past_T": args.D * args.HpD,
        "forecast_T": args.f_T,
        "train_size": trainData.__len__(),
        "batch_size": args.batch,
        "hidden_size": args.hidden,
        "feature_size": trainData.featureSize,
        "out_feats": 1,
        "lr": args.lr,
        "logname": args.logname,
        "subName": args.subName,
        "epochs": args.epochs,
        "l1": args.l1,
        "l2": args.l2,
        "patience": args.patience,
        "data": args.data,
        "numFeature": trainData.numFeatures
    }


    config = TrainConfig.from_dict(config_dict)
    modelDir = utils.mkdirectory(config, config.subName, True)
    logger.info(f"Training size: {config.train_size:d}.")

    ## --------------train------------- ##
    model_fix = run_parnn(config, train_Dataloader, vali_Dataloader, modelDir)

    ## test without error correction
    logger.info("test without error correction")
    test_pred_old = predict(model_fix, config, test_Dataloader)

    test_pred_old_ori = train_Dataloader.dataset.inverse_transform(test_pred_old)
    test_real_ori = test_Dataloader.dataset.targs_ori[config.past_T:]
    _ = evaluate_score(test_real_ori, test_pred_old_ori)

    logger.info("test end")

    ## -----------error correction----------- ##
    vali_pred_old = predict(model_fix, config, vali_Dataloader)
    vali_pred_old_ori = train_Dataloader.dataset.inverse_transform(vali_pred_old)
    vali_real_ori = vali_Dataloader.dataset.targs_ori[config.past_T:]

    trainData_update = Dataset_Update(
        logger=logger,
        feats=valiData.feats[config.past_T:],
        targs=valiData.targs[config.past_T:],
        targs_ori=valiData.targs_ori[config.past_T:],
        targs_pred=vali_pred_old.numpy(),
        past_T=config.past_T,
        future_T=config.forecast_T
    )

    testData_update = Dataset_Update(
        logger=logger,
        feats=testData.feats[config.past_T:],
        targs=testData.targs[config.past_T:],
        targs_ori=testData.targs_ori[config.past_T:],
        targs_pred=test_pred_old.numpy(),
        past_T=config.past_T,
        future_T=config.forecast_T
    )

    index = np.arange(trainData_update.__len__())
    trainIndex, valiIndex = train_test_split(index, test_size=args.test_size, shuffle=False)
    valiIndex = valiIndex[valiIndex % 24 == 0]
    train_subsampler = torch.utils.data.SubsetRandomSampler(trainIndex)
    vali_subsampler = torch.utils.data.SubsetRandomSampler(valiIndex)

    train_Dataloader_update = DataLoader(
        trainData_update,
        batch_size=config.batch_size,
        sampler=train_subsampler)

    vali_Dataloader_update = DataLoader(
        trainData_update,
        batch_size=1,
        sampler=vali_subsampler)

    sampler = testSampler(testData.__len__() - config.past_T)
    sampler_vali = testSampler(trainData_update.__len__())

    test_Dataloader_update = DataLoader(
        testData_update,
        batch_size=1,
        sampler=sampler
    )


    ### new valiset result:
    new_vali_pred_old, new_vali_ori = predict_check(model_fix, config, vali_Dataloader_update)
    logger.info(f"new_vali_old: ")
    new_vali_pred_old_ori = trainData.inverse_transform(new_vali_pred_old)
    evaluate_score(new_vali_ori.numpy(), new_vali_pred_old_ori)

    model_update = update_model(config, train_Dataloader_update, vali_Dataloader_update, modelDir, model_fix,
                 train_Dataloader, args.updateCkpName, args.tryNumber)

    new_vali_pred_updated, _, new_vali_ori_2, _ = predict_update(model_update, config, vali_Dataloader_update)

    new_vali_pred_updated_ori = trainData.inverse_transform(new_vali_pred_updated)
    logger.info(f"Old score vali: ")
    evaluate_score(new_vali_ori.numpy(), new_vali_pred_old_ori)

    logger.info(f"New score vali: ")
    evaluate_score(new_vali_ori_2.numpy(), new_vali_pred_updated_ori)
    logger.info("update end")

    if args.final_run:

        logger.info(f"parnn_original: ")

        evaluate_score(test_real_ori[config.past_T:], test_pred_old_ori[config.past_T:])
        # evaluate_score(test_real_ori, test_pred_old_ori)

        ############# adjust

        logger.info("update start")
        test_pred_updated, _, _, _ = predict_update(model_update, config, test_Dataloader_update)
        logger.info("update end")

        test_pred_updated_ori = train_Dataloader.dataset.inverse_transform(test_pred_updated)

        logger.info(f"parnn_update: ")
        evaluate_score(test_real_ori[config.past_T:], test_pred_updated_ori)
        logger.info(f"data: {config.data}.")
        if config.data == "ISONE":
            name1 = "y_update_ISO_" + args.updateCkpName + ".npy"
            name2 = "y_ori_ISO_" + args.updateCkpName + ".npy"

        else:
            name1 = "y_update_Utility_" + args.updateCkpName + ".npy"
            name2 = "y_ori_Utility_" + args.updateCkpName + ".npy"

        save(os.path.join(modelDir, name1), test_pred_updated_ori)
        save(os.path.join(modelDir, name2), test_pred_old_ori)
    logger.info("-----End-----")
