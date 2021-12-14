
import os
import typing
import utils
import argparse
import numpy as np
import matplotlib.pyplot as plt
import joblib
from numpy import save
from datetime import date, datetime

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR

from custom_types import ANLF, TrainData, TrainConfig

from modules import Encoder, Decoder, FeatureLayer
from sklearn import metrics
from utils import EarlyStopping

from torch.utils.data import DataLoader
from utils import Dataset_ISO, testSampler, Dataset_Utility, Dataset_Update
from torch.utils.data.sampler import BatchSampler
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
    parser.add_argument("--dirName", action="store", type=str, default='root', help="name of the directory of the saved model")
    parser.add_argument("--ei", type=int, default='4', help="load checkpoint_ei's model")
    parser.add_argument("--subName", action="store", type=str, default='test', help="name of the directory of current run")
    parser.add_argument("--l1", type=float, default=0.01, help="L1 norm weight")
    parser.add_argument("--l2", type=float, default=0, help="variance weight")
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--data', action="store",  type=str, default='ISONE', help='data')
    parser.add_argument("--final_run", action="store_true", help="test for parnn")
    parser.add_argument("--updateErr", action="store_true", help="test for parnn")
    parser.add_argument("--test_size", type=float, default=0.5, help="test size")
    parser.add_argument("--updateCkpName", action="store", type=str, default='checkpoint_update1', help="name of checkpoint update name")


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
    feature = FeatureLayer(config.feature_size, config.numFeature+1).to(device)

    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=config.lr)
    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad],
        lr=config.lr)
    feature_optimizer = optim.Adam(
        params=[p for p in feature.parameters() if p.requires_grad],
        lr=config.lr)
    # encoder_optimizer = optim.SGD(encoder.parameters(), lr=config.lr, momentum=0.9)
    # decoder_optimizer = optim.SGD(decoder.parameters(), lr=config.lr, momentum=0.9)
    # feature_optimizer = optim.SGD(feature.parameters(), lr=config.lr, momentum=0.9)
    pa_rnn_net = ANLF(encoder, decoder, feature, encoder_optimizer, decoder_optimizer, feature_optimizer)

    return pa_rnn_net


def train_update(net: ANLF, train_Dataloader, vali_Dataloader, t_cfg: TrainConfig, criterion, modelDir, updateErr, train_Dataloader_ori, ckpName):



    iter_loss = []
    iter_loss_all = []
    epoch_loss = []
    vali_loss = []
    early_stopping = EarlyStopping(logger, patience=t_cfg.patience, verbose=True)

    checkpointBest = ckpName
    path = os.path.join(modelDir, checkpointBest)
    n_iter = 0 # counting iteration number
    # for e_i in range(1):
    for e_i in range(t_cfg.epochs):

        logger.info(f"# of epoches: {e_i}")
        # for t_i in range(0, end_point, t_cfg.batch_size):
        for t_i, (feats, err, y_target, target_feats, err_target, targs_pred, y_target_ori, y_history) in enumerate(train_Dataloader):

            loss = train_iteration_update(net, criterion, feats, err, y_target, target_feats,t_cfg.l1,t_cfg.l2, targs_pred, err_target, updateErr)
            # iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = loss
            iter_loss.append(loss)
            n_iter += 1

        # epoch_losses[e_i] = np.mean(iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])
        epoch_losses = np.average(iter_loss)
        iter_loss = np.array(iter_loss).reshape(-1)
        iter_loss_all.append(iter_loss)
        epoch_loss.append(epoch_losses)
        iter_loss = []
        y_vali_pred,  y_vali, y_vali_ori, _ = predict_update(net, t_cfg, vali_Dataloader)

        if e_i==0:
            y_vali_pred_checkOld, y_vali_ori_checkOld = predict_check(net, t_cfg, vali_Dataloader)
            logger.info(f"parnn_old: ")
            y_predict_checkOld = train_Dataloader_ori.dataset.inverse_transform(y_vali_pred_checkOld)
            evaluate_score(y_vali_ori_checkOld.numpy(), y_predict_checkOld)

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
    net.enc_opt.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    net.dec_opt.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
    net.fea_opt.load_state_dict(checkpoint['feature_optimizer_state_dict'])
    net.encoder.eval()
    net.decoder.eval()
    net.feature.eval()
    iter_loss_all = np.array(iter_loss_all).reshape(-1)
    train_loss = np.array(epoch_loss).reshape(-1)
    vali_loss = np.array(vali_loss).reshape(-1)
    save(os.path.join(modelDir, 'iter_loss.npy'), iter_loss_all)
    save(os.path.join(modelDir, 'train_loss.npy'), train_loss)
    save(os.path.join(modelDir, 'vali_loss.npy'), vali_loss)
    return net


def train_iteration_update(t_net, loss_func, X, err, y_target, target_feats, l1, l2, targs_pred, err_target, updateErr):

    '''
    training process (forward and backwark) for each iteration
    :param t_net: pa_rnn_net(encoder, decoder, encoder_optimizer, decoder_optimizer)
    :param loss_func: nn.MSELoss() defined in config
    :param X: input feature
    :param y_history: y_history
    :param y_target: true forecast value
    :return: loss value
    '''
    t_net.enc_opt.zero_grad()
    t_net.dec_opt.zero_grad()
    # t_net.fea_opt.zero_grad()

    X = X.type(torch.FloatTensor).to(device)
    err = err.type(torch.FloatTensor).to(device)
    y_target = y_target.type(torch.FloatTensor).to(device)
    target_feats = target_feats.type(torch.FloatTensor).to(device)
    err_target = err_target.type(torch.FloatTensor).to(device)
    targs_pred = targs_pred.type(torch.FloatTensor).to(device)
    attn_weights, weighted_history_input, weighted_future_input = t_net.feature(X, target_feats)
    input_encoded, hidden,cell = t_net.encoder(weighted_history_input,err)
    out = t_net.decoder(input_encoded, weighted_history_input, err, weighted_future_input,hidden,cell)

    if updateErr:
        y_pred = targs_pred + out
    else:
        y_pred = out

    loss = loss_func(y_pred, y_target)
    # loss = loss_func(out,err_target)
    loss.backward()

    t_net.enc_opt.step()
    t_net.dec_opt.step()

    return loss.item()


def predict(t_net: ANLF, t_cfg: TrainConfig, vali_Dataloader):
    '''

    :param t_net: pa_rnn_net(encoder, decoder, encoder_optimizer, decoder_optimizer)
    :param t_dat: data include train and evaluation and test.
    :param t_cfg: config file for train
    :param on_train: when on_train, predict will evaluate the result using train sample, otherwise, use test sample
    :return: y_pred:
    '''
    y_pred = []  # (test_size)

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
    '''

    :param t_net: pa_rnn_net(encoder, decoder, encoder_optimizer, decoder_optimizer)
    :param t_dat: data include train and evaluation and test.
    :param t_cfg: config file for train
    :param on_train: when on_train, predict will evaluate the result using train sample, otherwise, use test sample
    :return: y_pred:
    '''
    y_pred = []  # (test_size)
    y_true_ori = []

    with torch.no_grad():
        for _, (X, err, y_target, target_feats, err_target, targs_pred, y_target_ori, y_history) in enumerate(vali_Dataloader):
            X = X.type(torch.FloatTensor).to(device)
            y_history = y_history.type(torch.FloatTensor).to(device)
            target_feats = target_feats.type(torch.FloatTensor).to(device)
            y_target_ori = y_target_ori.type(torch.FloatTensor)
            attn_weights, weighted_history_input, weighted_future_input = t_net.feature(X, target_feats)
            input_encoded, hidden, cell = t_net.encoder(weighted_history_input, y_history)
            output = t_net.decoder(input_encoded, weighted_history_input, y_history, weighted_future_input, hidden, cell).view(-1)
            y_pred.append(output)
            y_true_ori.append(y_target_ori)

        out = torch.stack(y_pred, 0).reshape(-1, 1)
        y_true_ori = torch.stack(y_true_ori, 0).reshape(-1, 1)

    return out.cpu(), y_true_ori

def predict_update(t_net: ANLF, t_cfg: TrainConfig, vali_Dataloader):
    '''

    :param t_net: pa_rnn_net(encoder, decoder, encoder_optimizer, decoder_optimizer)
    :param t_dat: data include train and evaluation and test.
    :param t_cfg: config file for train
    :param on_train: when on_train, predict will evaluate the result using train sample, otherwise, use test sample
    :return: y_pred:
    '''
    y_pred = []  # (test_size)
    y_true = []
    y_true_ori = []
    y_pred_old  = []

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
            output = t_net.decoder(input_encoded, weighted_history_input, err, weighted_future_input, hidden, cell).view(-1)
            y_pred.append(output+targs_pred.view(-1))
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
    # print('MAE', metrics.mean_absolute_error(y_real, y_predict))

    # RMS
    logger.info(f"RMS: {np.sqrt(metrics.mean_squared_error(y_real, y_predict))}")
    # print('RMS', np.sqrt(metrics.mean_squared_error(y_real, y_predict)))

    # MAPE
    mapeScore = mape(y_real, y_predict)
    logger.info(f"MAPE: {mapeScore}")
    # print('MAPE', mape(y_real, y_predict))

    # NRMSE
    logger.info(f"NRMSE: {np.sqrt(metrics.mean_squared_error(y_real, y_predict))/np.mean(y_real)*100}")
    # print('NRMSE', np.sqrt(metrics.mean_squared_error(y_real, y_predict))/np.mean(y_real)*100)
    return mapeScore

def save_or_show_plot(file_nm: str, save: bool):
    if save:
        plt.savefig(os.path.join(os.path.dirname(__file__), "plots", file_nm))
    else:
        plt.show()



def update_model(config, train_Dataloader, vali_Dataloader, modelDir, model_fix, updateErr, train_Dataloader_ori, updateCkpName):

    model = pa_rnn(config)
    model.encoder.train()
    model.decoder.train()
    model.encoder.load_state_dict(model_fix.encoder.state_dict())
    model.decoder.load_state_dict(model_fix.decoder.state_dict())
    model.feature.load_state_dict(model_fix.feature.state_dict())

    criterion = nn.MSELoss()

    logger.info("Update Training start")
    model_update = train_update(model, train_Dataloader, vali_Dataloader, config, criterion, modelDir, updateErr, train_Dataloader_ori, updateCkpName)
    logger.info("Training end")


    return model_update



if __name__ == '__main__':
    # main()
    # try:
    args = get_args()
    utils.mkdir("log/" + args.subName)
    logger = utils.setup_log(args.subName, args.logname)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using computation device: {device}")
    logger.info(args)
    save_plots = args.save_plots
    updateErr = args.updateErr

    config = joblib.load(os.path.join(args.dirName, "config.pkl"))

    data_dict = {
        'ISONE': Dataset_ISO,
        'Utility': Dataset_Utility
    }
    Data = data_dict[config.data]
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


    logger.info(f"test size: {testData.__len__():d}.")

    model_fix = pa_rnn(config)
    checkpointName = "checkpoint_best.ckpt"

    checkpoint = torch.load(os.path.join(args.dirName, checkpointName), map_location='cpu')
    model_fix.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model_fix.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model_fix.feature.load_state_dict(checkpoint['feature_state_dict'])
    model_fix.enc_opt.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    model_fix.dec_opt.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
    model_fix.fea_opt.load_state_dict(checkpoint['feature_optimizer_state_dict'])

    model_fix.encoder.eval()
    model_fix.decoder.eval()
    model_fix.feature.eval()

    logger.info("test start")
    test_pred_old = predict(model_fix, config, test_Dataloader)

    test_pred_old_ori = train_Dataloader.dataset.inverse_transform(test_pred_old)
    test_real_ori = test_Dataloader.dataset.targs_ori[config.past_T:]
    _ = evaluate_score(test_real_ori, test_pred_old_ori)

    # add_zero = np.zeros((config.past_T, 1))
    # y_pred_add_zero = np.concatenate((add_zero, final_y_pred), 1)
    logger.info("test end")

    logger.info("update start")

    train_pred_old = predict(model_fix, config, train_Dataloader_for_predict)
    train_pred_old_ori = train_Dataloader.dataset.inverse_transform(train_pred_old)
    train_real_ori = train_Dataloader.dataset.targs_ori[config.past_T:]

    vali_pred_old = predict(model_fix, config, vali_Dataloader)
    vali_pred_old_ori = train_Dataloader.dataset.inverse_transform(vali_pred_old)
    vali_real_ori = vali_Dataloader.dataset.targs_ori[config.past_T:]
    # trainData_update = Dataset_Update(
    #     logger=logger,
    #     feats=valiData.feats[config.past_T:],
    #     targs=valiData.targs[config.past_T:],
    #     targs_ori=valiData.targs_ori[config.past_T:],
    #     targs_pred=vali_pred_old.numpy(),
    #     past_T=config.past_T,
    #     future_T=config.forecast_T
    # )
    #
    #
    #
    # testData_update = Dataset_Update(
    #     logger=logger,
    #     feats=testData.feats[config.past_T:],
    #     targs=testData.targs[config.past_T:],
    #     targs_ori=testData.targs_ori[config.past_T:],
    #     targs_pred=test_pred_old.numpy(),
    #     past_T=config.past_T,
    #     future_T=config.forecast_T
    # )
    #
    # index = np.arange(trainData_update.__len__())
    # trainIndex, valiIndex = train_test_split(index, test_size=args.test_size)
    # train_subsampler = torch.utils.data.SubsetRandomSampler(trainIndex)
    # vali_subsampler = torch.utils.data.SubsetRandomSampler(valiIndex)
    #
    # train_Dataloader_update = DataLoader(
    #     trainData_update,
    #     batch_size=args.batch,
    #     sampler=train_subsampler)
    #
    # vali_Dataloader_update = DataLoader(
    #     trainData_update,
    #     batch_size=1,
    #     sampler=vali_subsampler)
    #
    # sampler = testSampler(testData.__len__()-config.past_T)
    # sampler_vali = testSampler(trainData_update.__len__())
    #
    #
    # test_Dataloader_update = DataLoader(
    #     testData_update,
    #     batch_size=1,
    #     sampler=sampler
    # )
    #
    # vali_Dataloader_for_predict = DataLoader(
    #     trainData_update,
    #     batch_size=1,
    #     sampler=sampler_vali
    # )


    trainData_update = Dataset_Update(
        logger=logger,
        feats=trainData.feats[config.past_T:],
        targs=trainData.targs[config.past_T:],
        targs_ori=trainData.targs_ori[config.past_T:],
        targs_pred=train_pred_old.numpy(),
        past_T=config.past_T,
        future_T=config.forecast_T
    )

    valiData_update = Dataset_Update(
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

    train_Dataloader_update = DataLoader(
        trainData_update,
        batch_size=args.batch,
        shuffle=True)


    vali_Dataloader_update = DataLoader(
        valiData_update,
        batch_size=1,
        sampler=testSampler(valiData_update.__len__())
    )

    sampler_vali = testSampler(trainData_update.__len__())


    test_Dataloader_update = DataLoader(
        testData_update,
        batch_size=1,
        sampler=testSampler(testData_update.__len__())
    )

    # vali_Dataloader_for_predict = DataLoader(
    #     trainData_update,
    #     batch_size=1,
    #     sampler=sampler_vali
    # )


    # model_fix.feature.train()
    model_update = update_model(config, train_Dataloader_update, vali_Dataloader_update, args.dirName, model_fix, updateErr, train_Dataloader, args.updateCkpName)

    # model_update = pa_rnn(config)
    # checkpointName = "checkpoint_best_update.ckpt"
    #
    # checkpoint = torch.load(os.path.join(args.dirName, checkpointName), map_location='cpu')
    # model_update.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    # model_update.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    # model_update.feature.load_state_dict(checkpoint['feature_state_dict'])
    # model_update.enc_opt.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    # model_update.dec_opt.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
    # model_update.fea_opt.load_state_dict(checkpoint['feature_optimizer_state_dict'])
    #
    # model_update.encoder.eval()
    # model_update.decoder.eval()
    # model_update.feature.eval()

    vali_pred_updated, _, _, _ = predict_update(model_update, config, vali_Dataloader_update)

    vali_pred_updated_ori = train_Dataloader.dataset.inverse_transform(vali_pred_updated)
    logger.info(f"Old score: ")
    evaluate_score(vali_real_ori[config.past_T:], vali_pred_old_ori[config.past_T:])

    logger.info(f"New score: ")
    evaluate_score(vali_real_ori[config.past_T:], vali_pred_updated_ori)
    logger.info("update end")

    if args.final_run:

        logger.info(f"parnn_original: ")

        evaluate_score(test_real_ori[config.past_T:], test_pred_old_ori[config.past_T:])

        ############# adjust

        logger.info("update start")
        test_pred_updated,_,_,_ = predict_update(model_update, config, test_Dataloader_update)
        logger.info("update end")

        test_pred_updated_ori = train_Dataloader.dataset.inverse_transform(test_pred_updated)

        logger.info(f"parnn_update: ")
        evaluate_score(test_real_ori[config.past_T:], test_pred_updated_ori)

        if config.data is "ISONE":
            name1 = "y_update_ISO_" + args.updateCkpName + ".npy"
            name2 = "y_ori_ISO_" + args.updateCkpName + ".npy"

        else:
            name1 = "y_update_Utility_" + args.updateCkpName + ".npy"
            name2 = "y_ori_Utility_" + args.updateCkpName + ".npy"

        save(os.path.join(args.dirName, name1), test_pred_updated_ori)
        save(os.path.join(args.dirName, name2), test_pred_old_ori)

    logger.info("-----End-----")


