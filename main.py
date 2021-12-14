
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
from utils import Dataset_ISO, testSampler, Dataset_Utility


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


def train(net: ANLF, train_Dataloader, vali_Dataloader, t_cfg: TrainConfig, criterion, modelDir):

    # iter_per_epoch = train_Dataloader.__len__()
    # iter_losses = np.zeros(t_cfg.epochs * iter_per_epoch) # iteration number
    # epoch_losses = np.zeros(t_cfg.epochs)
    iter_loss = []
    iter_loss_all = []
    epoch_loss = []
    vali_loss = []
    early_stopping = EarlyStopping(logger, patience=t_cfg.patience, verbose=True)

    y_vali = vali_Dataloader.dataset.targs[t_cfg.past_T:]
    y_vali = torch.from_numpy(y_vali).type(torch.FloatTensor)

    scheduler_enc = StepLR(net.enc_opt, step_size=30, gamma=0.1)
    scheduler_dec = StepLR(net.dec_opt, step_size=30, gamma=0.1)
    scheduler_fea = StepLR(net.fea_opt, step_size=30, gamma=0.1)

    checkpointBest = "checkpoint_best.ckpt"
    path = os.path.join(modelDir, checkpointBest)
    n_iter = 0 # counting iteration number
    for e_i in range(t_cfg.epochs):

        logger.info(f"# of epoches: {e_i}")
        # for t_i in range(0, end_point, t_cfg.batch_size):
        for t_i, (feats, y_history, y_target, target_feats) in enumerate(train_Dataloader):

            loss = train_iteration(net, criterion, feats, y_history, y_target, target_feats,t_cfg.l1,t_cfg.l2)
            # iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = loss
            iter_loss.append(loss)
            n_iter += 1

        # epoch_losses[e_i] = np.mean(iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])
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
                'encoder_optimizer_state_dict': net.enc_opt.state_dict(),
                'decoder_optimizer_state_dict': net.dec_opt.state_dict(),
                'feature_optimizer_state_dict': net.fea_opt.state_dict(),
            }, os.path.join(modelDir, checkpointName))


        early_stopping(mapeScore, net, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler_enc.step()
        scheduler_dec.step()
        scheduler_fea.step()

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




def train_iteration(t_net: ANLF, loss_func: typing.Callable, X, y_history, y_target, target_feats, l1, l2):

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
    t_net.fea_opt.zero_grad()

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

    t_net.fea_opt.step()
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

    return y_predict




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
    # runiso=True
    # Dataloder
    # data, scaler, data_ori, out_feats, test_data, test_data_ori = utils.data_loader(logger, args.debug, runiso)
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

    train_Dataloader = DataLoader(
        trainData,
        batch_size=args.batch,
        shuffle=True)

    sampler = testSampler(valiData.__len__())

    vali_Dataloader = DataLoader(
        valiData,
        batch_size=1,
        sampler=sampler
    )

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

    run_parnn(config, train_Dataloader, vali_Dataloader, modelDir)

    # except Exception as e:
    #     logger.error(f"Errors:{e}")

    logger.info("-----End-----")
