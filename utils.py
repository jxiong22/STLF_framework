import logging
import logging.handlers
import os
from datetime import datetime, date
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler


def setup_log(subName='', tag='root'):
    # create logger
    logger = logging.getLogger(tag)
    # logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    # logger.handlers = []
    logger.addHandler(ch)

    # file log
    log_name = tag + datetime.now().strftime('log_%Y_%m_%d.log')

    log_path = os.path.join('log', subName, log_name)
    fh = logging.handlers.RotatingFileHandler(
        log_path, mode='a', maxBytes=100 * 1024 * 1024, backupCount=1, encoding='utf-8'
    )

    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


def save_or_show_plot(file_nm: str, save: bool):
    if save:
        plt.savefig(os.path.join(os.path.dirname(__file__), "plots", file_nm))
    else:
        plt.show()


def split_data(x):
    return x.iloc[:-365*24], x.iloc[-365*24:]


def get_season(date_time):
    # dummy leap year to include leap days(year-02-29) in our range
    leap_year = 2000
    seasons = [('winter', (date(leap_year, 1, 1), date(leap_year, 3, 20))),
               ('spring', (date(leap_year, 3, 21), date(leap_year, 6, 20))),
               ('summer', (date(leap_year, 6, 21), date(leap_year, 9, 22))),
               ('autumn', (date(leap_year, 9, 23), date(leap_year, 12, 20))),
               ('winter', (date(leap_year, 12, 21), date(leap_year, 12, 31)))]

    if isinstance(date_time, datetime):
        date_time = date_time.date()
    # we don't really care about the actual year so replace it with our dummy leap_year
    date_time = date_time.replace(year=leap_year)
    # return season our date falls in.
    return next(season for season, (start, end) in seasons
                if start <= date_time <= end)


def create_datetype_column(data_set):
    # cloning the input dataset.
    local = data_set.copy()

    # add season column
    local['Season'] = pd.Series(local.index).apply(get_season).values

    # add holiday column
    cal = calendar()
    holidays = cal.holidays(start='2008-01-01', end='2020-07-31')

    dateTime = pd.to_datetime(data_set.index.date)
    local['Holiday'] = dateTime.isin(holidays).astype(int)

    # add day of week column
    local['day_of_week'] = pd.Series(local.index).dt.day_name().values
    # local['day_of_week'] = pd.Series(local.index).dt.dayofweek.values

    # add month of year column
    local['month_of_year'] = pd.Series(local.index).dt.month_name().values
    # local['month_of_year'] = pd.Series(local.index).dt.month.values

    # add hour of day column
    local['hour_of_day'] = pd.Series(local.index).dt.hour.values

    # one-hot encoding
    local = pd.get_dummies(local)
    local = pd.get_dummies(local, columns=['hour_of_day'])


    return local


def mkdir(dirName):
    if not os.path.exists(dirName):
        if os.name == 'nt':
            os.system('mkdir {}'.format(dirName.replace('/', '\\')))
        else:
            os.system('mkdir -p {}'.format(dirName))


def mkdirectory(config, subName, saveModel):
    log_name = '_' + config.logname + datetime.now().strftime('log_%Y_%m_%d')

    dirName_data = "data/" + subName
    dirName_log = "log/" + subName
    mkdir(dirName_data)
    mkdir(dirName_log)

    if saveModel is True:
        model_name = "/model_iso_" + "D_" + str(int(config.past_T / 24)) + "_batch_" + str(
            config.batch_size) + "_ed_" + str(config.hidden_size) + "_epochs_" + str(
            config.epochs) + log_name

        dirName_model = "history_model/" + subName + model_name
        mkdir(dirName_model)
        return dirName_model


class EarlyStopping:
    def __init__(self, logger, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.logger = logger

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, net, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save({
            'encoder_state_dict': net.encoder.state_dict(),
            'decoder_state_dict': net.decoder.state_dict(),
            'feature_state_dict': net.feature.state_dict(),
            'optimizer_state_dict': net.opt.state_dict(),
        }, path)

        self.val_loss_min = val_loss



def fillZero(df, type=0):
    zeroIdx = df[df.isin([0]).any(axis=1)].index
    if type is 0:
        nextIdx = zeroIdx + pd.offsets.Hour(1)
        prevIdx = zeroIdx + pd.offsets.Hour(-1)
        df.loc[zeroIdx] = (df.loc[nextIdx].values + df.loc[prevIdx].values) / 2
    else:
        nextIdx = zeroIdx + pd.DateOffset(1)
        prevIdx = zeroIdx + pd.DateOffset(-1)
        df.loc[zeroIdx] = (df.loc[nextIdx].values + df.loc[prevIdx].values) / 2
    return df


data_dict = dict()

class Dataset_ISO(Dataset):
    def __init__(self, logger, flag='train', past_T=24*7, future_T=24,
                 data_path='ISONE', target=("load",), scalerX=None, scalerY=None, debug=False):
        self.logger = logger
        self.past_T = past_T
        self.future_T = future_T
        # initial
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.target = target
        self.data_path = data_path

        if self.set_type != 0 and (scalerX is None or scalerY is None):
            self.logger.error("Vali or Test without scaler")
            os.exit(-1)
        self.scalerX = scalerX
        self.scalerY = scalerY
        self.debug = debug
        self.__read_data__()


    def __read_data__(self):
        global data_dict
        if self.data_path not in data_dict:
            self.scalerX = StandardScaler()
            self.scalerY = StandardScaler()
            df_raw = pd.read_csv(os.path.join("data",
                                              self.data_path+".csv"))
            date_rng = pd.date_range(start='1/1/2015 00:00:00', end='12/31/2019 23:00:00', freq='H')
            df_raw = df_raw.iloc[-365 * 24 * 5 - 25:-1, -df_raw.shape[1] + 3:].set_index(date_rng)
            df_raw = create_datetype_column(df_raw)
            self.logger.info(f"features: {df_raw.columns.values}.")
            # df_raw.to_csv('isodata.csv',index=False)
        else:
            df_raw = data_dict[self.data_path]
        data_len = df_raw.shape[0]


        if self.debug:
            border1s = [0, 10 * 24, 20*24]
            border2s = [10 * 24, 20 * 24, 30*24]
        else:
            border1s = [0, 365*24*3+24, data_len-365*24]
            border2s = [365*24*3+192+24, 365*24*4+24, data_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        proc_dat = df_raw[border1:border2].values

        mask = np.ones(proc_dat.shape[1], dtype=bool)  # 82 true
        dat_cols = list(df_raw.columns)  # get all column name
        for col_name in self.target:
            mask[dat_cols.index(col_name)] = False
            feats = proc_dat[:, mask]
            targs = proc_dat[:, ~mask]

        self.numFeatures = df_raw.columns.get_loc("load")
        if self.data_path not in data_dict:
            self.scalerX.fit(feats[:, :self.numFeatures])
            self.scalerY.fit(targs)
            data_dict[self.data_path] = df_raw
        feats_scaled = self.scalerX.transform(feats[:, :self.numFeatures])
        targs_scaled = self.scalerY.transform(targs)
        feats_scaled_combine = np.concatenate([feats_scaled, feats[:, self.numFeatures:]], axis=1)
        self.feats = feats_scaled_combine
        self.targs = targs_scaled
        self.targs_ori = targs
        self.featureSize = self.feats.shape[1]



    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.past_T
        r_begin = s_end
        r_end = r_begin + self.future_T

        feats = self.feats[s_begin:s_end]
        target_feats = self.feats[r_begin:r_end]
        y_history = self.targs[s_begin:s_end]
        y_target = self.targs[r_begin:r_end]

        return feats, y_history, y_target, target_feats

    def __len__(self):
        return len(self.feats) - self.past_T - self.future_T + 1

    def inverse_transform(self, data):
        return self.scalerY.inverse_transform(data)

class Dataset_Utility(Dataset):
    def __init__(self, logger, flag='train', past_T=24*7, future_T=24,
                 data_path='Utility', target=("load",), scalerX=None, scalerY=None, debug=False):
        self.logger = logger
        self.past_T = past_T
        self.future_T = future_T
        # initial
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.target = target
        self.data_path = data_path

        if self.set_type != 0 and (scalerX is None or scalerY is None):
            self.logger.error("Vali or Test without scaler")
            os.exit(-1)
        self.scalerX = scalerX
        self.scalerY = scalerY
        self.debug = debug
        self.__read_data__()


    def __read_data__(self):
        global data_dict
        if self.data_path not in data_dict:
            self.scalerX = StandardScaler()
            self.scalerY = StandardScaler()
            df_temp = pd.read_fwf(os.path.join("data", "input.txt"), header=None).iloc[:-1,:]
            df_load = pd.read_fwf(os.path.join("data", "output.txt"), header=None).iloc[:-1,:]
            df_temp[0]=pd.to_datetime(df_temp[0], format='%m/%d/%y')
            df_load[0]=pd.to_datetime(df_load[0], format='%m/%d/%y')
            df_temp=df_temp.set_index(0)
            df_load = df_load.set_index(0)

            df_temp = df_temp.loc['1987-1-1':'1991-12-31'].values.reshape(-1)
            df_load = df_load.loc['1987-1-1':'1991-12-31'].values.reshape(-1)

            date_rng = pd.date_range(start='1/1/1987 00:00:00', end='12/31/1991 23:00:00', freq='H')

            df_raw = pd.DataFrame({'T':df_temp,'load':df_load}, index=date_rng)
            df_raw = fillZero(df_raw)

            df_raw = create_datetype_column(df_raw)
            self.logger.info(f"features: {df_raw.columns.values}.")
        else:
            df_raw = data_dict[self.data_path]
        data_len = df_raw.shape[0]


        if self.debug:
            border1s = [0, 10 * 24, 20*24]
            border2s = [10 * 24, 20 * 24, 30*24]
        else:
            border1s = [0, 365*24*3+24, data_len-365*24]
            border2s = [365*24*3+192+24, 365*24*4+24, data_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        proc_dat = df_raw[border1:border2].values
        # proc_dat = dat

        mask = np.ones(proc_dat.shape[1], dtype=bool)  # 82 true
        dat_cols = list(df_raw.columns)  # get all column name
        for col_name in self.target:
            mask[dat_cols.index(col_name)] = False
            feats = proc_dat[:, mask]
            targs = proc_dat[:, ~mask]

        self.numFeatures = df_raw.columns.get_loc("load")
        if self.data_path not in data_dict:
            self.scalerX.fit(feats[:, :self.numFeatures])
            self.scalerY.fit(targs)
            data_dict[self.data_path] = df_raw
        feats_scaled = self.scalerX.transform(feats[:, :self.numFeatures])
        targs_scaled = self.scalerY.transform(targs)
        feats_scaled_combine = np.concatenate([feats_scaled, feats[:, self.numFeatures:]], axis=1)
        self.feats = feats_scaled_combine
        self.targs = targs_scaled
        self.targs_ori = targs
        self.featureSize = self.feats.shape[1]


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.past_T
        r_begin = s_end
        r_end = r_begin + self.future_T

        feats = self.feats[s_begin:s_end]
        target_feats = self.feats[r_begin:r_end]
        y_history = self.targs[s_begin:s_end]
        y_target = self.targs[r_begin:r_end]

        return feats, y_history, y_target, target_feats

    def __len__(self):
        return len(self.feats) - self.past_T - self.future_T + 1

    def inverse_transform(self, data):
        return self.scalerY.inverse_transform(data)



class testSampler(Sampler):

    def __init__(self, length):
        self.length = length


    def __iter__(self):
        return iter(range(0,self.length,24))

    def __len__(self) -> int:
        return len(range(0,self.length,24))

class Dataset_Update(Dataset):
    def __init__(self, logger, feats, targs, targs_pred, targs_ori, past_T, future_T):
        self.logger = logger
        self.feats = feats
        self.targs = targs
        self.targs_pred = targs_pred
        self.targs_ori = targs_ori
        self.err = self.targs - self.targs_pred
        self.past_T = past_T
        self.future_T = future_T

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.past_T
        r_begin = s_end
        r_end = r_begin + self.future_T

        feats = self.feats[s_begin:s_end]
        target_feats = self.feats[r_begin:r_end]
        err = self.err[s_begin:s_end]
        y_history = self.targs[s_begin:s_end]
        y_target = self.targs[r_begin:r_end]
        err_target = self.err[r_begin:r_end]
        targs_pred = self.targs_pred[r_begin:r_end]
        y_target_ori = self.targs_ori[r_begin:r_end]

        return feats, err, y_target, target_feats, err_target, targs_pred, y_target_ori, y_history

    def __len__(self):
        return len(self.feats) - self.past_T - self.future_T + 1


