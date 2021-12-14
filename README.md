## Short-term load forecasting
This repository contains models for short term load forecasting.

### Attention-based load forecasting (ANLF)
Training:
```
python main.py --run_parnn --encoder 256 --decoder 256 --epochs 20 --batch 128 --lr 0.001 --logname parnn_1_Batch_128_hidden_256_lr_0.001 --subName parnn
```
Testing:
```
python parnn_test.py --dirName history_model/anlf --ei 4 --test_parnn --logname parnn_conference_test
```

### Encoder-decoder LSTM (EDLSTM)
Training:
```
python run_lstm.py --encoder 256 --decoder 256 --epochs 20 --batch 128 --lr 0.001 --logname EDLSTM_Batch_128_hidden_256_lr_0.001 --subName EDLSTM
```
Testing:
```
python lstm_test.py --dirName history_model/EDLSTM --ei 4 --logname EDLSTM_conference_test
```

### Encoder-decoder BiLSTM (EDBiLSTM)
Training:
```
python run_lstm.py --EDBiLSTM --encoder 256 --decoder 256 --epochs 20 --batch 128 --lr 0.001 --logname EDBiLSTM_Batch_128_hidden_256_lr_0.001 --subName EDBiLSTM
```
Testing:
```
python lstm_test.py --EDBiLSTM --dirName history_model/EDBiLSTM --ei 9 --logname EDBiLSTM_conference_test
```

### Gradient Boosting Machine (GBM)
Training:
```
python Comparison.py --run_gbm --feats_with_pnf --log gbm
```
Testing:
```
python Comparison.py --test --run_gbm --feats_with_pnf --log gbm
```

### Support Vector Regressor (SVR)
Training:
```
python Comparison.py --run_svr --feats_with_pnf --log svr
```
Testing:
```
python Comparison.py --test --run_svr --feats_with_pnf --log svr
```

### Random Forest (RF)
Training:
```
python Comparison.py --run_rf --feats_with_pnf --log rf
```
Testing:
```
python Comparison.py --test --run_rf --feats_with_pnf --log rf
```
