## Short-term load forecasting
This repository contains models for short term load forecasting.

### Attention-based load forecasting Framework with LSTM implementation (PM-LSTM)
The following commend contains load forecasting model training, update model training and test result:
```
python ANLFF_test.py --hidden 128 --epochs 150 --batch 128 --lr 0.001  --final_run --updateCkpName checkpoint_update_ISO --logname ANLFF_ISO --subName ANLFF_ISO --test_size 0.2 --patience 30 --tryNumber 5
```
