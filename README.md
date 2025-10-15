# Facial Expression Recognition (FER) using the KDEF dataset

## Dataset Link: https://www.kaggle.com/datasets/tom99763/testtt

## Kaggle Notebook: https://www.kaggle.com/code/shasan07/fer-pytorch

The images were transformed and normalized using torch, then we fed them to the **pretrained DenseNet121 model for fine-Tuning**. (Several other popular pre-trained models are also listed along with it).

A **learning rate of 1e-4** was used for the **AdamW** optimizer, along with **batch size of 8**, and the training was conducted for **50 epochs**.
**The best validation accuracy achieved by our model is 94.49%**.

![Abstract: ](Abstract.png)

Training Summary:

    Epoch [1/50]  Train Loss: 1.1639 | Train Acc: 57.76%  || Val Loss: 0.5370 | Val Acc: 82.76%
    Saved new best model with val_acc: 82.76%
    Epoch [2/50]  Train Loss: 0.4200 | Train Acc: 85.83%  || Val Loss: 0.3926 | Val Acc: 86.73%
    Saved new best model with val_acc: 86.73%
    Epoch [3/50]  Train Loss: 0.2386 | Train Acc: 92.34%  || Val Loss: 0.3453 | Val Acc: 89.59%
    Saved new best model with val_acc: 89.59%
    Epoch [4/50]  Train Loss: 0.1351 | Train Acc: 95.89%  || Val Loss: 0.3586 | Val Acc: 89.90%
    Saved new best model with val_acc: 89.90%
    Epoch [5/50]  Train Loss: 0.1163 | Train Acc: 96.38%  || Val Loss: 0.2585 | Val Acc: 91.73%
    Saved new best model with val_acc: 91.73%
    Epoch [6/50]  Train Loss: 0.0898 | Train Acc: 97.14%  || Val Loss: 0.4773 | Val Acc: 92.55%
    Saved new best model with val_acc: 92.55%
    Epoch [7/50]  Train Loss: 0.0843 | Train Acc: 97.14%  || Val Loss: 0.2090 | Val Acc: 93.67%
    Saved new best model with val_acc: 93.67%
    Epoch [8/50]  Train Loss: 0.0682 | Train Acc: 97.88%  || Val Loss: 0.3692 | Val Acc: 89.39%
    Epoch [9/50]  Train Loss: 0.0634 | Train Acc: 98.03%  || Val Loss: 0.3282 | Val Acc: 91.73%
    Epoch [10/50]  Train Loss: 0.0483 | Train Acc: 98.67%  || Val Loss: 0.2505 | Val Acc: 92.35%
    Epoch [11/50]  Train Loss: 0.0515 | Train Acc: 98.37%  || Val Loss: 0.3511 | Val Acc: 91.94%
    Epoch [12/50]  Train Loss: 0.0399 | Train Acc: 98.88%  || Val Loss: 0.4205 | Val Acc: 89.59%
    Epoch [13/50]  Train Loss: 0.0360 | Train Acc: 99.00%  || Val Loss: 0.2864 | Val Acc: 92.55%
    Epoch [14/50]  Train Loss: 0.0369 | Train Acc: 98.88%  || Val Loss: 0.3243 | Val Acc: 92.24%
    Epoch [15/50]  Train Loss: 0.0299 | Train Acc: 99.16%  || Val Loss: 0.4572 | Val Acc: 90.51%
    Epoch [16/50]  Train Loss: 0.0427 | Train Acc: 98.88%  || Val Loss: 0.3331 | Val Acc: 90.92%
    Epoch [17/50]  Train Loss: 0.0337 | Train Acc: 99.13%  || Val Loss: 0.3805 | Val Acc: 91.22%
    Epoch [18/50]  Train Loss: 0.0354 | Train Acc: 99.03%  || Val Loss: 0.4215 | Val Acc: 89.80%
    Epoch [19/50]  Train Loss: 0.0449 | Train Acc: 98.70%  || Val Loss: 0.2631 | Val Acc: 92.65%
    Epoch [20/50]  Train Loss: 0.0225 | Train Acc: 99.26%  || Val Loss: 0.2593 | Val Acc: 93.88%
    Saved new best model with val_acc: 93.88%
    Epoch [21/50]  Train Loss: 0.0298 | Train Acc: 99.08%  || Val Loss: 0.3229 | Val Acc: 92.65%
    Epoch [22/50]  Train Loss: 0.0189 | Train Acc: 99.34%  || Val Loss: 0.3232 | Val Acc: 92.96%
    Epoch [23/50]  Train Loss: 0.0320 | Train Acc: 99.23%  || Val Loss: 0.4447 | Val Acc: 92.76%
    Epoch [24/50]  Train Loss: 0.0253 | Train Acc: 99.26%  || Val Loss: 0.3163 | Val Acc: 92.35%
    Epoch [25/50]  Train Loss: 0.0300 | Train Acc: 99.23%  || Val Loss: 0.2847 | Val Acc: 93.57%
    Epoch [26/50]  Train Loss: 0.0167 | Train Acc: 99.46%  || Val Loss: 0.3133 | Val Acc: 92.96%
    Epoch [27/50]  Train Loss: 0.0194 | Train Acc: 99.57%  || Val Loss: 0.2613 | Val Acc: 93.78%
    Epoch [28/50]  Train Loss: 0.0299 | Train Acc: 99.13%  || Val Loss: 0.3292 | Val Acc: 93.06%
    Epoch [29/50]  Train Loss: 0.0210 | Train Acc: 99.34%  || Val Loss: 0.3586 | Val Acc: 93.06%
    Epoch [30/50]  Train Loss: 0.0211 | Train Acc: 99.36%  || Val Loss: 0.3061 | Val Acc: 93.78%
    Epoch [31/50]  Train Loss: 0.0211 | Train Acc: 99.41%  || Val Loss: 0.3750 | Val Acc: 92.76%
    Epoch [32/50]  Train Loss: 0.0169 | Train Acc: 99.46%  || Val Loss: 0.2669 | Val Acc: 93.98%
    Saved new best model with val_acc: 93.98%
    Epoch [33/50]  Train Loss: 0.0238 | Train Acc: 99.11%  || Val Loss: 0.3924 | Val Acc: 92.14%
    Epoch [34/50]  Train Loss: 0.0265 | Train Acc: 99.11%  || Val Loss: 0.3600 | Val Acc: 92.45%
    Epoch [35/50]  Train Loss: 0.0269 | Train Acc: 99.16%  || Val Loss: 0.2951 | Val Acc: 93.88%
    Epoch [36/50]  Train Loss: 0.0089 | Train Acc: 99.77%  || Val Loss: 0.2502 | Val Acc: 93.47%
    Epoch [37/50]  Train Loss: 0.0158 | Train Acc: 99.59%  || Val Loss: 0.3338 | Val Acc: 92.86%
    Epoch [38/50]  Train Loss: 0.0248 | Train Acc: 99.46%  || Val Loss: 0.3596 | Val Acc: 92.96%
    Epoch [39/50]  Train Loss: 0.0290 | Train Acc: 99.23%  || Val Loss: 0.3590 | Val Acc: 92.24%
    Epoch [40/50]  Train Loss: 0.0287 | Train Acc: 99.16%  || Val Loss: 0.2972 | Val Acc: 93.78%
    Epoch [41/50]  Train Loss: 0.0138 | Train Acc: 99.59%  || Val Loss: 0.3068 | Val Acc: 92.76%
    Epoch [42/50]  Train Loss: 0.0257 | Train Acc: 99.39%  || Val Loss: 0.2569 | Val Acc: 93.57%
    Epoch [43/50]  Train Loss: 0.0149 | Train Acc: 99.67%  || Val Loss: 0.2555 | Val Acc: 94.18%
    Saved new best model with val_acc: 94.18%
    Epoch [44/50]  Train Loss: 0.0145 | Train Acc: 99.49%  || Val Loss: 0.3501 | Val Acc: 92.24%
    Epoch [45/50]  Train Loss: 0.0093 | Train Acc: 99.74%  || Val Loss: 0.3772 | Val Acc: 93.06%
    Epoch [46/50]  Train Loss: 0.0210 | Train Acc: 99.49%  || Val Loss: 0.3676 | Val Acc: 92.76%
    Epoch [47/50]  Train Loss: 0.0131 | Train Acc: 99.74%  || Val Loss: 0.2568 | Val Acc: 94.49%
    Saved new best model with val_acc: 94.49%
    Epoch [48/50]  Train Loss: 0.0081 | Train Acc: 99.72%  || Val Loss: 0.3309 | Val Acc: 93.88%
    Epoch [49/50]  Train Loss: 0.0169 | Train Acc: 99.57%  || Val Loss: 0.2729 | Val Acc: 92.96%
    Epoch [50/50]  Train Loss: 0.0118 | Train Acc: 99.67%  || Val Loss: 0.3195 | Val Acc: 93.57%


Obtained Results:

    Classification Report:
              precision    recall  f1-score   support

       Anger       0.98      0.91      0.95       140
     Disgust       0.90      0.97      0.93       140
        Fear       0.93      0.89      0.91       140
       Happy       0.99      0.99      0.99       140
     Neutral       0.96      0.98      0.97       140
         Sad       0.93      0.92      0.92       140
    Surprise       0.93      0.96      0.94       140

    accuracy                           0.94       980
   macro avg       0.95      0.94      0.94       980
weighted avg       0.95      0.94      0.94       980


    Confusion Matrix:
    [[128   6   2   0   1   3   0]
     [  1 136   0   1   0   2   0]
     [  1   0 124   1   3   3   8]
     [  0   0   1 138   0   0   1]
     [  0   1   0   0 137   2   0]
     [  0   7   2   0   1 129   1]
     [  0   1   5   0   0   0 134]]


![Confusion Matrix: ](Conf_Mat.png)

![Accuracy Curve: ](Acc.png)

![Loss_Curve): ](Loss.png)
