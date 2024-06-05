# usage: python ./alt-loss/stochastic_online_linear.py
## needs `archive/Variant I.csv` and `params/model1_tune.pth`
import matplotlib.pyplot as plt
from numpy import  *
import pandas as pd

import pylab as pl
from IPython import display
base = pd.read_csv('archive/Variant I.csv')
# remove 'income', 'customer_age', 'employment_status' columns as they are protected
base = base.drop(columns=['income', 'customer_age', 'employment_status'])
# convert categorical variables in 'payment_type' to integers
base['payment_type'] = base['payment_type'].astype('category')
base['housing_status'] = base['housing_status'].astype('category')
base['source'] = base['source'].astype('category')
base['device_os'] = base['device_os'].astype('category')

cat_columns = base.select_dtypes(['category']).columns
base[cat_columns] = base[cat_columns].apply(lambda x: x.cat.codes)
base = base.to_numpy()

from sklearn.model_selection import train_test_split

X = base[:,1:]
y = base[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# class prior
prior = sum(base[:,0])/len(base)

print('class prior for positive:', prior)
print('class prior for negative:', 1-prior)

def TPR(pred, label):
    TP = sum((pred==1) & (label==1))
    FN = sum((pred==0) & (label==1))
    return TP/(TP+FN)

def FPR(pred, label):
    FP = sum((pred==1) & (label==0))
    TN = sum((pred==0) & (label==0))
    return FP/(FP+TN)

def roc(pred, label):
    fpr, tpr = [], []
    tmax = max(pred)
    tmin = min(pred)
    for t in linspace(tmin, tmax, 100):
        p = pred > t
        tpr.append(TPR(p, label))
        fpr.append(FPR(p, label))
    return fpr, tpr

def AUC(fpr, tpr):
    return sum([(tpr[i]+tpr[i-1])*(fpr[i-1]-fpr[i])/2 for i in range(1, len(fpr))])

import torch as t
import pylab as pl
from IPython import display

# Let us maximize the AUC
import torch # bring out the big gun

torch.manual_seed(0)

class Model1(torch.nn.Module):
    def __init__(self, d):
        super(Model1, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(d))
        self.alpha = torch.nn.Parameter(t.tensor(0.1,))
        self.a = torch.nn.Parameter(t.tensor(0.))
        self.b = torch.nn.Parameter(t.tensor(0.))
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, X, y):
        p1 = y.sum() / len(y)

        alpha = self.alpha
        a = self.a; b = self.b
        w = self.w

        Xw = X @ w # (batch,)
        y = (y > 0).float() # y = 1 if it is positive class, else 0
        loss = (1-p1) * (Xw - a).square() * y
        loss += p1 * (Xw - b).square() * (1.-y)
        loss += 2 * (1.+alpha)*(p1*Xw * (1.-y) -(1.-p1)*Xw*y)
        loss += -p1*(1-p1)*alpha.square()
        loss = loss.mean()
        pred = Xw
        return pred, loss

    def adjust_gradients_for_opt(self):
        """see eqn. 13 from https://papers.nips.cc/paper_files/paper/2016/file/c52f1bd66cc19d05628bd8bf27af3ad6-Paper.pdf"""
        self.alpha.grad = -self.alpha.grad



X_tr = torch.tensor(X_train, dtype = torch.float32)
y_tr = torch.tensor(y_train, dtype = torch.float32)
X_te = torch.tensor(X_test, dtype = torch.float32)
y_te = torch.tensor(y_test, dtype = torch.float32)

dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
weights = torch.ones_like(y_tr)

weights[y_tr == 1] = (len(y_tr) - y_tr.sum()) / y_tr.sum() # == num_p0 * w_0 == num_p1 * w1
weights[y_tr == 0] = 1.

weights = weights / weights.sum()
sampler = torch.utils.data.WeightedRandomSampler(weights, len(dataset), replacement=True)
trainload = torch.utils.data.DataLoader(dataset, batch_size = 10000, sampler=sampler)

ppath = "params/model1_tune.pth"
model1 = Model1(X_tr.shape[1])
model1.load_state_dict(torch.load(ppath))

nepoch = 5
opt = torch.optim.Adam(model1.parameters(), lr=3e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nepoch, eta_min=1e-5)

for epoch in range(nepoch):
    for x, y in trainload:
        model1.train()
        opt.zero_grad()
        # min step, optimizes w, a, b
        pred, loss = model1(x, y)
        (loss).backward()
        model1.adjust_gradients_for_opt()
        opt.step()


    pred_te, test_loss = model1(X_te, y_te)
    test_loss.detach()
    pred_te = pred_te.detach().numpy()

    pred_tr, train_loss = model1(X_tr, y_tr)
    train_loss.detach()
    pred_tr = pred_tr.detach().numpy()


    fpr_te, tpr_te = roc(pred_te, y_test)
    auc_te = AUC(fpr_te, tpr_te)

    fpr_tr, tpr_tr = roc(pred_tr, y_train)
    auc_tr = AUC(fpr_tr, tpr_tr)

    print(f"epoch = {epoch}, tr_loss = {train_loss}, te_loss = {test_loss}, auc_tr = {auc_tr}, auc_te = {auc_te}")
    # pl.plot(fpr, tpr, label = 'epoch %d, AUC: %0.2f' % (epoch, auc))
    # pl.legend(loc = 'lower right')
    # display.display(pl.gcf())
    # display.clear_output(wait=True)