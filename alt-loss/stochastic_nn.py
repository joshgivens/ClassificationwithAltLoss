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

base_pd = base

missing_value = -10 # missing values are replaced by this!

def mean_std_scale(x): return (x - x.mean())/x.std()

df = pd.DataFrame(index=range(len(base)), columns=[])
df['fraud_bool'] = base['fraud_bool'].copy().astype(float)
df['name_email_similarity'] = base['name_email_similarity'].copy()

## scale
prev = base['prev_address_months_count'].copy().astype(float)
prev[prev > 0] = (prev[prev > 0] / prev.max()).copy()
prev[prev < 0] = missing_value
df['prev_address_months_count'] = prev.copy()

prev = base['current_address_months_count'].copy().astype(float)
prev[prev > 0] = (prev[prev > 0] / prev.max()).copy()
prev[prev < 0] = missing_value
df['current_address_months_count'] = prev.copy()

df['days_since_request'] = mean_std_scale(base['days_since_request']).copy()
df['intended_balcon_amount'] = mean_std_scale(base['intended_balcon_amount']).copy()

pt = pd.get_dummies(base, columns=['payment_type'])
for i in range(5): df[f'payment_type_{i}'] = pt[f'payment_type_{i}'].astype(float)


df['zip_count_4w'] = mean_std_scale(base['zip_count_4w'].copy().astype(float))

## not sure these should be mean/std scaled tbh... maybe min/max scaling would be better
## https://www.dropbox.com/scl/fo/vg4b2hyapa9o9ajanbfl3/AL1RUfD1rAb5RBvgFQwc8eI/bank-account-fraud/documents?dl=0&preview=datasheet.pdf&rlkey=2r99po055q5pjbg1934ga0c8i&subfolder_nav_tracking=1
for col in ['velocity_6h', 'velocity_24h', 'velocity_4w',
            'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w',
            'credit_risk_score']:
    df[col] = mean_std_scale(base[col]).copy()

df['email_is_free'] = base['email_is_free'].copy().astype(float)


pt = pd.get_dummies(base, columns=['housing_status'])
for i in range(7): df[f'housing_status_{i}'] = pt[f'housing_status_{i}'].astype(float)

df['phone_home_valid'] = base['phone_home_valid'].copy().astype(float)
df['phone_mobile_valid'] = base['phone_mobile_valid'].copy().astype(float)


prev = base['bank_months_count'].copy().astype(float)
prev[prev > 0] = (prev[prev > 0] / prev.max()).copy()
prev[prev < 0] = missing_value
df['bank_months_count'] = prev.copy()

df['has_other_cards'] = base['has_other_cards'].copy().astype(float)
df['foreign_request'] = base['foreign_request'].copy().astype(float)


## not sure these should be mean/std scaled tbh... maybe min/max scaling would be better
## https://www.dropbox.com/scl/fo/vg4b2hyapa9o9ajanbfl3/AL1RUfD1rAb5RBvgFQwc8eI/bank-account-fraud/documents?dl=0&preview=datasheet.pdf&rlkey=2r99po055q5pjbg1934ga0c8i&subfolder_nav_tracking=1
for col in ['proposed_credit_limit']:
    df[col] = mean_std_scale(base[col]).copy()


df['source'] = base['source'].copy().astype(float)

prev = base['session_length_in_minutes'].copy().astype(float)
prev[prev > 0] = (prev[prev > 0] / prev.max()).copy()
prev[prev < 0] = missing_value
df['session_length_in_minutes'] = prev.copy()


pt = pd.get_dummies(base, columns=['device_os'])
for i in range(5): df[f'device_os_{i}'] = pt[f'device_os_{i}'].astype(float)

df['keep_alive_session'] = base['keep_alive_session'].copy().astype(float)

prev = base['device_distinct_emails_8w'].copy().astype(float)
prev[prev > 0] = (prev[prev > 0] / prev.max()).copy()
prev[prev < 0] = missing_value
df['device_distinct_emails_8w'] = prev.copy()

df['device_fraud_count'] = base['device_fraud_count'].copy().astype(float)

pt = pd.get_dummies(base, columns=['month'])
for i in range(8): df[f'month_{i}'] = pt[f'month_{i}'].astype(float)

# base = base.to_numpy()

base = df.to_numpy()

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

class MLP(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = torch.nn.Linear(d, d*2)
        self.fc2 = torch.nn.Linear(d*2, d*2)
        self.out = torch.nn.Linear(d*2, 2)

        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        # self.lin = torch.nn.Linear(d, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)
        # return self.lin(x)

class Model3(torch.nn.Module):
    def __init__(self, d):
        super(Model3, self).__init__()
        self.h = MLP(d)

    def forward(self, X, returns='logits'):
        logits = self.h(X).squeeze()

        if returns == 'logits': return logits
        elif returns == 'posprob':
            ps = torch.nn.functional.softmax(logits, dim=1)
            return ps[:,1] # p(y==1|x) is second columns of ps
        elif returns == 'diff':
            ps = torch.nn.functional.softmax(logits, dim=1)
            return ps[:,1] - ps[:,0] # positive
        else:
            raise ValueError('returns is not valid')



device = 'cpu'
X_tr = torch.tensor(X_train, dtype = torch.float32, device=device)
y_tr = torch.tensor(y_train, dtype = torch.float32, device=device)
X_te = torch.tensor(X_test, dtype = torch.float32, device=device)
y_te = torch.tensor(y_test, dtype = torch.float32, device=device)

dataset = torch.utils.data.TensorDataset(X_tr, y_tr)

pos_dataset = torch.utils.data.TensorDataset(X_tr[y_tr == 1], y_tr[y_tr == 1])
neg_dataset = torch.utils.data.TensorDataset(X_tr[y_tr == 0], y_tr[y_tr == 0])

batch_size = min([10000, len(pos_dataset), len(neg_dataset)])

pos_loader = torch.utils.data.DataLoader(pos_dataset, batch_size = batch_size)
neg_loader = torch.utils.data.DataLoader(neg_dataset, batch_size = batch_size)

# ppath = "params/model3_tune.pth"
model3 = Model3(X_tr.shape[1]).to(device)
# model3.load_state_dict(torch.load(ppath))

nepoch = 200
opt = torch.optim.Adam(model3.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nepoch, eta_min=1e-5)

"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
for epoch in range(nepoch):
    model3.train()
    xneg = None; yneg = None
    xpos, ypos = next(iter(pos_loader))
    nrepeat = 100
    for i, (_xneg, _yneg) in enumerate(neg_loader):
        if i == nrepeat: break
        if xneg is None:
            xneg = _xneg; yneg = _yneg
        else:
            xneg = torch.cat((xneg, _xneg), 0)
            yneg = torch.cat((yneg, _yneg), 0)

    xpos = t.cat([xpos]*nrepeat, 0)
    ypos = t.cat([ypos]*nrepeat, 0)

    xpos = xpos.to(device); xneg = xneg.to(device)

    opt.zero_grad()
    hpos = model3(xpos, returns='diff')
    hneg = model3(xneg, returns='diff')

    loss = (1 - hpos + hneg).square().mean()
    (loss).backward()
    opt.step()
    scheduler.step()

    model3.eval()
    pred_te = model3(X_te, returns='diff')
    pred_te = pred_te.detach().cpu().numpy()

    pred_tr = model3(X_tr, returns='diff')
    pred_tr = pred_tr.detach().cpu().numpy()


    fpr_te, tpr_te = roc(pred_te, y_test)
    auc_te = AUC(fpr_te, tpr_te)

    fpr_tr, tpr_tr = roc(pred_tr, y_train)
    auc_tr = AUC(fpr_tr, tpr_tr)

    loss_print = loss.detach().cpu().item()
    lr = opt.param_groups[0]['lr']

    pred_te = model3(X_te, returns='posprob')
    pred_te = pred_te.detach().cpu().numpy()

    pred_tr = model3(X_tr, returns='posprob')
    pred_tr = pred_tr.detach().cpu().numpy()

    acc_tr = sum((pred_tr > 0.5) == y_train)/len(y_train)
    acc_te = sum((pred_te > 0.5) == y_test)/len(y_test)
    print(f"ep {epoch:04d}|loss {loss:.4f}|auc_tr {auc_tr:.4f}|auc_te {auc_te:.4f}| acc_tr {acc_tr:.3f}|acc_te {acc_te:.3f}|lr {lr:.5f}")
"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

# loss_fn = torch.nn.CrossEntropyLoss()

# for epoch in range(nepoch):
#     xpos, ypos = next(iter(pos_loader))
#     xneg, yneg = next(iter(neg_loader))

#     xpos = xpos.to(device); xneg = xneg.to(device)
#     ypos = ypos.to(device); yneg = yneg.to(device)

#     X = t.cat((xpos, xneg), 0)
#     y = t.cat((ypos, yneg), 0)

#     rixs = t.randperm(len(X))

#     X = X[rixs]; y = y[rixs]

#     model3.train()
#     opt.zero_grad()
#     logits = model3(X)

#     loss = loss_fn(logits, y.long())

#     (loss).backward()
#     opt.step()
#     scheduler.step()


#     model3.eval()
#     pred_te = model3(X_te, returns='diff')
#     pred_te = pred_te.detach().cpu().numpy()

#     pred_tr = model3(X_tr, returns='diff')
#     pred_tr = pred_tr.detach().cpu().numpy()


#     fpr_te, tpr_te = roc(pred_te, y_test)
#     auc_te = AUC(fpr_te, tpr_te)

#     fpr_tr, tpr_tr = roc(pred_tr, y_train)
#     auc_tr = AUC(fpr_tr, tpr_tr)

#     loss_print = loss.detach().cpu().item()
#     lr = opt.param_groups[0]['lr']


#     pred_te = model3(X_te, returns='posprob')
#     pred_te = pred_te.detach().cpu().numpy()

#     pred_tr = model3(X_tr, returns='posprob')
#     pred_tr = pred_tr.detach().cpu().numpy()

#     acc_tr = sum((pred_tr > 0.5) == y_train)/len(y_train)
#     acc_te = sum((pred_te > 0.5) == y_test)/len(y_test)
#     print(f"ep {epoch:04d}|loss {loss:.4f}|auc_tr {auc_tr:.4f}|auc_te {auc_te:.4f}|acc_tr {acc_tr:.2f}|acc_te {acc_te:.2f}|lr {lr:.5f}")