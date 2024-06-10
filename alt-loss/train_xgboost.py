
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

import xgboost as xgb
print("shallow model:")
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() # == # negative / # positive
# scale_pos_weight = 1.
bst = xgb.XGBClassifier(n_estimators=100, max_depth=5, objective='binary:logistic', scale_pos_weight=scale_pos_weight)
bst.fit(X_train, y_train)
pred_te = bst.predict(X_test)
pred_tr = bst.predict(X_train)

prob_tr = bst.predict_proba(X_train)
prob_te = bst.predict_proba(X_test)
score_tr = prob_tr[:,1] - prob_tr[:,0]
score_te = prob_te[:,1] - prob_te[:,0]

fpr_te, tpr_te = roc(score_te, y_test)
auc_te = AUC(fpr_te, tpr_te)

fpr_tr, tpr_tr = roc(score_tr, y_train)
auc_tr = AUC(fpr_tr, tpr_tr)

acc_tr = sum((pred_tr > 0.5) == y_train)/len(y_train)
acc_te = sum((pred_te > 0.5) == y_test)/len(y_test)
print(f"auc_tr {auc_tr:.4f}|auc_te {auc_te:.4f}|acc_tr {acc_tr:.2f}|acc_te {acc_te:.2f}")


print("deeper model:")
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() # == # negative / # positive
# scale_pos_weight = 1.
bst = xgb.XGBClassifier(n_estimators=100, max_depth=20, objective='binary:logistic', scale_pos_weight=scale_pos_weight)
bst.fit(X_train, y_train)
pred_te = bst.predict(X_test)
pred_tr = bst.predict(X_train)

prob_tr = bst.predict_proba(X_train)
prob_te = bst.predict_proba(X_test)
score_tr = prob_tr[:,1] - prob_tr[:,0]
score_te = prob_te[:,1] - prob_te[:,0]

fpr_te, tpr_te = roc(score_te, y_test)
auc_te = AUC(fpr_te, tpr_te)

fpr_tr, tpr_tr = roc(score_tr, y_train)
auc_tr = AUC(fpr_tr, tpr_tr)

acc_tr = sum((pred_tr > 0.5) == y_train)/len(y_train)
acc_te = sum((pred_te > 0.5) == y_test)/len(y_test)
print(f"auc_tr {auc_tr:.4f}|auc_te {auc_te:.4f}|acc_tr {acc_tr:.2f}|acc_te {acc_te:.2f}")