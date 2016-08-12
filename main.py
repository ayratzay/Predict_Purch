
from utils.settings import HOST, PORT, USERNAME, PASSCODE_DEBUG, CAT_DB
from utils.handlers import mysqlHandler
from utils import user_att, user_lvl, user_ssn

import pylab
pylab.show()

import numpy as np
import datetime

from sklearn.preprocessing import StandardScaler
# from scikits.statsmodels.tools import categorical
from sklearn.cross_validation import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report


def vectorize(item_list):
    temp_array = np.array(item_list)
    item_set = set(item_list)
    array = np.zeros([len(item_list), len(item_set)])
    for ind, r in enumerate(item_set):
        array[:, ind] = (temp_array == r).astype(int)
    return array


def process_lvls(lvl_list, shape):

    lvl_array = np.zeros((shape[0], 9 * 12), dtype= float) #colnams = 'score', 'moves', 'time', 'ERwin', 'ERdrop', 'ERmoves', 'ERbombs', 'ERexit', 'ERseconds'
    for lvlline in lvl_list:
        try:
            uid, score, moves, time, lvl, win, drp, ermoves, bombs, exit, secs = lvlline
            lvl  =  int(lvl)
            uind, rind = index_dict[uid], (lvl - 1) * 9
            if score < 0:
                print(i)
            lvl_array[uind, 0 + rind] += score/1000.
            lvl_array[uind, 1 + rind] += moves
            lvl_array[uind, 2 + rind] += time/60.
            lvl_array[uind, 3 + rind] += win
            lvl_array[uind, 4 + rind] += drp
            lvl_array[uind, 5 + rind] += ermoves
            lvl_array[uind, 6 + rind] += bombs
            lvl_array[uind, 7 + rind] += exit
            lvl_array[uind, 8 + rind] += secs
        except:
            print (lvlline)
            break
    return lvl_array


def process_sessions(ssndata, shape):
    ssn_array = np.zeros((shape[0], 8), dtype= int) #colnams = 0_3, 3_6, 6_9, 9_12, 12_15, 15_18, 18_21, 21_24
    for ssn_line in ssndata:
        uid = ssn_line[0]
        try:
            uind = index_dict[uid]
            ssn_array[uind] += np.array(ssn_line)[1:,]
            for i in range(0, 8):
                if ssn_array[uind, i] > 360:
                    if i != 7:
                        ssn_array[uind, i + 1] += ssn_array[uind, i] - 360
                        ssn_array[uind, i] = 360
        except:
            print (ssn_line)
    return ssn_array


def create_model(X_table, y_table):

    p_true = float(y_table.sum()) * 1 / y_table.shape[0]
    bmask = np.random.choice([True, False], size=len(y_table), p=[p_true, 1 - p_true])
    balance_mask = np.any([bmask, y == 1], axis=0)

    dtrain = xgb.DMatrix(X_table[balance_mask], label=y_table[balance_mask])

    param = {'bst:max_depth': 5, 'bst:eta': 0.1, 'silent': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    param['min_child_weight'] = 50
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    evallist = [(dtrain, 'train')]

    num_round = 50
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10, verbose_eval=49)
    return bst


CATS = mysqlHandler(HOST, PORT, USERNAME, PASSCODE_DEBUG, CAT_DB)

fdate = "'2016-02-01'"
udata, lvldata, ssndata = CATS.get_results(user_att % fdate), CATS.get_results(user_lvl % fdate), CATS.get_results(user_ssn % fdate)


uid, uparams, ucreated, uref, upaid = [], [], [], [], []
for i in udata:
    iid, created_at, age, sex, ref, ref_type, fp = i
    uparams.append([age, sex, ref_type])
    ucreated.append(created_at)
    uref.append(ref)
    uid.append(iid)
    if not fp:
        upaid.append(np.nan)
    else:
        upaid.append(fp)

uarray, date_index, piadarray = np.array(uparams), np.array(ucreated), np.array(upaid)
refarray = vectorize(uref)
index_dict = dict(zip(np.array(uid), range(len(uid))))

lvl_array = process_lvls(lvldata, uarray.shape)
ssn_array = process_sessions(ssndata, uarray.shape)


# for q, w, in zip(lvl_array.sum(axis = 0), ['score', 'moves', 'time', 'ERwin', 'ERdrop', 'ERmoves', 'ERbombs', 'ERexit', 'ERseconds'] * 12):
#     print(q,w)


scaler = StandardScaler()
X_all = np.concatenate([uarray, refarray, ssn_array, lvl_array], axis = 1)
X_all = scaler.fit_transform(X_all)

y_all = (np.isnan(piadarray) == False).astype(int)

not_empty_mask = np.any([lvl_array.sum(axis = 1) > 3, ssn_array.sum(axis = 1) > 20], axis = 0)
# date_index, piadarray

X_clean, y_clean = X_all[not_empty_mask, :], y_all[not_empty_mask]








# mask = np.any([np.random.randint(0,30,len(y_bin)) > 28, y_bin==1], axis = 0)
mask = np.random.randint(1,2, len(y_bin)) > 0

X_train, X_test, y_train, y_test = train_test_split(X[mask], y_bin[mask], test_size = 0.25, random_state = 666, stratify = y[mask])

sum(y_train) * 0.999 / y_train.shape
sum(y_test) * 0.999 / y_test.shape



from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve



lr = LogisticRegression()
gnb = GaussianNB()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100)

for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest')]:
    clf.fit(X_train, y_train)
    ypred = clf.predict(X_test)

    print (name)
    print (confusion_matrix(y_test, ypred, labels=[0, 1]))
    print (accuracy_score(y_test, ypred))
    print (classification_report(y_test, ypred))





# dtrain = xgb.DMatrix( X_train, label=y_train, missing = -999.0)
# dtest = xgb.DMatrix( X_test, label=y_test, missing = -999.0)

# dtrain.save_binary(r"Gazlowe\data\train.buffer")
# dtrain = xgb.DMatrix(r"Gazlowe\data\train.buffer")

# dtest.save_binary(r"Gazlowe\data\train.buffer")
# dtest = xgb.DMatrix(r"Gazlowe\data\train.buffer")

# param = {'bst:max_depth':6, 'bst:eta':0.1, 'silent':1, 'objective':'binary:logistic' }
# param['nthread'] = 4
# param['eval_metric'] = 'auc'
# param['min_child_weight'] = 100
# param['subsample'] = 0.7
# param['colsample_bytree'] = 0.7
# evallist  = [(dtest,'eval'), (dtrain,'train')]



num_round = 100
bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)

ypred = bst.predict(dtest)

plt.hist(ypred)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print (confusion_matrix(y_test, ypred> 0.5, labels=[0, 1]))
print (accuracy_score(y_test, ypred> 0.5))
print (classification_report(y_test, ypred> 0.4))



ypred = bst.predict(dtest)




# from collections import Counter
# Counter(y_test)
# Counter(y_train)

# import os
# mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
# os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']


# xgb.__version__
# data = np.random.rand(5,10) # 5 entities, each contains 10 features
# label = np.random.randint(2, size=5) # binary target
# dtrain = xgb.DMatrix( data, label=label)
#
# dtest = dtrain
#
# param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
# param['nthread'] = 4
# param['eval_metric'] = 'auc'
#
# evallist  = [(dtest,'eval'), (dtrain,'train')]
#
# num_round = 10
# bst = xgb.train( param, dtrain, num_round, evallist )
#
# bst.dump_model('dump.raw.txt')







#
# time_fhorizon = datetime.datetime(2016, 4, 1)
# time_bhorizon = datetime.datetime(2016, 2, 1)
# predicted_mask = date_index < datetime.datetime(2016, 4, 1)
#
#
#
# time_mask_layer1 = np.all([date_index < time_fhorizon, date_index > time_bhorizon], axis=0)
# time_mask_layer2 = np.all([date_index < time_fhorizon, y_clean == 1], axis=0)
# time_mask = np.any([time_mask_layer1, time_mask_layer2], axis=0)
# X, y = X_clean[time_mask, :], y_clean[time_mask]
#
# model_dict = {}
# model_dict[time_fhorizon] = create_model(X, y)
#
#
# time_fhorizon += datetime.timedelta(1)
# time_bhorizon += datetime.timedelta(1)
#
#
# def eveluate_model(model):
#
# time_mask_eval = np.all([date_index < time_fhorizon, predicted_mask == False], axis=0)
# deval = xgb.DMatrix(X_clean[time_mask_eval], label=y_clean[time_mask_eval])
#
# mdl = model_dict[time_fhorizon - datetime.timedelta(1)]
# y_pred = (mdl.predict(deval) > 0.4).astype(int)
# [(TF,FP),(FN,TN)] = confusion_matrix(y_clean[time_mask_eval], y_pred).tolist()
#
# predicted_mask = np.any(predicted_mask, time_mask_eval)