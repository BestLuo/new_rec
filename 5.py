import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import gc, os
import time
from datetime import datetime
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85  # 程序最多只能占用指定gpu75%的显存
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)


data_path = './news_recommendation_data/'
save_path = './tmp_results/'
offline = False

# 重新读取数据的时候，发现click_article_id是一个浮点数，所以将其转换成int类型
train_user_item_feats_df = pd.read_csv(save_path + 'train_user_item_feats_df.csv')
train_user_item_feats_df['click_article_id'] = train_user_item_feats_df['click_article_id'].astype(int)

if offline:
    val_user_item_feats_df = pd.read_csv(save_path + 'val_user_item_feats_df.csv')
    val_user_item_feats_df['click_article_id'] = val_user_item_feats_df['click_article_id'].astype(int)
else:
    val_user_item_feats_df = None

test_user_item_feats_df = pd.read_csv(save_path + 'test_user_item_feats_df.csv')
test_user_item_feats_df['click_article_id'] = test_user_item_feats_df['click_article_id'].astype(int)

# 做特征的时候为了方便，给测试集也打上了一个无效的标签，这里直接删掉就行
del test_user_item_feats_df['label']


def submit(recall_df, topk=5, model_name=None):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 判断是不是每个用户都有5篇文章及以上
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    assert tmp.min() >= topk

    del recall_df['pred_score']
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()

    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    # 按照提交格式定义列名
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2',
                                    3: 'article_3', 4: 'article_4', 5: 'article_5'})

    save_name = save_path + model_name + '_' + datetime.today().strftime('%m-%d') + '.csv'
    submit.to_csv(save_name, index=False, header=True)

# 排序结果归一化
def norm_sim(sim_df, weight=0.0):
    # print(sim_df.head())
    min_sim = sim_df.min()
    max_sim = sim_df.max()
    if max_sim == min_sim:
        sim_df = sim_df.apply(lambda sim: 1.0)
    else:
        sim_df = sim_df.apply(lambda sim: 1.0 * (sim - min_sim) / (max_sim - min_sim))

    sim_df = sim_df.apply(lambda sim: sim + weight)  # plus one
    return sim_df


# 防止中间出错之后重新读取数据
train_user_item_feats_df_rank_model = train_user_item_feats_df.copy()

if offline:
    val_user_item_feats_df_rank_model = val_user_item_feats_df.copy()

test_user_item_feats_df_rank_model = test_user_item_feats_df.copy()

# 定义特征列
lgb_cols = ['sim0', 'time_diff0', 'word_diff0','sim_max', 'sim_min', 'sim_sum',
            'sim_mean', 'score','click_size', 'time_diff_mean', 'active_level',
            'click_environment','click_deviceGroup', 'click_os', 'click_country',
            'click_region','click_referrer_type', 'user_time_hob1', 'user_time_hob2',
            'words_hbo', 'category_id', 'created_at_ts','words_count']

# 排序模型分组
train_user_item_feats_df_rank_model.sort_values(by=['user_id'], inplace=True)
g_train = train_user_item_feats_df_rank_model.groupby(['user_id'], as_index=False).count()["label"].values

if offline:
    val_user_item_feats_df_rank_model.sort_values(by=['user_id'], inplace=True)
    g_val = val_user_item_feats_df_rank_model.groupby(['user_id'], as_index=False).count()["label"].values

# 排序模型定义
lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                            max_depth=-1, n_estimators=100, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                            learning_rate=0.01, min_child_weight=50, random_state=2021, n_jobs=16)

# 排序模型训练
if offline:
    lgb_ranker.fit(train_user_item_feats_df_rank_model[lgb_cols], train_user_item_feats_df_rank_model['label'], group=g_train,
                eval_set=[(val_user_item_feats_df_rank_model[lgb_cols], val_user_item_feats_df_rank_model['label'])],
                eval_group= [g_val], eval_at=[1, 2, 3, 4, 5], eval_metric=['ndcg', ], early_stopping_rounds=50, )
else:
    lgb_ranker.fit(train_user_item_feats_df[lgb_cols], train_user_item_feats_df['label'], group=g_train)

# 模型预测
test_user_item_feats_df['pred_score'] = lgb_ranker.predict(test_user_item_feats_df[lgb_cols], num_iteration=lgb_ranker.best_iteration_)

# 将这里的排序结果保存一份，用户后面的模型融合
test_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']].to_csv(save_path + 'lgb_ranker_score.csv', index=False)

# 预测结果重新排序, 及生成提交结果
rank_results = test_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']]
rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
submit(rank_results, topk=5, model_name='lgb_ranker')


# 五折交叉验证，这里的五折交叉是以用户为目标进行五折划分
#  这一部分与前面的单独训练和验证是分开的
def get_kfold_users(train_df, n=5):
    user_ids = train_df['user_id'].unique()
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set


k_fold = 5
train_df = train_user_item_feats_df_rank_model
user_set = get_kfold_users(train_df, n=k_fold)

score_list = []
score_df = train_df[['user_id', 'click_article_id', 'label']]
sub_preds = np.zeros(test_user_item_feats_df_rank_model.shape[0])

# 五折交叉验证，并将中间结果保存用于staking
for n_fold, valid_user in enumerate(user_set):
    train_idx = train_df[~train_df['user_id'].isin(valid_user)]  # add slide user
    valid_idx = train_df[train_df['user_id'].isin(valid_user)]

    # 训练集与验证集的用户分组
    train_idx.sort_values(by=['user_id'], inplace=True)
    g_train = train_idx.groupby(['user_id'], as_index=False).count()["label"].values

    valid_idx.sort_values(by=['user_id'], inplace=True)
    g_val = valid_idx.groupby(['user_id'], as_index=False).count()["label"].values

    # 定义模型
    lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                max_depth=-1, n_estimators=100, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                                learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=16)
    # 训练模型
    lgb_ranker.fit(train_idx[lgb_cols], train_idx['label'], group=g_train,
                   eval_set=[(valid_idx[lgb_cols], valid_idx['label'])], eval_group=[g_val],
                   eval_at=[1, 2, 3, 4, 5], eval_metric=['ndcg', ], )

    # 预测验证集结果
    valid_idx['pred_score'] = lgb_ranker.predict(valid_idx[lgb_cols], num_iteration=lgb_ranker.best_iteration_)

    # 对输出结果进行归一化
    valid_idx['pred_score'] = valid_idx[['pred_score']].transform(lambda x: norm_sim(x))

    valid_idx.sort_values(by=['user_id', 'pred_score'])
    valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 将验证集的预测结果放到一个列表中，后面进行拼接
    score_list.append(valid_idx[['user_id', 'click_article_id', 'pred_score', 'pred_rank']])

    # 如果是线上测试，需要计算每次交叉验证的结果相加，最后求平均
    if not offline:
        sub_preds += lgb_ranker.predict(test_user_item_feats_df_rank_model[lgb_cols], lgb_ranker.best_iteration_)

score_df_ = pd.concat(score_list, axis=0)
score_df = score_df.merge(score_df_, how='left', on=['user_id', 'click_article_id'])
# 保存训练集交叉验证产生的新特征
score_df[['user_id', 'click_article_id', 'pred_score', 'pred_rank', 'label']].to_csv(
    save_path + 'train_lgb_ranker_feats.csv', index=False)

# 测试集的预测结果，多次交叉验证求平均,将预测的score和对应的rank特征保存，可以用于后面的staking，这里还可以构造其他更多的特征
test_user_item_feats_df_rank_model['pred_score'] = sub_preds / k_fold
test_user_item_feats_df_rank_model['pred_score'] = test_user_item_feats_df_rank_model['pred_score'].transform(
    lambda x: norm_sim(x))
test_user_item_feats_df_rank_model.sort_values(by=['user_id', 'pred_score'])
test_user_item_feats_df_rank_model['pred_rank'] = test_user_item_feats_df_rank_model.groupby(['user_id'])[
    'pred_score'].rank(ascending=False, method='first')

# 保存测试集交叉验证的新特征
test_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score', 'pred_rank']].to_csv(
    save_path + 'test_lgb_ranker_feats.csv', index=False)

# 模型及参数的定义
lgb_Classfication = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                            max_depth=-1, n_estimators=500, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                            learning_rate=0.01, min_child_weight=50, random_state=2021, n_jobs=16, verbose=10)

# 模型训练
if offline:
    lgb_Classfication.fit(train_user_item_feats_df_rank_model[lgb_cols], train_user_item_feats_df_rank_model['label'],
                    eval_set=[(val_user_item_feats_df_rank_model[lgb_cols], val_user_item_feats_df_rank_model['label'])],
                    eval_metric=['auc', ],early_stopping_rounds=50, )
else:
    lgb_Classfication.fit(train_user_item_feats_df_rank_model[lgb_cols], train_user_item_feats_df_rank_model['label'])
# 模型预测
test_user_item_feats_df['pred_score'] = lgb_Classfication.predict_proba(test_user_item_feats_df[lgb_cols])[:,1]

# 将这里的排序结果保存一份，用户后面的模型融合
test_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']].to_csv(save_path + 'lgb_cls_score.csv', index=False)
# 预测结果重新排序, 及生成提交结果
rank_results = test_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']]
rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
submit(rank_results, topk=5, model_name='lgb_cls')


# 五折交叉验证，这里的五折交叉是以用户为目标进行五折划分
#  这一部分与前面的单独训练和验证是分开的
def get_kfold_users(train_df, n=5):
    user_ids = train_df['user_id'].unique()
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set


k_fold = 5
train_df = train_user_item_feats_df_rank_model
user_set = get_kfold_users(train_df, n=k_fold)

score_list = []
score_df = train_df[['user_id', 'click_article_id', 'label']]
sub_preds = np.zeros(test_user_item_feats_df_rank_model.shape[0])

# 五折交叉验证，并将中间结果保存用于staking
for n_fold, valid_user in enumerate(user_set):
    train_idx = train_df[~train_df['user_id'].isin(valid_user)]  # add slide user
    valid_idx = train_df[train_df['user_id'].isin(valid_user)]

    # 模型及参数的定义
    lgb_Classfication = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                           max_depth=-1, n_estimators=100, subsample=0.7, colsample_bytree=0.7,
                                           subsample_freq=1,
                                           learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=16,
                                           verbose=10)
    # 训练模型
    lgb_Classfication.fit(train_idx[lgb_cols], train_idx['label'], eval_set=[(valid_idx[lgb_cols], valid_idx['label'])],
                          eval_metric=['auc', ], )

    # 预测验证集结果
    valid_idx['pred_score'] = lgb_Classfication.predict_proba(valid_idx[lgb_cols],
                                                              num_iteration=lgb_Classfication.best_iteration_)[:, 1]

    # 对输出结果进行归一化 分类模型输出的值本身就是一个概率值不需要进行归一化
    # valid_idx['pred_score'] = valid_idx[['pred_score']].transform(lambda x: norm_sim(x))

    valid_idx.sort_values(by=['user_id', 'pred_score'])
    valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 将验证集的预测结果放到一个列表中，后面进行拼接
    score_list.append(valid_idx[['user_id', 'click_article_id', 'pred_score', 'pred_rank']])

    # 如果是线上测试，需要计算每次交叉验证的结果相加，最后求平均
    if not offline:
        sub_preds += lgb_Classfication.predict_proba(test_user_item_feats_df_rank_model[lgb_cols],
                                                     num_iteration=lgb_Classfication.best_iteration_)[:, 1]

score_df_ = pd.concat(score_list, axis=0)
score_df = score_df.merge(score_df_, how='left', on=['user_id', 'click_article_id'])
# 保存训练集交叉验证产生的新特征
score_df[['user_id', 'click_article_id', 'pred_score', 'pred_rank', 'label']].to_csv(
    save_path + 'train_lgb_cls_feats.csv', index=False)

# 测试集的预测结果，多次交叉验证求平均,将预测的score和对应的rank特征保存，可以用于后面的staking，这里还可以构造其他更多的特征
test_user_item_feats_df_rank_model['pred_score'] = sub_preds / k_fold
test_user_item_feats_df_rank_model['pred_score'] = test_user_item_feats_df_rank_model['pred_score'].transform(
    lambda x: norm_sim(x))
test_user_item_feats_df_rank_model.sort_values(by=['user_id', 'pred_score'])
test_user_item_feats_df_rank_model['pred_rank'] = test_user_item_feats_df_rank_model.groupby(['user_id'])[
    'pred_score'].rank(ascending=False, method='first')

# 保存测试集交叉验证的新特征
test_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score', 'pred_rank']].to_csv(
    save_path + 'test_lgb_cls_feats.csv', index=False)

# 预测结果重新排序, 及生成提交结果
rank_results = test_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score']]
rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
submit(rank_results, topk=5, model_name='lgb_cls')

if offline:
    all_data = pd.read_csv('./news_recommendation_data/train_click_log.csv')
else:
    train_data = pd.read_csv('./news_recommendation_data/train_click_log.csv')
    test_data = pd.read_csv('./news_recommendation_data/testA_click_log.csv')
    all_data = train_data.append(test_data)

hist_click =all_data[['user_id', 'click_article_id']].groupby('user_id').agg({list}).reset_index()
his_behavior_df = pd.DataFrame()
his_behavior_df['user_id'] = hist_click['user_id']
his_behavior_df['hist_click_article_id'] = hist_click['click_article_id']

train_user_item_feats_df_din_model = train_user_item_feats_df.copy()

if offline:
    val_user_item_feats_df_din_model = val_user_item_feats_df.copy()
else:
    val_user_item_feats_df_din_model = None

test_user_item_feats_df_din_model = test_user_item_feats_df.copy()
train_user_item_feats_df_din_model = train_user_item_feats_df_din_model.merge(his_behavior_df, on='user_id')

if offline:
    val_user_item_feats_df_din_model = val_user_item_feats_df_din_model.merge(his_behavior_df, on='user_id')
else:
    val_user_item_feats_df_din_model = None

test_user_item_feats_df_din_model = test_user_item_feats_df_din_model.merge(his_behavior_df, on='user_id')

# 导入deepctr
from deepctr.models import DIN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
# import tensorflow as tf

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 数据准备函数
def get_din_feats_columns(df, dense_fea, sparse_fea, behavior_fea, his_behavior_fea, emb_dim=32, max_len=100):
    """
    数据准备函数:
    df: 数据集
    dense_fea: 数值型特征列
    sparse_fea: 离散型特征列
    behavior_fea: 用户的候选行为特征列
    his_behavior_fea: 用户的历史行为特征列
    embedding_dim: embedding的维度， 这里为了简单， 统一把离散型特征列采用一样的隐向量维度
    max_len: 用户序列的最大长度
    """

    # sparse_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].nunique() + 1, use_hash=True, embedding_dim=emb_dim) for feat in sparse_fea]
    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].max() + 1, embedding_dim=emb_dim) for feat in
                              sparse_fea]

    dense_feature_columns = [DenseFeat(feat, 1, ) for feat in dense_fea]

    # var_feature_columns = [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=df['click_article_id'].nunique() + 1, use_hash=True, embedding_dim=emb_dim, embedding_name='click_article_id'), maxlen=max_len) for feat in hist_behavior_fea]
    var_feature_columns = [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=df['click_article_id'].max() + 1,
                                                       embedding_dim=emb_dim, embedding_name='click_article_id'),
                                            maxlen=max_len) for feat in hist_behavior_fea]

    dnn_feature_columns = sparse_feature_columns + dense_feature_columns + var_feature_columns

    # 建立x, x是一个字典的形式
    x = {}
    for name in get_feature_names(dnn_feature_columns):
        if name in his_behavior_fea:
            # 这是历史行为序列
            his_list = [l for l in df[name]]
            x[name] = pad_sequences(his_list, maxlen=max_len, padding='post')  # 二维数组
        else:
            x[name] = df[name].values

    return x, dnn_feature_columns

# 把特征分开
sparse_fea = ['user_id', 'click_article_id', 'category_id', 'click_environment', 'click_deviceGroup',
              'click_os', 'click_country', 'click_region', 'click_referrer_type', 'is_cat_hab']

behavior_fea = ['click_article_id']

hist_behavior_fea = ['hist_click_article_id']

dense_fea = ['sim0', 'time_diff0', 'word_diff0', 'sim_max', 'sim_min', 'sim_sum', 'sim_mean', 'score',
             'rank','click_size','time_diff_mean','active_level','user_time_hob1','user_time_hob2',
             'words_hbo','words_count']

# dense特征进行归一化, 神经网络训练都需要将数值进行归一化处理
mm = MinMaxScaler()

# 下面是做一些特殊处理，当在其他的地方出现无效值的时候，不处理无法进行归一化，刚开始可以先把他注释掉，在运行了下面的代码
# 之后如果发现报错，应该先去想办法处理如何不出现inf之类的值
# train_user_item_feats_df_din_model.replace([np.inf, -np.inf], 0, inplace=True)
# test_user_item_feats_df_din_model.replace([np.inf, -np.inf], 0, inplace=True)

for feat in dense_fea:
    mm.fit(train_user_item_feats_df_din_model[[feat]])
    train_user_item_feats_df_din_model[feat] = mm.transform(train_user_item_feats_df_din_model[[feat]])

    if val_user_item_feats_df_din_model is not None:
        val_user_item_feats_df_din_model[feat] = mm.transform(val_user_item_feats_df_din_model[[feat]])

    test_user_item_feats_df_din_model[feat] = mm.transform(test_user_item_feats_df_din_model[[feat]])

dense_fea = [x for x in dense_fea if x != 'label']
# 准备训练数据
x_train, dnn_feature_columns = get_din_feats_columns(train_user_item_feats_df_din_model, dense_fea,
                                                     sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
y_train = train_user_item_feats_df_din_model['label'].values

if offline:
    # 准备验证数据
    x_val, dnn_feature_columns = get_din_feats_columns(val_user_item_feats_df_din_model, dense_fea,
                                                       sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
    y_val = val_user_item_feats_df_din_model['label'].values

x_test, dnn_feature_columns = get_din_feats_columns(test_user_item_feats_df_din_model, dense_fea,
                                                    sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)

# 建立模型
model = DIN(dnn_feature_columns, behavior_fea)

# 查看模型结构
model.summary()

# 模型编译
model.compile('adam', 'binary_crossentropy',metrics=['binary_crossentropy', tf.keras.metrics.AUC()])

# 模型训练
if offline:
    history = model.fit(x_train, y_train, verbose=1, epochs=10, validation_data=(x_val, y_val) , batch_size=256)
else:
    #history = model.fit(x_train, y_train, verbose=1, epochs=3, validation_split=0.3, batch_size=256)
    history = model.fit(x_train, y_train, verbose=1, epochs=2, batch_size=256)

# 模型预测
test_user_item_feats_df_din_model['pred_score'] = model.predict(x_test, verbose=1, batch_size=256)
test_user_item_feats_df_din_model[['user_id', 'click_article_id', 'pred_score']].to_csv(save_path + 'din_rank_score.csv', index=False)

# 预测结果重新排序, 及生成提交结果
rank_results = test_user_item_feats_df_din_model[['user_id', 'click_article_id', 'pred_score']]
submit(rank_results, topk=5, model_name='din')


# 五折交叉验证，这里的五折交叉是以用户为目标进行五折划分
#  这一部分与前面的单独训练和验证是分开的
def get_kfold_users(train_df, n=5):
    user_ids = train_df['user_id'].unique()
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set


k_fold = 5
train_df = train_user_item_feats_df_din_model
user_set = get_kfold_users(train_df, n=k_fold)

score_list = []
score_df = train_df[['user_id', 'click_article_id', 'label']]
sub_preds = np.zeros(test_user_item_feats_df_rank_model.shape[0])

dense_fea = [x for x in dense_fea if x != 'label']
x_test, dnn_feature_columns = get_din_feats_columns(test_user_item_feats_df_din_model, dense_fea,
                                                    sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)

# 五折交叉验证，并将中间结果保存用于staking
for n_fold, valid_user in enumerate(user_set):
    train_idx = train_df[~train_df['user_id'].isin(valid_user)]  # add slide user
    valid_idx = train_df[train_df['user_id'].isin(valid_user)]

    # 准备训练数据
    x_train, dnn_feature_columns = get_din_feats_columns(train_idx, dense_fea,
                                                         sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
    y_train = train_idx['label'].values

    # 准备验证数据
    x_val, dnn_feature_columns = get_din_feats_columns(valid_idx, dense_fea,
                                                       sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
    y_val = valid_idx['label'].values

    history = model.fit(x_train, y_train, verbose=1, epochs=2, validation_data=(x_val, y_val), batch_size=256)

    # 预测验证集结果
    valid_idx['pred_score'] = model.predict(x_val, verbose=1, batch_size=256)

    valid_idx.sort_values(by=['user_id', 'pred_score'])
    valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 将验证集的预测结果放到一个列表中，后面进行拼接
    score_list.append(valid_idx[['user_id', 'click_article_id', 'pred_score', 'pred_rank']])

    # 如果是线上测试，需要计算每次交叉验证的结果相加，最后求平均
    if not offline:
        sub_preds += model.predict(x_test, verbose=1, batch_size=256)[:, 0]

score_df_ = pd.concat(score_list, axis=0)
score_df = score_df.merge(score_df_, how='left', on=['user_id', 'click_article_id'])
# 保存训练集交叉验证产生的新特征
score_df[['user_id', 'click_article_id', 'pred_score', 'pred_rank', 'label']].to_csv(
    save_path + 'train_din_cls_feats.csv', index=False)

# 测试集的预测结果，多次交叉验证求平均,将预测的score和对应的rank特征保存，可以用于后面的staking，这里还可以构造其他更多的特征
test_user_item_feats_df_din_model['pred_score'] = sub_preds / k_fold
test_user_item_feats_df_din_model['pred_score'] = test_user_item_feats_df_din_model['pred_score'].transform(
    lambda x: norm_sim(x))
test_user_item_feats_df_din_model.sort_values(by=['user_id', 'pred_score'])
test_user_item_feats_df_din_model['pred_rank'] = test_user_item_feats_df_din_model.groupby(['user_id'])[
    'pred_score'].rank(ascending=False, method='first')

# 保存测试集交叉验证的新特征
test_user_item_feats_df_din_model[['user_id', 'click_article_id', 'pred_score', 'pred_rank']].to_csv(
    save_path + 'test_din_cls_feats.csv', index=False)

# 读取多个模型的排序结果文件
lgb_ranker = pd.read_csv(save_path + 'lgb_ranker_score.csv')
lgb_cls = pd.read_csv(save_path + 'lgb_cls_score.csv')
din_ranker = pd.read_csv(save_path + 'din_rank_score.csv')

# 这里也可以换成交叉验证输出的测试结果进行加权融合

rank_model = {'lgb_ranker': lgb_ranker,
              'lgb_cls': lgb_cls,
              'din_ranker': din_ranker}


def get_ensumble_predict_topk(rank_model, topk=5):
    final_recall = rank_model['lgb_cls'].append(rank_model['din_ranker'])
    rank_model['lgb_ranker']['pred_score'] = rank_model['lgb_ranker']['pred_score'].transform(lambda x: norm_sim(x))

    final_recall = final_recall.append(rank_model['lgb_ranker'])
    final_recall = final_recall.groupby(['user_id', 'click_article_id'])['pred_score'].sum().reset_index()

    submit(final_recall, topk=topk, model_name='ensemble_fuse')

get_ensumble_predict_topk(rank_model)

# 读取多个模型的交叉验证生成的结果文件
# 训练集
train_lgb_ranker_feats = pd.read_csv(save_path + 'train_lgb_ranker_feats.csv')
train_lgb_cls_feats = pd.read_csv(save_path + 'train_lgb_cls_feats.csv')
train_din_cls_feats = pd.read_csv(save_path + 'train_din_cls_feats.csv')

# 测试集
test_lgb_ranker_feats = pd.read_csv(save_path + 'test_lgb_ranker_feats.csv')
test_lgb_cls_feats = pd.read_csv(save_path + 'test_lgb_cls_feats.csv')
test_din_cls_feats = pd.read_csv(save_path + 'test_din_cls_feats.csv')

# 将多个模型输出的特征进行拼接

finall_train_ranker_feats = train_lgb_ranker_feats[['user_id', 'click_article_id', 'label']]
finall_test_ranker_feats = test_lgb_ranker_feats[['user_id', 'click_article_id']]

for idx, train_model in enumerate([train_lgb_ranker_feats, train_lgb_cls_feats, train_din_cls_feats]):
    for feat in [ 'pred_score', 'pred_rank']:
        col_name = feat + '_' + str(idx)
        finall_train_ranker_feats[col_name] = train_model[feat]

for idx, test_model in enumerate([test_lgb_ranker_feats, test_lgb_cls_feats, test_din_cls_feats]):
    for feat in [ 'pred_score', 'pred_rank']:
        col_name = feat + '_' + str(idx)
        finall_test_ranker_feats[col_name] = test_model[feat]

# 定义一个逻辑回归模型再次拟合交叉验证产生的特征对测试集进行预测
# 这里需要注意的是，在做交叉验证的时候可以构造多一些与输出预测值相关的特征，来丰富这里简单模型的特征
from sklearn.linear_model import LogisticRegression

feat_cols = ['pred_score_0', 'pred_rank_0', 'pred_score_1', 'pred_rank_1', 'pred_score_2', 'pred_rank_2']

train_x = finall_train_ranker_feats[feat_cols]
train_y = finall_train_ranker_feats['label']

test_x = finall_test_ranker_feats[feat_cols]

# 定义模型
lr = LogisticRegression()

# 模型训练
lr.fit(train_x, train_y)

# 模型预测
finall_test_ranker_feats['pred_score'] = lr.predict_proba(test_x)[:, 1]

# 预测结果重新排序, 及生成提交结果
rank_results = finall_test_ranker_feats[['user_id', 'click_article_id', 'pred_score']]
submit(rank_results, topk=5, model_name='ensumble_staking')

