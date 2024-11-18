import time, math, os
from tqdm import tqdm
import gc
import pickle
import random
from datetime import datetime
from operator import itemgetter
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
import collections
warnings.filterwarnings('ignore')

data_path = './news_recommendation_data/' # 数据路径
save_path = './tmp_results/'  # 中间结果存储路径

# 节约内存的一个标配函数
def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                           100*(start_mem-end_mem)/start_mem,
                                                                                                           (time.time()-starttime)/60))
    return df


# # 从训练集中取出一部分数据来调试代码
# def get_sampling_click_data(data_path, sample_nums=10000):
#     """
#         功能：训练集中采样一部分数据调试
#         data_path: 原数据的存储路径
#         sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
#     """
#     all_click = pd.read_csv(data_path + 'train_click_log.csv')
#     all_user_ids = all_click.user_id.unique()
#
#     sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
#     all_click = all_click[all_click['user_id'].isin(sample_user_ids)]
#
#     all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
#     return all_click

# 从训练集中取出一部分数据来调试代码
def get_sampling_click_data(data_path, sample_nums=10000,offline=True):
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
        all_user_ids = all_click.user_id.unique()
        sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
        all_click = all_click[all_click['user_id'].isin(sample_user_ids)]
        all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        all_user_ids1 = trn_click.user_id.unique()
        sample_user_ids1 = np.random.choice(all_user_ids1, size=sample_nums, replace=False)
        trn_click = trn_click[trn_click['user_id'].isin(sample_user_ids1)]
        trn_click = trn_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))

        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
        all_user_ids2 = tst_click.user_id.unique()
        sample_user_ids2 = np.random.choice(all_user_ids2, size=sample_nums, replace=False)
        tst_click = tst_click[tst_click['user_id'].isin(sample_user_ids2)]
        tst_click = tst_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
        # all_click = trn_click.append(tst_click)
        all_click = pd.concat([trn_click,tst_click])

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click

def get_all_click_data(data_path='../news_recommendation_data/', offline=True):
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')

        # all_click = trn_click.append(tst_click)
        all_click = pd.concat([trn_click,tst_click])

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click

# 全量训练集
# all_click_data = get_all_click_data(data_path, offline=False)
all_click_data = get_sampling_click_data(data_path,offline=False)

# 获取用户的点击新闻和时间信息
# 格式：{user1: [(item1, time1), (item2, time2)..]...}
def get_user_item_time_data(click_df):
    # 按时间排序
    click_df = click_df.sort_values('click_timestamp')
    # 构建 item-time 数据组
    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))
    # 按用户分组
    user_item_time_df = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(lambda x: make_item_time_pair(x)).reset_index().rename(columns={0: 'item_time_list'})
    # 构建数据字典
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    return user_item_time_dict

# 获取近期点击最多的新闻资讯
def get_topk_item(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


def cal_itemcf_sim(df):
    """
        基于物品的协同过滤，新闻与新闻之间的相似性矩阵计算，参数与返回值如下
        df: 数据表
        item_created_time_dict:  新闻创建时间的字典
        return : 新闻与新闻的相似性矩阵
    """
    # 获取 用户，新闻，点击时间 数据组
    user_item_time_dict = get_user_item_time_data(df)

    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 填充与构建相似度矩阵
        for i, i_click_time in item_time_list:
            item_cnt[i] += 1  #用来计算分母的
            i2i_sim.setdefault(i, {})
            for j, j_click_time in item_time_list:
                if (i == j):
                    continue
                i2i_sim[i].setdefault(j, 0)
                a = len(item_time_list)
                i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

    # 将得到的相似性矩阵保存到本地
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))

    return i2i_sim_

i2i_sim = cal_itemcf_sim(all_click_data)


# 基于商品相似度的召回i2i
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, topk_items):
    """
        基于物品(新闻)协同过滤的召回，参数和返回值如下：

        user_id: 用户id
        user_item_time_dict: 字典, 根据点击时间获取用户的点击新闻序列 {user1: [(item1, time1), (item2, time2)..]...}
        i2i_sim: 字典，新闻相似性矩阵
        sim_item_topk: 整数， 选择与当前新闻最相似的前k篇新闻
        recall_item_num: 整数， 最后的召回新闻数量
        topk_items: 列表，点击次数最多的新闻列表，用户召回补全

        return: 召回的新闻列表 {item1:score1, item2: score2...}
    """

    # 获取用户历史交互的新闻user_past_items:[(224171, 1507029683061), (223931, 1507029713061)]         user_past_items_: {223931, 224171}
    user_past_items = user_item_time_dict[user_id]
    user_past_items_ = {user_id for user_id, _ in user_past_items}

    # 遍历相似度矩阵
    item_rank = {}
    for loc, (i, click_time) in enumerate(user_past_items):
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_past_items_:
                continue

            item_rank.setdefault(j, 0)
            item_rank[j] += wij

    # 没有达到推荐的数量，则用topk热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(topk_items):
            if item in item_rank.items():  # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = - i - 100
            if len(item_rank) == recall_item_num:
                break

    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return item_rank

# 定义
user_recall_items_dict = collections.defaultdict(dict)

# 获取 （用户，新闻，点击时间）数据字典
user_item_time_dict = get_user_item_time_data(all_click_data)

# 去取新闻相似度
i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))

# 相似新闻的数量
sim_item_topk = 10

# 召回新闻数量
recall_item_num = 10

# 用户热度补全
topk_items = get_topk_item(all_click_data, k=50)

for user in tqdm(all_click_data['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim,
                                                        sim_item_topk, recall_item_num, topk_items)


# 将字典的形式转换成df
user_item_score_list = []

for user, items in tqdm(user_recall_items_dict.items()):
    for item, score in items:
        user_item_score_list.append([user, item, score])

recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])


# 生成提交文件
def submit(recall_df, topk=5, model_name=None):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 判断是不是每个用户都有5篇新闻及以上
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

# 获取测试集
tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
tst_users = tst_click['user_id'].unique()

# 从所有的召回数据中将测试集中的用户选出来
tst_recall = recall_df[recall_df['user_id'].isin(tst_users)]

# 生成提交文件
submit(tst_recall, topk=5, model_name='itemcf_baseline')