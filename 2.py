
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', size=13)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

import os,gc,re,warnings,sys
warnings.filterwarnings("ignore")

# 文件路径
path = './news_recommendation_data/' # 天池平台路径

# 训练集
train_clicks = pd.read_csv(path+'train_click_log.csv')
item_df = pd.read_csv(path+'articles.csv')
item_df = item_df.rename(columns={'article_id': 'click_article_id'})  #重命名，方便后续match
item_emb_df = pd.read_csv(path+'articles_emb.csv')

# 测试集
test_clicks = pd.read_csv(path+'testA_click_log.csv')

# 对每个用户的点击数据，根据时间戳排序
train_clicks['rank'] = train_clicks.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
test_clicks['rank'] = test_clicks.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)

#计算用户点击文章的次数，存放到新字段click_cnts中
train_clicks['click_cnts'] = train_clicks.groupby(['user_id'])['click_timestamp'].transform('count')
test_clicks['click_cnts'] = test_clicks.groupby(['user_id'])['click_timestamp'].transform('count')

train_clicks = train_clicks.merge(item_df, how='left', on=['click_article_id'])
train_clicks.head()

#用户点击日志信息
train_clicks.info()