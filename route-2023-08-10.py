#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: XuXu

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, chi2_contingency
import streamlit as st
import plotly.express as px
import plotly.io as pio

pio.templates.default = 'plotly_white'


# 00 相关性函数
# ------------------------------------------------------------------------------------------------------
def corr_pearson(ser_1, ser_2):
    return pearsonr(ser_1, ser_2).correlation


def corr_kendalltau(ser_1, ser_2):
    return kendalltau(ser_1, ser_2).correlation


def corr_cramersv(ser_1, ser_2):
    ser_x, ser_y = ser_1.copy(), ser_2.copy()
    ser_x, ser_y = ser_x.astype('str'), ser_y.astype('str')
    dataset = pd.crosstab(ser_x, ser_y).values
    x2 = chi2_contingency(dataset, correction=False)[0]
    n_free = np.sum(dataset)
    minimum_dimension = min(dataset.shape) - 1
    corr_v = np.sqrt((x2 / n_free) / minimum_dimension)
    return corr_v


def corr_cramersv_con_cat(ser_con, ser_cat, n_bins_in=3):
    ser_x, ser_y = ser_con.copy(), ser_cat.copy()
    ser_x = pd.cut(ser_x, n_bins_in).astype('str')
    return corr_cramersv(ser_x, ser_y)


def corr_custom(x_in, y_in, x_group_in, y_group_in):
    if x_group_in == '连续型特征' and y_group_in == '连续型特征':
        return corr_pearson(x_in, y_in)
    elif x_group_in == '连续型特征' and y_group_in == '有序分类':
        return corr_kendalltau(x_in, y_in)
    elif x_group_in == '连续型特征' and y_group_in == '无序分类':
        return corr_cramersv_con_cat(x_in, y_in)
    elif x_group_in == '有序分类' and y_group_in == '连续型特征':
        return corr_kendalltau(x_in, y_in)
    elif x_group_in == '有序分类' and y_group_in == '有序分类':
        return corr_kendalltau(x_in, y_in)
    elif x_group_in == '有序分类' and y_group_in == '无序分类':
        return corr_cramersv(x_in, y_in)
    elif x_group_in == '无序分类' and y_group_in == '连续型特征':
        return corr_cramersv_con_cat(y_in, x_in)
    elif x_group_in == '无序分类' and y_group_in == '有序分类':
        return corr_cramersv(x_in, y_in)
    elif x_group_in == '无序分类' and y_group_in == '无序分类':
        return corr_cramersv(x_in, y_in)
    else:
        pass


# 01 设定变量
# ------------------------------------------------------------------------------------------------------
path_result_train, path_result_predict = './results/result_train.xlsx', './results/result_predict.xlsx'
str_version_date = '2023-08-10'

df_info = pd.DataFrame([
    {
        '特征1类型': '连续型特征',
        '特征2类型': '连续型特征',
        '相关系数类型': 'Pearson相关',
        '取值范围': '[-1, 1]',
        '备注': ''
    },
    {
        '特征1类型': '连续型特征',
        '特征2类型': '有序分类特征',
        '相关系数类型': "Kendall's tau-b相关",
        '取值范围': '[-1, 1]',
        '备注': '连续退化成等级，计算等级相关'
    },
    {
        '特征1类型': '连续型特征',
        '特征2类型': '无序分类特征',
        '相关系数类型': "Cramer's V相关",
        '取值范围': '[0, 1]',
        '备注': '连续退化成无序，计算交叉表相关'
    },

    {
        '特征1类型': '有序分类特征',
        '特征2类型': '有序分类特征',
        '相关系数类型': "Kendall's tau-b相关",
        '取值范围': '[-1, 1]',
        '备注': ''
    },
    {
        '特征1类型': '有序分类特征',
        '特征2类型': '无序分类特征',
        '相关系数类型': "Cramer's V相关",
        '取值范围': '[0, 1]',
        '备注': '有序退化成无序，计算交叉表相关'
    },

    {
        '特征1类型': '无序分类特征',
        '特征2类型': '无序分类特征',
        '相关系数类型': "Cramer's V相关",
        '取值范围': '[0, 1]',
        '备注': ''
    }
])

# 02 读取数据
# ------------------------------------------------------------------------------------------------------
# df_train = pd.read_excel(path_result_train, '01-df_train')
df_importances_origin = pd.read_excel(path_result_train, '03-df_importances_origin')
df_train = pd.read_excel(path_result_predict, '01-df_train')


# 03 数据预处理
# ------------------------------------------------------------------------------------------------------
list_colnames_all = df_importances_origin.features_names.tolist() + ['α参数', 'β参数']
df = df_train.loc[:, list_colnames_all]
df['交通量（自然数）（辆/日）'] = pd.cut(
    df.loc[:, '交通量（自然数）（辆/日）'],
    [-1, 5000, 10000, 20000, 99999],
    labels=[1, 2, 3, 4]
).astype('int')
df.iloc[:, 4] = df.iloc[:, 4].fillna(0)
n_columns = len(df.columns)
group_columns = np.repeat('连续型特征', n_columns)
group_columns[:3] = '无序分类'
group_columns[7] = '有序分类'
del df_importances_origin, df_train, path_result_train, path_result_predict, list_colnames_all


# 04 计算相关矩阵
# ------------------------------------------------------------------------------------------------------
arr_temp = np.zeros((len(group_columns), len(group_columns)))
for i in range(n_columns):
    for j in range(n_columns):
        if i == j:
            arr_temp[i, j] = 1
        else:
            arr_temp[i, j] = corr_custom(df.iloc[:, i], df.iloc[:, j], group_columns[i], group_columns[j])
df_corr = pd.DataFrame(arr_temp, index=df.columns, columns=df.columns)
del arr_temp, i, j


# 05 生成相关热图
# ------------------------------------------------------------------------------------------------------
fig_train_heatmap = px.imshow(
    df_corr, color_continuous_scale='RdYlBu_r', text_auto=True, color_continuous_midpoint=0, aspect='auto',
    title='定制化相关热图  兼容连续型特征、有序分类特征和无序分类特征'
)


# 06 显示
# ------------------------------------------------------------------------------------------------------
# 简介
# ------------------------------------------------------------------------------------------------------
st.markdown(f'## 定制化相关热图结果-{str_version_date}')
st.markdown('---')

# st.markdown('### 一、建模过程简介')
st.markdown('''
&emsp;&emsp;本次主要演示的是，对 **连续型特征**、**有序分类特征** 和 **无序分类特征** 两两定制化相关系数的计算结果。
请尽量确定两点：一是 **这三种特征类型已经包含了所有特征类型**，二是 **所有两两相关系数计算方法跟业务需求相一致**。
''')

st.markdown('''
&emsp;&emsp;本次最主要考虑的是，**这些相关系数计算方法对不同类型特征之间相关性是否能科学呈现**，至于数据取值、数据预处理等，并不是考虑的重点。
举个例子来说，构造物和路面总厚度之间采用了Cramer's V相关的计算方法，那这种相关系数的计算方法，是否能清楚描述出了这两个特征之间的相关，这是重点考虑的事情。
而至于构造物本身这次数据的取值有没有问题，路面总厚度的作为连续型特征进行处理有没有问题，这些目前不需要考虑。
''')

st.markdown('''
&emsp;&emsp;另外，本次开发出了所有特征类型之间的定制化相关系数的计算方法，因此 **其他所有情况都可以看成是此次开发的特例**。
举个例子来说，本次开发涉及到了所有特征类型之间的相关系数计算方法，那么连续型特征对连续型特征的相关系数其实已经包含在里面，
因此以后如果涉及到连续型特征对连续型特征的相关系数计算，那也不会有问题。但是这里有一个前提，就是 **上述第一段需要尽量确定的两点已经得到了确定**。
''')

st.markdown('''
&emsp;&emsp;本次使用5月份的样例数据作为演示，数据请查看 **数据汇总表** 中的 **样例数据**。
样例数据包括12列，分别是构造物、养护措施、结构层类型、通车年限、设计弯沉（0.01mm）、路面总厚度（cm）、沥青层厚度（cm）、
交通量（自然数）（辆/日）、三四五六类车（辆/日）、重车比例（%）、α参数和β参数，
具体明细请查看 **数据汇总表** 中的 **特征明细**。
其中，交通量做了有序变换，数值1、2、3、4分别代表0到5千之间、5千到1万之间、1万到2万之间、2万以上。
''')

st.markdown('''
&emsp;&emsp;在 **数据汇总表** 中的 **相关系数计算明细** 中，详细列出了每两种特征类型的相关系数计算方法，以及其他信息。
另外，在 **数据可视化** 中，也把相关系数矩阵进行了热力图可视化。
''')

df_group = pd.DataFrame({
    '特征': df.columns,
    '类型': group_columns
})


with st.expander('🧾 数据汇总表：样例数据、特征明细和相关系数计算明细'):
    list_tabs = st.tabs(['样例数据', '特征明细', '相关系数计算明细'])
    with list_tabs[0]:
        st.dataframe(df)
    with list_tabs[1]:
        st.dataframe(df_group)
    with list_tabs[2]:
        st.dataframe(df_info)

with st.expander('📊 数据可视化：定制化相关热图'):
    st.plotly_chart(fig_train_heatmap, use_container_width=True)

