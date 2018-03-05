import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('F:\zhangyi\course\data mining\HR_attribute_split.csv')

leave_evaluation = df.loc[(df['left'] == 1), ['last_evaluation']]
stay_evaluation = df.loc[(df['left'] == 0), ['last_evaluation']]
fig = plt.figure(figsize=(15, 4),)
ax = sns.distplot(df.loc[(df['left'] == 0), ['last_evaluation']], color='b', label='no left', kde=False)
ax = sns.distplot(df.loc[(df['left'] == 1), ['last_evaluation']], color='r', label='left', kde=False)
ax.set(xlabel='Employee Evaluation', ylabel='Frequency')
plt.title('Employee Evaluation Distribution - left V.S. No left')
plt.show(ax)


# # 以例子
# def entropy_calculation(target_count, target_left):
#     area = df.loc[(df['left'] == 1)]
#     split = [0.09, 0.12, 0.36, 0.47, 2.0]
#     e_cal = 0
#     for i in range(0, len(split)-1):
#         count = 0
#         left = [0, 0]
#         for part in target_count.index:
#             if split[i] <= part < split[i+1]:
#                 count = count + target_count.loc[part, 'left']
#                 tmp = target_left.loc[part, 'sales']
#                 for k in tmp.index:
#                     left[k] = left[k] + tmp[k]
#         if count != 0:
#             for j in left:
#                 if j != 0:
#                     e_cal = e_cal - ((count*1.0)/(total_count*1.0))*((j*1.0)/(count*1.0))*math.log((j*1.0)/(count*1.0), 2)