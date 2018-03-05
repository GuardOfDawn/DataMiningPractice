import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

# Reading data locally
# corr = df.corr()
# plt.show(sns.heatmap(corr))

# satisfaction_level { <0.11, 0.35<s<0.47, >0.71 } --- 1; other --- 0
# last_evaluation { <0.58, >0.76 } --- 1; other --- 0
# average_montly_hours { <162, >216 } --- 1; other --- 0

# y = df['satisfaction_level']
# plt.plot(y, '.')
# plt.show()

groups = ['low', 'medium', 'high']
lists = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'left', 'promotion_last_5years']
df = pd.read_csv('F:\zhangyi\course\data mining\HR_comma_sep.csv')


def split_continuity(target, data_frame, min_epos=0.05, max_turns=-1):
    target_left = data_frame[[target, 'left', 'sales']].groupby([target, 'left']).count()
    target_count = data_frame[[target, 'left']].groupby(target).count()
    index = []
    for item in target_count.index:
        index.append(item)
    split = [index[0], index[len(index)-1]+1]  # 初始split，规则left<=value<right

    total_count = data_frame[target].count()
    p_left = data_frame.loc[(data_frame['left'] == 1), 'left'].count() / (total_count + 0.0)
    e_before = -p_left*math.log(p_left, 2)-(1-p_left)*math.log((1-p_left), 2)
    print(e_before)
    e_after = -1
    to_continue = True
    add_split = 0.0
    turns = 0
    while to_continue and len(index) > 0:
        e_cal = 0
        for s in index:
            e_cal = 0
            split.append(s)
            split.sort()
            for i in range(0, len(split)-1):
                count = 0
                left = [0, 0]
                for part in target_count.index:
                    if (split[i] <= part) and (part < split[i+1]):
                        count = count + target_count.loc[part, 'left']
                        tmp = target_left.loc[part, 'sales']
                        for k in tmp.index:
                            left[k] = left[k] + tmp[k]
                if count != 0:
                    for j in left:
                        if j != 0:
                            e_cal = e_cal - ((count*1.0)/(total_count*1.0))*((j*1.0)/(count*1.0))*math.log((j*1.0)/(count*1.0), 2)
            if e_after == -1:
                e_after = e_cal
                add_split = s
            elif e_cal < e_after:
                e_after = e_cal
                add_split = s
            split.remove(s)
        if e_before - e_after >= min_epos:
            print('Add split %s' % add_split)
            print('E is %f' % e_after)
            split.append(add_split)
            split.sort()
            index.remove(add_split)
            e_before = e_after
            e_after = -1
            turns = turns + 1
            if max_turns > 0:
                if turns >= max_turns:
                    to_continue = False
        else:
            print('Current split %s' % add_split)
            print('E is %f' % e_after)
            print('Stop')
            to_continue = False
    print('Split of %s' % target)
    print(split)
    return split


split_target = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']
split_lists = []
split_satisfaction = split_continuity('satisfaction_level', df, 0.00, 5)
split_evaluation = split_continuity('last_evaluation', df, 0.02)
split_number_project = split_continuity('number_project', df, 0.05)
split_monthly_hour = split_continuity('average_montly_hours', df, 0.02)
split_time_spend = split_continuity('time_spend_company', df, 0.018)
split_lists.append(split_satisfaction)
split_lists.append(split_evaluation)
split_lists.append(split_number_project)
split_lists.append(split_monthly_hour)
split_lists.append(split_time_spend)

for s in range(0, len(split_target)):
    split = split_lists[s]
    split_attribute = split_target[s]
    for i in range(0, len(split)-1):
        df.loc[((split[i] <= df[split_attribute]) & (df[split_attribute] < split[i + 1])), split_attribute] = split[i]
    for i in range(0, len(split)-1):
        df.loc[(df[split_attribute] == split[i]), split_attribute] = i + 1
df.to_csv('F:\zhangyi\course\data mining\HR_attribute_split.csv', float_format='%.2f', na_rep="NAN!", index=None)

