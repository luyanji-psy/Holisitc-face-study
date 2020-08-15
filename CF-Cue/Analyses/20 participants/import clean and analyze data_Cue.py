# -*- coding: utf-8 -*-
"""
Created on 2/July/19
Updated on 28/Nov/19


@author: Ji

"""

"""
Updated on Sep 16
1. Top/Bottom as within-subject variable, put into the ANOVA
2. Use R packages to do ANOVA, with effect size output

Updated on Nov 28
1. Tradition design, same incongruent trials
2. Plot, same congruent (complementary design) + same incongruent (traditional design)

updated on April 24, 2020
1. Descriptive results
old: 
    np.mean([RT_IAS_T] + [RT_IMS_T]); np.std([RT_IAS_T] + [RT_IMS_T])
    np.mean([RT_IAS_B] + [RT_IMS_B]); np.std([RT_IAS_B] + [RT_IMS_B])

new: 
    a = IAS_IMS_RT_TB.groupby(['Participant','CuedHalf'])['reactionTime'].describe()    
    a.groupby(['CuedHalf'])['mean'].describe() 

"""


"""
Read several excel files from a directory into pandas and
concatenate them into one big Dataframe.

"""

import glob
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy import stats
from statsmodels.stats.anova import AnovaRM
Z = norm.ppf

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import r, pandas2ri

afex = rpackages.importr('afex')
emmeans = rpackages.importr('emmeans', robject_translations = {"recover.data.call": "recover_data_call1"})

import seaborn as sns

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 24

raw_path = r'D:\Post-doc HKU\Haiyang\Cue_Complete\CF_Complete_Cue\Excel Data'
path = r'D:\Post-doc HKU\Haiyang\Cue_Complete\CF_Complete_Cue\Analysis'

# change the working folder to path
os.chdir(raw_path)
allFiles = glob.glob(raw_path + "/*.xlsx")
Data = pd.concat((pd.read_excel(f) for f in allFiles), sort = False)

os.chdir(path)


"""
Top, bottom, separately

"""
Data.groupby(['CuedHalf','Congruency', 'Alignment', 'SameDifferent'])['isCorrect'].mean()  

Data_CA_T = Data[(Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'A')]
Data_CM_T = Data[(Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'M')]
Data_IA_T = Data[(Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'A')]
Data_IM_T = Data[(Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'M')]

Data_CA_B = Data[(Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'A')]
Data_CM_B = Data[(Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'M')]
Data_IA_B = Data[(Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'A')]
Data_IM_B = Data[(Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'M')]

acc_CA_T = Data_CA_T.groupby(['Participant'])['isCorrect'].mean()
acc_CM_T = Data_CM_T.groupby(['Participant'])['isCorrect'].mean()
acc_IA_T = Data_IA_T.groupby(['Participant'])['isCorrect'].mean()
acc_IM_T = Data_IM_T.groupby(['Participant'])['isCorrect'].mean()

acc_CA_B = Data_CA_B.groupby(['Participant'])['isCorrect'].mean()
acc_CM_B = Data_CM_B.groupby(['Participant'])['isCorrect'].mean()
acc_IA_B = Data_IA_B.groupby(['Participant'])['isCorrect'].mean()
acc_IM_B = Data_IM_B.groupby(['Participant'])['isCorrect'].mean()

means_acc_C_T = [np.mean(acc_CA_T), np.mean(acc_CM_T)]; errors_C_T = [stats.sem(acc_CA_T),stats.sem(acc_CM_T)]
means_acc_I_T = [np.mean(acc_IA_T), np.mean(acc_IM_T)]; errors_I_T = [stats.sem(acc_IA_T),stats.sem(acc_IM_T)]

means_acc_C_B = [np.mean(acc_CA_B), np.mean(acc_CM_B)]; errors_C_B = [stats.sem(acc_CA_B),stats.sem(acc_CM_B)]
means_acc_I_B = [np.mean(acc_IA_B), np.mean(acc_IM_B)]; errors_I_B = [stats.sem(acc_IA_B),stats.sem(acc_IM_B)]

# barplot
index = np.arange(2)
bar_width = 0.35
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
rects1 = ax.bar(index,means_acc_C_T, bar_width, yerr = errors_C_T, label = 'Congruent')
rects1 = ax.bar(index + bar_width,means_acc_I_T, bar_width, yerr = errors_I_T, label = 'Incongruent')
ax.set_ylim((0,1)) 
ax.set_ylabel('Acurracy')
ax.set_title('Top: Congruency by Alignment', fontweight="bold", fontsize=24)
ax.set_xticks(index + bar_width/ 2)
ax.set_xticklabels(('Aligned', 'Misaligned'))
ax.legend()
plt.savefig('Congruency by alignment_acc_cueT.tiff')
plt.show()

index = np.arange(2)
bar_width = 0.35
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
rects1 = ax.bar(index,means_acc_C_B, bar_width, yerr = errors_C_B, label = 'Congruent')
rects1 = ax.bar(index + bar_width,means_acc_I_B, bar_width, yerr = errors_I_B, label = 'Incongruent')
ax.set_ylim((0,1)) 
ax.set_ylabel('Acurracy')
ax.set_title('Bottom: Congruency by Alignment', fontweight="bold", fontsize=24)
ax.set_xticks(index + bar_width/ 2)
ax.set_xticklabels(('Aligned', 'Misaligned'))
ax.legend()
plt.savefig('Congruency by alignment_acc_cueB.tiff')
plt.show()

aovrm2way_acc_TB = AnovaRM(Data, 'isCorrect', 'Participant', within = ['CuedHalf','Congruency', 'Alignment'], aggregate_func = 'mean')
res2way_acc_TB = aovrm2way_acc_TB.fit()
print (res2way_acc_TB) # Effect of Top/Bottom, main effect of congruency, interaction between congruency and alignment, and interaction between cued half and alignment，
                       # the three way interaction is also significant

DataT = Data[(Data['CuedHalf'] == 'T')]
DataB = Data[(Data['CuedHalf'] == 'B')]

aovrm2way_accT = AnovaRM(DataT, 'isCorrect', 'Participant', within = ['Congruency', 'Alignment'], aggregate_func = 'mean')
res2way_accT = aovrm2way_accT.fit()
print (res2way_accT)

aovrm2way_accB = AnovaRM(DataB, 'isCorrect', 'Participant', within = ['Congruency', 'Alignment'], aggregate_func = 'mean')
res2way_accB = aovrm2way_accB.fit()
print (res2way_accB)

"""
Traditional design: same incongruent

"""
## t test: same incongruent aligned - same incongruent misaligned
# acc
Data_IAS_T = Data[(Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'A') & (Data['SameDifferent'] == 'S')]
Data_IMS_T = Data[(Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'M') & (Data['SameDifferent'] == 'S')]
acc_IAS_T = Data_IAS_T.groupby(['Participant'])['isCorrect'].mean()
acc_IMS_T = Data_IMS_T.groupby(['Participant'])['isCorrect'].mean()
t, p = stats.ttest_rel(acc_IAS_T, acc_IMS_T)
print(t,p)  # top: significant difference between IAS and IMS, acc(IAS) < acc(IMS), p < .001

Data_IAS_B = Data[(Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'A') & (Data['SameDifferent'] == 'S')]
Data_IMS_B = Data[(Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'M') & (Data['SameDifferent'] == 'S')]
acc_IAS_B = Data_IAS_B.groupby(['Participant'])['isCorrect'].mean()
acc_IMS_B = Data_IMS_B.groupby(['Participant'])['isCorrect'].mean()
t, p = stats.ttest_rel(acc_IAS_B, acc_IMS_B)
print(t,p)  # bottom: no difference between IAS and IMS, p = .896

IAS_IMS_acc_TB = pd.DataFrame(data = acc_IAS_T)
IAS_IMS_acc_TB = IAS_IMS_acc_TB.append(pd.DataFrame(data = acc_IMS_T))
IAS_IMS_acc_TB = IAS_IMS_acc_TB.append(pd.DataFrame(data = acc_IAS_B))
IAS_IMS_acc_TB = IAS_IMS_acc_TB.append(pd.DataFrame(data = acc_IMS_B))

IAS_IMS_acc_TB['CuedHalf'] = ['Top']*40 + ['Bottom']*40
IAS_IMS_acc_TB['Alignment'] = ['Aligned']*20 + ['MisAligned']*20 + ['Aligned']*20 + ['MisAligned']*20
IAS_IMS_acc_TB['participant'] = list(range(1,21)) * 4

# ANOVA I
aovrm2way_acc_IS_TB = AnovaRM(IAS_IMS_acc_TB, 'isCorrect', 'participant', within = ['CuedHalf','Alignment'], aggregate_func = 'mean')
res2way_acc_IS_TB = aovrm2way_acc_IS_TB.fit()
print (res2way_acc_IS_TB)
# main effect of Cuedhalf, F(1,19) <1, p = .34; main effect of alignment, F(1,19) = 12.13, p = .0025  
# Interaction between Cuedhalf and alignment, F(1,19) = 19.23, p = .0003

# ANOVA II
pandas2ri.activate()
r_IAS_IMAS_acc = pandas2ri.py2ri(IAS_IMS_acc_TB)

model = afex.aov_ez('participant', 'isCorrect', r_IAS_IMAS_acc, within = ['CuedHalf', 'Alignment'])
print(model)

# descriptive
a = IAS_IMS_acc_TB.groupby(['Participant','CuedHalf','Alignment'])['isCorrect'].describe()    
a.groupby(['CuedHalf','Alignment'])['mean'].describe() 

# Plot
means_acc_IAS_TB = [np.mean(acc_IAS_T), np.mean(acc_IAS_B)]; errors_IAS_TB = [stats.sem(acc_IAS_T),stats.sem(acc_IAS_B)]
means_acc_IMS_TB = [np.mean(acc_IMS_T), np.mean(acc_IMS_B)]; errors_IMS_TB = [stats.sem(acc_IMS_T),stats.sem(acc_IMS_B)]

index = np.arange(2)
bar_width = 0.35
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
rects1 = ax.bar(index,means_acc_IAS_TB, bar_width, yerr = errors_IAS_TB, label = 'Aligned')
rects1 = ax.bar(index + bar_width,means_acc_IMS_TB, bar_width, yerr = errors_IMS_TB, label = 'Misaligned')
ax.set_ylabel('Acurracy')
ax.set_ylim((0,1))
ax.set_title('Incongruent Same Conditions', fontweight="bold", fontsize=24)
ax.set_xticks(index + bar_width/ 2)
ax.set_xticklabels(('Top', 'Bottom'))
ax.legend()
plt.savefig('acc_SameIncongruent a vs. m_cueTB.tiff')
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(12,10),dpi=100)
plt.title('Top&Bottom, Same Incongruent, 20 participants', fontweight="bold", fontsize=24)
plt.ylabel('Accuracy')
plt.ylim(0,1)
sns.boxplot(data=[acc_IAS_T, acc_IMS_T, acc_IAS_B, acc_IMS_B], palette = "Blues", showmeans=True)
sns.swarmplot(data=[acc_IAS_T, acc_IMS_T, acc_IAS_B, acc_IMS_B], color = ".25")
sns.despine()
plt.xticks([0,1,2,3], ['Aligned Top','Misaligned Top','Aligned Bottom','Misaligned Bottom'])
plt.savefig('acc_SameIncongruent a vs. m_boxplot_cueTB.tiff')
plt.show()


"""
Complementary design (part-whole like): same congruent

"""
## t test: same congruent aligned - same congruent misaligned
Data_CAS_T = Data[(Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'A') & (Data['SameDifferent'] == 'S')]
Data_CMS_T = Data[(Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'M') & (Data['SameDifferent'] == 'S')]
acc_CAS_T = Data_CAS_T.groupby(['Participant'])['isCorrect'].mean()
acc_CMS_T = Data_CMS_T.groupby(['Participant'])['isCorrect'].mean()
t, p = stats.ttest_rel(acc_CAS_T, acc_CMS_T)
print(t,p)  # top: no difference between CAS and CMS, p = .546

Data_CAS_B = Data[(Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'A') & (Data['SameDifferent'] == 'S')]
Data_CMS_B = Data[(Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'M') & (Data['SameDifferent'] == 'S')]
acc_CAS_B = Data_CAS_B.groupby(['Participant'])['isCorrect'].mean()
acc_CMS_B = Data_CMS_B.groupby(['Participant'])['isCorrect'].mean()
t, p = stats.ttest_rel(acc_CAS_B, acc_CMS_B)
print(t,p)  # bottom: significant difference between CAS and CMS, t = 3.251, p = .0042

CAS_CMS_acc_TB = pd.DataFrame(data = acc_CAS_T)
CAS_CMS_acc_TB = CAS_CMS_acc_TB.append(pd.DataFrame(data = acc_CMS_T))
CAS_CMS_acc_TB = CAS_CMS_acc_TB.append(pd.DataFrame(data = acc_CAS_B))
CAS_CMS_acc_TB = CAS_CMS_acc_TB.append(pd.DataFrame(data = acc_CMS_B))

CAS_CMS_acc_TB['CuedHalf'] = ['Top']*40 + ['Bottom']*40
CAS_CMS_acc_TB['Alignment'] = ['Aligned']*20 + ['MisAligned']*20 + ['Aligned']*20 + ['MisAligned']*20
CAS_CMS_acc_TB['participant'] = list(range(1,21)) * 4

# ANOVA I
aovrm2way_acc_CS_TB = AnovaRM(CAS_CMS_acc_TB, 'isCorrect', 'participant', within = ['CuedHalf','Alignment'], aggregate_func = 'mean')
res2way_acc_CS_TB = aovrm2way_acc_CS_TB.fit()
print (res2way_acc_CS_TB)
# main effect of Cuedhalf, F(1,19) = 5.098, p = .0359; main effect of alignment, F(1,19) = 8.438, p = .0091  
# Interaction between Cuedhalf and alignment, F(1,19) = 4.718, p = .0427

# ANOVA II
pandas2ri.activate()
r_CAS_CMS_acc_TB = pandas2ri.py2ri(CAS_CMS_acc_TB)

model = afex.aov_ez('participant', 'isCorrect', CAS_CMS_acc_TB, within = ['CuedHalf', 'Alignment'])
print(model)


# descriptive
a = CAS_CMS_acc_TB.groupby(['Participant','CuedHalf','Alignment',])['isCorrect'].describe()    
a.groupby(['CuedHalf','Alignment'])['mean'].mean() 
a.groupby(['CuedHalf','Alignment'])['mean'].std() 

# plot
means_acc_CAS_TB = [np.mean(acc_CAS_T), np.mean(acc_CAS_B)]; errors_CAS_TB = [stats.sem(acc_CAS_T),stats.sem(acc_CAS_B)]
means_acc_CMS_TB = [np.mean(acc_CMS_T), np.mean(acc_CMS_B)]; errors_CMS_TB = [stats.sem(acc_CMS_T),stats.sem(acc_CMS_B)]

index = np.arange(2)
bar_width = 0.35
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
rects1 = ax.bar(index,means_acc_CAS_TB, bar_width, yerr = errors_CAS_TB, label = 'Aligned')
rects1 = ax.bar(index + bar_width,means_acc_CMS_TB, bar_width, yerr = errors_CMS_TB, label = 'Misaligned')
ax.set_ylabel('Acurracy')
ax.set_ylim((0,1))
ax.set_title('Congruent Same Conditions', fontweight="bold", fontsize=24)
ax.set_xticks(index + bar_width/ 2)
ax.set_xticklabels(('Top', 'Bottom'))
ax.legend()
plt.savefig('acc_SameCongruent a vs. m_cueTB.tiff')
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(12,10),dpi=100)
plt.title('Top&Bottom, Same Congruent, 20 participants', fontweight="bold", fontsize=24)
plt.ylabel('Accuracy')
plt.ylim(0,1)
sns.boxplot(data=[acc_CAS_T, acc_CMS_T, acc_CAS_B, acc_CMS_B], palette = "Blues", showmeans=True)
sns.swarmplot(data=[acc_CAS_T, acc_CMS_T, acc_CAS_B, acc_CMS_B], color = ".25")
sns.despine()
plt.xticks([0,1,2,3], ['Aligned Top','Misaligned Top','Aligned Bottom','Misaligned Bottom'])
plt.savefig('acc_SameCongruent a vs. m_boxplot_cueTB.tiff')
plt.show()

"""
plot: traditional design + part-whole (complementary) design
same incongruent condition + same congruent condition
Two conditions in one plot

"""

barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Top','Bottom','Top','Bottom']
values1 = [np.mean(acc_CAS_T), np.mean(acc_CAS_B), np.mean(acc_IAS_T), np.mean(acc_IAS_B)]
errors1 = [stats.sem(acc_CAS_T),stats.sem(acc_CAS_B), stats.sem(acc_IAS_T),stats.sem(acc_IAS_B)]
values2 = [np.mean(acc_CMS_T), np.mean(acc_CMS_B), np.mean(acc_IMS_T), np.mean(acc_IMS_B)]
errors2 = [stats.sem(acc_CMS_T),stats.sem(acc_CMS_B), stats.sem(acc_IMS_T),stats.sem(acc_IMS_B)]

fig,ax = plt.subplots(figsize=(8,6),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, color='lightgrey', edgecolor='white', label = 'Aligned')
plt.bar(r2, values2, yerr=errors2, width=barWidth, color='black', edgecolor='white', label = 'Misaligned')
plt.ylim((0,1))
plt.ylabel('Accuracy')
plt.xticks([(r + barWidth/2) for r in range(4)], names)
plt.legend(loc = "upper right", bbox_to_anchor=(1.08, 1.09), labelspacing=0.1)  # bbox_to_anchor indicates the position of the right corner of the legend relative to the whole box
plt.text(0.75, 1.1, "Same congruent", size=20,
         ha="center", va="center")
plt.text(2.75, 1.1, "Same incongruent", size=20,
         ha="center", va="center")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('Traditional + Complementary_acc.tiff')
plt.show()

# color plot
barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Same Congruent','Same Incongruent','Same Congruent','Same Incongruent']
values1 = [np.mean(acc_CAS_T), np.mean(acc_IAS_T), np.mean(acc_CAS_B), np.mean(acc_IAS_B)]
errors1 = [stats.sem(acc_CAS_T),stats.sem(acc_IAS_T), stats.sem(acc_CAS_B),stats.sem(acc_IAS_B)]
values2 = [np.mean(acc_CMS_T), np.mean(acc_IMS_T), np.mean(acc_CMS_B), np.mean(acc_IMS_B)]
errors2 = [stats.sem(acc_CMS_T),stats.sem(acc_IMS_T), stats.sem(acc_CMS_B),stats.sem(acc_IMS_B)]

fig,ax = plt.subplots(figsize=(12,10),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, label = 'Aligned')
plt.bar(r2, values2, yerr=errors2, width=barWidth, label = 'Misaligned')
plt.ylim((0,1))
plt.ylabel('Accuracy')
ax.set_title('Experiment 2', fontweight="bold", fontsize=24)
plt.xticks([(r + barWidth/2) for r in range(4)], names)
ax.legend(bbox_to_anchor=[0.5, 0.5, 0, 0.45], loc='upper center', ncol=2)
plt.text(0.75, 0.97, "Top", size=20,
         ha="center", va="center")
plt.text(2.75, 0.97, "Bottom", size=20,
         ha="center", va="center")
plt.savefig('Traditional + Complementary_acc_color.tiff')
plt.show()

# color plot2
fig,ax = plt.subplots(figsize=(8,6))
plt.bar(r1, values1, yerr=errors1, width=barWidth, label = 'Aligned')
plt.bar(r2, values2, yerr=errors2, width=barWidth, label = 'Misaligned')
plt.ylim((0,1.1))
plt.ylabel('Accuracy')
ax.set_title('Experiment 2', fontweight="bold", fontsize=24)
plt.xticks([(r + barWidth/2) for r in range(4)], names)
ax.tick_params(axis='x', which='major', labelsize=13)
ax.legend(bbox_to_anchor=[0.5, 0.5, 0, 0.45], loc='upper center', ncol=2, prop=dict(size=18))
plt.text(0.75, 1.06, "Top", size=20,
         ha="center", va="center")
plt.text(2.75, 1.06, "Bottom", size=20,
         ha="center", va="center")
plt.savefig('Traditional + Complementary_acc_color new.tiff')
plt.show()


"""
d prime
Top, bottom, separately

Adjust the extreme values.
Rates of 0 are replaced with 0.5/n, and rates of 1 are replaced with (n-0.5)/n,
where n is the number of signal or noise trials (Macmillan & Kaplan, 1985, see Stanisla & Todorov, 1999)

n = 40 in the present study 

"""
n = 40
def SDT (hits, fas):
    for i in range(20):
        if hits[i] == 1:
            hits[i] = (n-0.5)/n
        if hits[i] == 0:
            hits[i] = 0.5/n
        if fas[i] == 1:
            fas[i] = (n-0.5)/n
        if fas[i] == 0:
            fas[i] = 0.5/n
            
    d = Z(hits) - Z(fas)
    
    return(d)

# top
hit_CA_T = Data_CA_T[(Data_CA_T['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_CA_T = 1 - Data_CA_T[(Data_CA_T['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
#d_CA_T = Z(hit_CA_T) - Z(fa_CA_T)
d_CA_T = SDT(hit_CA_T.tolist(),fa_CA_T.tolist())

hit_CM_T = Data_CM_T[(Data_CM_T['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_CM_T = 1- Data_CM_T[(Data_CM_T['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_CM_T = SDT(hit_CM_T.tolist(),fa_CM_T.tolist())

hit_IA_T = Data_IA_T[(Data_IA_T['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_IA_T = 1 - Data_IA_T[(Data_IA_T['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_IA_T = SDT(hit_IA_T.tolist(),fa_IA_T.tolist())

hit_IM_T = Data_IM_T[(Data_IM_T['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_IM_T = 1 - Data_IM_T[(Data_IM_T['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_IM_T = SDT(hit_IM_T.tolist(),fa_IM_T.tolist())

# bottom
hit_CA_B = Data_CA_B[(Data_CA_B['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean() 
fa_CA_B = 1 - Data_CA_B[(Data_CA_B['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_CA_B = SDT(hit_CA_B.tolist(),fa_CA_B.tolist())

hit_CM_B = Data_CM_B[(Data_CM_B['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_CM_B = 1- Data_CM_B[(Data_CM_B['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_CM_B = SDT(hit_CM_B.tolist(),fa_CM_B.tolist())

hit_IA_B = Data_IA_B[(Data_IA_B['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_IA_B = 1 - Data_IA_B[(Data_IA_B['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_IA_B = SDT(hit_IA_B.tolist(),fa_IA_B.tolist())

hit_IM_B = Data_IM_B[(Data_IM_B['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_IM_B = 1 - Data_IM_B[(Data_IM_B['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_IM_B = SDT(hit_IM_B.tolist(),fa_IM_B.tolist())

## one-sample t test, whether d prime is different from zero
# in all conditions, the d prime was significant larger than zero, except p8!!!
t1, p1 = stats.ttest_1samp(d_CA_T,0.0)
t2, p2 = stats.ttest_1samp(d_CM_T,0.0) 
t3, p3 = stats.ttest_1samp(d_IA_T,0.0)
t4, p4 = stats.ttest_1samp(d_IM_T,0.0)
t5, p5 = stats.ttest_1samp(d_CA_B,0.0)
t6, p6 = stats.ttest_1samp(d_CM_B,0.0) 
t7, p7 = stats.ttest_1samp(d_IA_B,0.0)
t8, p8 = stats.ttest_1samp(d_IM_B,0.0) # incongruent misaligned bottom, t8 = 1.561, p8 = .135

dprime = pd.DataFrame(
    {'Congruent_Aligned_Top': d_CA_T,
     'Congruent_Misligned_Top': d_CM_T,
     'Incongruent_Aligned_Top': d_IA_T,
     'Incongruent_Misligned_Top': d_IM_T,
     'Congruent_Aligned_Bottom': d_CA_B,
     'Congruent_Misligned_Bottom': d_CM_B,
     'Incongruent_Aligned_Bottom': d_IA_B,
     'Incongruent_Misligned_Bottom': d_IM_B,
    })

dprime_long = pd.melt(dprime, value_name='d')
dprime_long['CuedHalf'] = ['Top']*80 + ['Bottom']*80
dprime_long['Alignment'] = (['Aligned']*20 + ['MisAligned']*20) * 4 
dprime_long['Congruency'] = (['Congruent']*40 + ['Incongruent']*40) * 2
dprime_long['participant'] = list(range(1,21)) * 8

# I
aovrm2way_dprime = AnovaRM(dprime_long, 'd', 'participant', within = ['CuedHalf','Alignment', 'Congruency'], aggregate_func = 'mean')
res2way_dprime = aovrm2way_dprime.fit()
print (res2way_dprime) # Effect of Top/Bottom, main effect of congruency, interaction between congruency and alignment, and interaction between cued half and alignment，
                       # the three way interaction is also significant

# II
pandas2ri.activate()
r_dprime_long = pandas2ri.py2ri(dprime_long)
model = afex.aov_ez('participant', 'd', r_dprime_long, within = ['CuedHalf','Alignment', 'Congruency'])
print(model)

dprimeT = dprime_long[(dprime_long['CuedHalf'] == 'Top')]
dprimeB = dprime_long[(dprime_long['CuedHalf'] == 'Bottom')]

aovrm2way_dprimeT = AnovaRM(dprimeT, 'd', 'participant', within = ['Congruency', 'Alignment'], aggregate_func = 'mean')
res2way_dprimeT = aovrm2way_dprimeT.fit()
print (res2way_dprimeT)

aovrm2way_dprimeB = AnovaRM(dprimeB, 'd', 'participant', within = ['Congruency', 'Alignment'], aggregate_func = 'mean')
res2way_dprimeB = aovrm2way_dprimeB.fit()
print (res2way_dprimeB)

#plot
means_C_T = [np.mean(d_CA_T), np.mean(d_CM_T)]; errors_C_T = [stats.sem(d_CA_T),stats.sem(d_CM_T)]
means_I_T = [np.mean(d_IA_T), np.mean(d_IM_T)]; errors_I_T = [stats.sem(d_IA_T),stats.sem(d_IM_T)]

index = np.arange(2)
bar_width = 0.35
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
rects1 = ax.bar(index,means_C_T, bar_width, yerr = errors_C_T, label = 'Congruent')
rects1 = ax.bar(index + bar_width,means_I_T, bar_width, yerr = errors_I_T, label = 'Incongruent')
ax.set_ylabel('d prime')
plt.ylim(0,2.8)
ax.set_title('Top: Congruency by Alignment', fontweight="bold", fontsize=24)
ax.set_xticks(index + bar_width/ 2)
ax.set_xticklabels(('Aligned', 'Misaligned'))
ax.legend()
plt.savefig('d_cueT.tiff')
plt.show()

means_C_B = [np.mean(d_CA_B), np.mean(d_CM_B)]; errors_C_B = [stats.sem(d_CA_B),stats.sem(d_CM_B)]
means_I_B = [np.mean(d_IA_B), np.mean(d_IM_B)]; errors_I_B = [stats.sem(d_IA_B),stats.sem(d_IM_B)]

index = np.arange(2)
bar_width = 0.35
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
rects1 = ax.bar(index,means_C_B, bar_width, yerr = errors_C_B, label = 'Congruent')
rects1 = ax.bar(index + bar_width,means_I_B, bar_width, yerr = errors_I_B, label = 'Incongruent')
ax.set_ylabel('d prime')
plt.ylim(0,2.8)
ax.set_title('Bottom: Congruency by Alignment', fontweight="bold", fontsize=24)
ax.set_xticks(index + bar_width/ 2)
ax.set_xticklabels(('Aligned', 'Misaligned'))
ax.legend()
plt.savefig('d_cueB.tiff')
plt.show()


# plot
###########
# Two ratio conditions in one plot, using subplot function
# Version II
barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Aligned','Misaligned','Aligned','Misaligned']
values1 = [np.mean(d_CA_T), np.mean(d_CM_T), np.mean(d_CA_B), np.mean(d_CM_B)]
errors1 = [stats.sem(d_CA_T),stats.sem(d_CM_T), stats.sem(d_CA_B),stats.sem(d_CM_B)]
values2 = [np.mean(d_IA_T), np.mean(d_IM_T), np.mean(d_IA_B), np.mean(d_IM_B)]
errors2 = [stats.sem(d_IA_T),stats.sem(d_IM_T), stats.sem(d_IA_B),stats.sem(d_IM_B)]

fig,ax = plt.subplots(figsize=(8,6),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, color='lightgrey', edgecolor='white', label = 'Congruent')
plt.bar(r2, values2, yerr=errors2, width=barWidth, color='black', edgecolor='white', label = 'Incongruent')
plt.ylim((0,3.5))
plt.ylabel('d prime')
plt.xticks([(r + barWidth/2) for r in range(4)], names)
plt.legend(loc = "upper right", bbox_to_anchor=(1.08, 1.05), labelspacing=0.1)  # bbox_to_anchor indicates the position of the right corner of the legend relative to the whole box
plt.text(0.75, 3.75, "Top", size=20,
         ha="center", va="center")
plt.text(2.75, 3.75, "Bottom", size=20,
         ha="center", va="center")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('Congruency by alignment_dprimev2.tiff')
plt.show()

# color graph
fig,ax = plt.subplots(figsize=(12,10),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, label = 'Congruent')
plt.bar(r2, values2, yerr=errors2, width=barWidth, label = 'Incongruent')
plt.ylim((0,3.5))
plt.ylabel('d prime')
ax.set_title('Experiment 2', fontweight="bold", fontsize=24)
plt.xticks([(r + barWidth/2) for r in range(4)], names)
ax.legend(bbox_to_anchor=[0.5, 0.5, 0, 0.45], loc='upper center', ncol=2)
plt.text(0.75, 3.35, "Top", size=20,
         ha="center", va="center")
plt.text(2.75, 3.35, "Bottom", size=20,
         ha="center", va="center")
plt.savefig('Congruency by alignment_dprimev2_color.tiff')
plt.show()

# color graph2
barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Congruent','Incongruent','Congruent','Incongruent']

values1 = [np.mean(d_CA_T), np.mean(d_IA_T), np.mean(d_CA_B), np.mean(d_IA_B)]
errors1 = [stats.sem(d_CA_T),stats.sem(d_IA_T), stats.sem(d_CA_B),stats.sem(d_IA_B)]
values2 = [np.mean(d_CM_T), np.mean(d_IM_T), np.mean(d_CM_B), np.mean(d_IM_B)]
errors2 = [stats.sem(d_CM_T),stats.sem(d_IM_T), stats.sem(d_CM_B),stats.sem(d_IM_B)]

fig,ax = plt.subplots(figsize=(8,6))
plt.bar(r1, values1, yerr=errors1, width=barWidth, label = 'Aligned')
plt.bar(r2, values2, yerr=errors2, width=barWidth, label = 'Misaligned')
plt.ylim((0,3.5))
plt.ylabel('d prime')
ax.set_title('Experiment 2', fontweight="bold", fontsize=24)
plt.xticks([(r + barWidth/2) for r in range(4)], names)
ax.tick_params(axis='x', which='major', labelsize=18)
ax.legend(bbox_to_anchor=[0.5, 0.5, 0, 0.45], loc='upper center', ncol=2, prop=dict(size=18))
plt.text(0.75, 3.35, "Top", size=20,
         ha="center", va="center")
plt.text(2.75, 3.35, "Bottom", size=20,
         ha="center", va="center")
plt.savefig('Congruency by alignment_dprimev2_color new.tiff')
plt.show()



"""
RT
"""
# Correct RT
# Top

DataT = Data[(Data['CuedHalf'] == 'T')]
DataT_cor = DataT[(DataT['isCorrect'] == 1)]

DataT_cor.groupby(['Congruency', 'Alignment'])['reactionTime'].mean()  

DataTcor_CA = DataT_cor[(DataT_cor['Congruency'] == 'C') & (DataT_cor['Alignment'] == 'A')]
DataTcor_CM = DataT_cor[(DataT_cor['Congruency'] == 'C') & (DataT_cor['Alignment'] == 'M')]
DataTcor_IA = DataT_cor[(DataT_cor['Congruency'] == 'I') & (DataT_cor['Alignment'] == 'A')]
DataTcor_IM = DataT_cor[(DataT_cor['Congruency'] == 'I') & (DataT_cor['Alignment'] == 'M')]

RTT_CA = DataTcor_CA.groupby(['Participant'])['reactionTime'].mean()
RTT_CM = DataTcor_CM.groupby(['Participant'])['reactionTime'].mean()
RTT_IA = DataTcor_IA.groupby(['Participant'])['reactionTime'].mean()
RTT_IM = DataTcor_IM.groupby(['Participant'])['reactionTime'].mean()

# Bottom
DataB = Data[(Data['CuedHalf'] == 'B')]
DataB_cor = DataB[(DataB['isCorrect'] == 1)]

DataB_cor.groupby(['Congruency', 'Alignment'])['reactionTime'].mean()  

DataBcor_CA = DataB_cor[(DataB_cor['Congruency'] == 'C') & (DataB_cor['Alignment'] == 'A')]
DataBcor_CM = DataB_cor[(DataB_cor['Congruency'] == 'C') & (DataB_cor['Alignment'] == 'M')]
DataBcor_IA = DataB_cor[(DataB_cor['Congruency'] == 'I') & (DataB_cor['Alignment'] == 'A')]
DataBcor_IM = DataB_cor[(DataB_cor['Congruency'] == 'I') & (DataB_cor['Alignment'] == 'M')]

RTB_CA = DataBcor_CA.groupby(['Participant'])['reactionTime'].mean()
RTB_CM = DataBcor_CM.groupby(['Participant'])['reactionTime'].mean()
RTB_IA = DataBcor_IA.groupby(['Participant'])['reactionTime'].mean()
RTB_IM = DataBcor_IM.groupby(['Participant'])['reactionTime'].mean()

# plot
###########
# Two cuedhalf conditions in one plot, using subplot function
# Version II
barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Aligned','Misaligned','Aligned','Misaligned']
values1 = [np.mean(RTT_CA) * 1000, np.mean(RTT_CM) * 1000, np.mean(RTB_CA) * 1000, np.mean(RTB_CM) * 1000]  # times 1000 to convert s to ms
errors1 = [stats.sem(RTT_CA) * 1000,stats.sem(RTT_CM) * 1000, stats.sem(RTB_CA) * 1000,stats.sem(RTB_CM) * 1000]
values2 = [np.mean(RTT_IA) * 1000, np.mean(RTT_IM) * 1000, np.mean(RTB_IA) * 1000, np.mean(RTB_IM) * 1000]
errors2 = [stats.sem(RTT_IA) * 1000,stats.sem(RTT_IM) * 1000, stats.sem(RTB_IA) * 1000,stats.sem(RTB_IM) * 1000]

fig,ax = plt.subplots(figsize=(8,6),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, color='lightgrey', edgecolor='white', label = 'Congruent')
plt.bar(r2, values2, yerr=errors2, width=barWidth, color='black', edgecolor='white', label = 'Incongruent')
plt.ylim((0,1000))
plt.ylabel('RT (ms)')
plt.xticks([(r + barWidth/2) for r in range(4)], names)
plt.legend(loc = "upper right", bbox_to_anchor=(1.08, 1.05), labelspacing=0.1)  # bbox_to_anchor indicates the position of the right corner of the legend relative to the whole box
plt.text(0.75, 1057, "Top", size=20,
         ha="center", va="center")
plt.text(2.75, 1057, "Bottom", size=20,
         ha="center", va="center")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('Congruency by alignment_RTv2.tiff')
plt.show()


# ANOVA
corrRT = pd.DataFrame(
    {'Congruent_Aligned_Top': RTT_CA,
     'Congruent_Misligned_Top': RTT_CM,
     'Incongruent_Aligned_Top': RTT_IA,
     'Incongruent_Misligned_Top': RTT_IM,
     'Congruent_Aligned_Bottom': RTB_CA,
     'Congruent_Misligned_Bottom': RTB_CM,
     'Incongruent_Aligned_Bottom': RTB_IA,
     'Incongruent_Misligned_Bottom': RTB_IM,
    })

corrRT_long = pd.melt(corrRT, value_name='corrRT')
corrRT_long['CuedHalf'] = ['Top']*80 + ['Bottom']*80
corrRT_long['Alignment'] = (['Aligned']*20 + ['MisAligned']*20) * 4 
corrRT_long['Congruency'] = (['Congruent']*40 + ['Incongruent']*40) * 2
corrRT_long['Participant'] = list(range(1,21)) * 8

# I
pandas2ri.activate()
r_corrRT_long = pandas2ri.py2ri(corrRT_long)

model = afex.aov_ez('Participant', 'corrRT', r_corrRT_long, within = ['CuedHalf', 'Alignment','Congruency'])
print(model)

# II
aovrm2way_corrRT = AnovaRM(corrRT_long, 'corrRT', 'Participant', within = ['CuedHalf','Alignment', 'Congruency'], aggregate_func = 'mean')
res2way_corrRT = aovrm2way_corrRT.fit()
print (res2way_corrRT)

# descriptive
a = corrRT_long.groupby(['Participant','Alignment'])['corrRT'].describe()    
a.groupby(['Alignment'])['mean'].describe() 

a = corrRT_long.groupby(['Participant','Congruency'])['corrRT'].describe()    
a.groupby(['Congruency'])['mean'].describe() 

a = corrRT_long.groupby(['Participant','CuedHalf'])['corrRT'].describe()    
a.groupby(['CuedHalf'])['mean'].describe() 


"""
Traditional design: same incongruent

"""

# RT
RT_IAS_T = Data_IAS_T.groupby(['Participant'])['reactionTime'].mean()
RT_IMS_T = Data_IMS_T.groupby(['Participant'])['reactionTime'].mean()
t, p = stats.ttest_rel(RT_IAS_T, RT_IMS_T)
print(t,p)  # top: no significant difference between IAS and IMS

RT_IAS_B = Data_IAS_B.groupby(['Participant'])['reactionTime'].mean()
RT_IMS_B = Data_IMS_B.groupby(['Participant'])['reactionTime'].mean()
t, p = stats.ttest_rel(RT_IAS_B, RT_IMS_B)
print(t,p)  # bottom: no difference between IAS and IMS

IAS_IMS_RT_TB = pd.DataFrame(data = RT_IAS_T)
IAS_IMS_RT_TB = IAS_IMS_RT_TB.append(pd.DataFrame(data = RT_IMS_T))
IAS_IMS_RT_TB = IAS_IMS_RT_TB.append(pd.DataFrame(data = RT_IAS_B))
IAS_IMS_RT_TB = IAS_IMS_RT_TB.append(pd.DataFrame(data = RT_IMS_B))

IAS_IMS_RT_TB['CuedHalf'] = ['Top']*40 + ['Bottom']*40
IAS_IMS_RT_TB['Alignment'] = ['Aligned']*20 + ['MisAligned']*20 + ['Aligned']*20 + ['MisAligned']*20
IAS_IMS_RT_TB['participant'] = list(range(1,21)) * 4

# ANOVA 
pandas2ri.activate()
r_IAS_IMAS_RT = pandas2ri.py2ri(IAS_IMS_RT_TB)

model = afex.aov_ez('participant', 'reactionTime', r_IAS_IMAS_RT, within = ['CuedHalf', 'Alignment'])
print(model)

# Descriptive
'''
np.mean([RT_IAS_T] + [RT_IMS_T]); np.std([RT_IAS_T] + [RT_IMS_T])
np.mean([RT_IAS_B] + [RT_IMS_B]); np.std([RT_IAS_B] + [RT_IMS_B])
'''

a = IAS_IMS_RT_TB.groupby(['Participant','CuedHalf'])['reactionTime'].describe()    
a.groupby(['CuedHalf'])['mean'].describe() 

# plot
means_corrRT_IAS_TB = [np.mean(RT_IAS_T) * 1000, np.mean(RT_IAS_B) * 1000]; errors_corrRT_CAS_TB = [stats.sem(RT_IAS_T) * 1000,stats.sem(RT_IAS_B) * 1000]
means_corrRT_IMS_TB = [np.mean(RT_IMS_T) * 1000, np.mean(RT_IMS_B) * 1000]; errors_corrRT_CMS_TB = [stats.sem(RT_IMS_T) * 1000,stats.sem(RT_IMS_B) * 1000]

index = np.arange(2)
bar_width = 0.35
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
rects1 = ax.bar(index,means_corrRT_IAS_TB, bar_width, yerr = errors_corrRT_CAS_TB, label = 'Aligned')
rects1 = ax.bar(index + bar_width,means_corrRT_IMS_TB, bar_width, yerr = errors_corrRT_CMS_TB, label = 'Misaligned')
ax.set_ylabel('RT')
ax.set_ylim((0,1000))
ax.set_title('Incongruent Same Conditions', fontweight="bold", fontsize=24)
ax.set_xticks(index + bar_width/ 2)
ax.set_xticklabels(('Top', 'Bottom'))
ax.legend()
plt.savefig('corrRT_SameIncongruent a vs. m_cueTB.tiff')
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(12,10),dpi=100)
plt.title('Top&Bottom, Same Congruent, 20 participants', fontweight="bold", fontsize=24)
plt.ylabel('Accuracy')
plt.ylim(0,800)
sns.boxplot(data=[RTT_CAS*1000, RTT_CMS*1000, RTB_CAS*1000, RTB_CMS*1000], palette = "Blues", showmeans=True)
sns.swarmplot(data=[RTT_CAS*1000, RTT_CMS*1000, RTB_CAS*1000, RTB_CMS*1000], color = ".25")
sns.despine()
plt.xticks([0,1,2,3], ['Aligned Top','Misaligned Top','Aligned Bottom','Misaligned Bottom'])
plt.savefig('corrRT_SameCongruent a vs. m_boxplot_cueTB.tiff')
plt.show()


"""
Complementary design (part-whole like): same congruent

"""

## t test: same congruent aligned - same congruent misaligned
# RT
DataTcor_CAS = DataT_cor[(DataT_cor['Congruency'] == 'C') & (DataT_cor['Alignment'] == 'A') & (DataT_cor['SameDifferent'] == 'S')]
DataTcor_CMS = DataT_cor[(DataT_cor['Congruency'] == 'C') & (DataT_cor['Alignment'] == 'M') & (DataT_cor['SameDifferent'] == 'S')]
RTT_CAS = DataTcor_CAS.groupby(['Participant'])['reactionTime'].mean()
RTT_CMS = DataTcor_CMS.groupby(['Participant'])['reactionTime'].mean()
t, p = stats.ttest_rel(RTT_CAS, RTT_CMS)
print(t,p) 

DataBcor_CAS = DataB_cor[(DataB_cor['Congruency'] == 'C') & (DataB_cor['Alignment'] == 'A') & (DataB_cor['SameDifferent'] == 'S')]
DataBcor_CMS = DataB_cor[(DataB_cor['Congruency'] == 'C') & (DataB_cor['Alignment'] == 'M') & (DataB_cor['SameDifferent'] == 'S')]
RTB_CAS = DataBcor_CAS.groupby(['Participant'])['reactionTime'].mean()
RTB_CMS = DataBcor_CMS.groupby(['Participant'])['reactionTime'].mean()
t, p = stats.ttest_rel(RTB_CAS, RTB_CMS)
print(t,p) 

CAS_CMS_corrRT = pd.DataFrame(
    {'Aligned_T': RTT_CAS,
     'Misaligned_T': RTT_CMS,
     'Aligned_B': RTB_CAS,
     'Misaligned_B': RTB_CMS,
    })

CAS_CMS_RT_long = pd.melt(CAS_CMS_corrRT, value_name='RT')
CAS_CMS_RT_long['CuedHalf'] = ['Top']*40 + ['Bottom']*40
CAS_CMS_RT_long['Alignment'] = (['Aligned']*20 + ['MisAligned']*20) * 2 
CAS_CMS_RT_long['Participant'] = list(range(1,21)) * 4

pandas2ri.activate()
r_CAS_CMS_RT_long = pandas2ri.py2ri(CAS_CMS_RT_long)

# I
model = afex.aov_ez('Participant', 'RT', r_CAS_CMS_RT_long, within = ['CuedHalf', 'Alignment'])
print(model)

# II
aovrm2way_corrRT_BT = AnovaRM(CAS_CMS_RT_long, 'RT', 'Participant', within = ['CuedHalf','Alignment'])
res2way_corrRT_BT = aovrm2way_corrRT_BT.fit()
print (res2way_corrRT_BT)


# descriptive
'''
np.mean([RTT_CAS] + [RTB_CAS])
np.std([RTT_CAS] + [RTB_CAS])
np.mean([RTT_CMS] + [RTB_CMS])
np.std([RTT_CMS] + [RTB_CMS])
'''

a = CAS_CMS_RT_long.groupby(['Participant','Alignment'])['RT'].describe()    
a.groupby(['Alignment'])['mean'].mean() 
a.groupby(['Alignment'])['mean'].std() 

# plot
means_corrRT_CAS_TB = [np.mean(RTT_CAS) * 1000, np.mean(RTB_CAS) * 1000]; errors_corrRT_CAS_TB = [stats.sem(RTT_CAS) * 1000,stats.sem(RTB_CAS) * 1000]
means_corrRT_CMS_TB = [np.mean(RTT_CMS) * 1000, np.mean(RTB_CMS) * 1000]; errors_corrRT_CMS_TB = [stats.sem(RTT_CMS) * 1000,stats.sem(RTB_CMS) * 1000]

index = np.arange(2)
bar_width = 0.35
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
rects1 = ax.bar(index,means_corrRT_CAS_TB, bar_width, yerr = errors_corrRT_CAS_TB, label = 'Aligned')
rects1 = ax.bar(index + bar_width,means_corrRT_CMS_TB, bar_width, yerr = errors_corrRT_CMS_TB, label = 'Misaligned')
ax.set_ylabel('RT')
ax.set_ylim((0,1000))
ax.set_title('Congruent Same Conditions', fontweight="bold", fontsize=24)
ax.set_xticks(index + bar_width/ 2)
ax.set_xticklabels(('Top', 'Bottom'))
ax.legend()
plt.savefig('corrRT_SameCongruent a vs. m_cueTB.tiff')
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(12,10),dpi=100)
plt.title('Top&Bottom, Same Congruent, 20 participants', fontweight="bold", fontsize=24)
plt.ylabel('Accuracy')
plt.ylim(0,800)
sns.boxplot(data=[RTT_CAS*1000, RTT_CMS*1000, RTB_CAS*1000, RTB_CMS*1000], palette = "Blues", showmeans=True)
sns.swarmplot(data=[RTT_CAS*1000, RTT_CMS*1000, RTB_CAS*1000, RTB_CMS*1000], color = ".25")
sns.despine()
plt.xticks([0,1,2,3], ['Aligned Top','Misaligned Top','Aligned Bottom','Misaligned Bottom'])
plt.savefig('corrRT_SameCongruent a vs. m_boxplot_cueTB.tiff')
plt.show()



"""
plot: traditional design + part-whole (complementary) design
same incongruent condition + same congruent condition
Two conditions in one plot

"""

# RT
barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Top','Bottom','Top','Bottom']
values1 = [np.mean(RTT_CAS*1000), np.mean(RTB_CAS*1000), np.mean(RT_IAS_T*1000), np.mean(RT_IAS_B*1000)]
errors1 = [stats.sem(RTT_CAS*1000),stats.sem(RTB_CAS*1000), stats.sem(RT_IAS_T*1000),stats.sem(RT_IAS_B*1000)]
values2 = [np.mean(RTT_CMS*1000), np.mean(RTB_CMS*1000), np.mean(RT_IMS_B*1000), np.mean(RT_IMS_B*1000)]
errors2 = [stats.sem(RTT_CMS*1000),stats.sem(RTB_CMS*1000), stats.sem(RT_IMS_B*1000),stats.sem(RT_IMS_B*1000)]

fig,ax = plt.subplots(figsize=(8,6),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, color='lightgrey', edgecolor='white', label = 'Aligned')
plt.bar(r2, values2, yerr=errors2, width=barWidth, color='black', edgecolor='white', label = 'Misaligned')
plt.ylim((0,1000))
plt.ylabel('RT (ms)')
plt.xticks([(r + barWidth/2) for r in range(4)], names)
plt.legend(loc = "upper right", bbox_to_anchor=(1.08, 1.09), labelspacing=0.1)  # bbox_to_anchor indicates the position of the right corner of the legend relative to the whole box
plt.text(0.75, 1100, "Same congruent", size=20,
         ha="center", va="center")
plt.text(2.75, 1100, "Same incongruent", size=20,
         ha="center", va="center")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('Traditional + Complementary_RT.tiff')
plt.show()



"""
save data

"""
# save the file to csv file
Data.to_csv('composite task_cue.csv', sep='\t')
CAS_CMS_acc_TB.to_csv('acc_Same congruent_long.csv', sep='\t')
IAS_IMS_acc_TB.to_csv('acc_Same incongruent_long.csv', sep='\t')

#CAS_CMS_acc_TB_wide = CAS_CMS_acc_TB.pivot(index='participant', values = 'isCorrect')

dprime_long.to_csv('dprime_composite task_cue.csv', sep='\t')
