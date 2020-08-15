# -*- coding: utf-8 -*-
"""
Created on 2/July/19
Updated on 28/Nov/19
Updated on 14/July/2020


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


updated on July 14, 2020
1. 32 participants
2. check whether there are outlier participants
3. remove nr.9 which is the outlier participants

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
from scipy.stats import norm, zscore
from scipy import stats
from statsmodels.stats.anova import AnovaRM
Z = norm.ppf

import pingouin as pg
from pingouin import ttest,pairwise_ttests

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
check whether there are any outlier participants

"""
# grand mean of all conditions
summary_par = Data.groupby(['Participant'])['isCorrect'].mean()  
summary_par = summary_par.to_frame()
summary_par['z_summary_par'] = stats.zscore(summary_par)
summary_par['z_outlier'] = ['no' if (x > -2.5 and x < 2.5) else 'yes' for x in summary_par['z_summary_par']]
# !!! partipant nr.9 is the outlier!!!

# mean of each condition
summary = Data.groupby(['Participant','CuedHalf','Congruency', 'Alignment'], as_index=False)['isCorrect'].mean()  

summary_wide = pd.pivot_table(summary, index = "Participant", columns = ['CuedHalf','Congruency', 'Alignment'], aggfunc = max)
summary_wide_zscore = summary_wide.apply(zscore)
i= 0
for i in range(8):
    summary_wide_zscore[str(i)] = ['no' if (x > -2.5 and x < 2.5) else 'yes' for x in summary_wide_zscore.iloc[:,i]]
    i = i + 1
# !!! partipant nr.2, 17, 24, 25 had outlier acc in some of the conditions !!!

# remove participant Nr.4 outlier
Data = Data[Data.Participant != 9]


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
plt.savefig('Congruency by alignment_acc_cueT_nr9 removed.tiff')
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
plt.savefig('Congruency by alignment_acc_cueB_nr9 removed.tiff')
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
print(t,p)  # bottom: no difference between IAS and IMS, p = .129

IAS_IMS_acc_TB = pd.DataFrame(data = acc_IAS_T)
IAS_IMS_acc_TB = IAS_IMS_acc_TB.append(pd.DataFrame(data = acc_IMS_T))
IAS_IMS_acc_TB = IAS_IMS_acc_TB.append(pd.DataFrame(data = acc_IAS_B))
IAS_IMS_acc_TB = IAS_IMS_acc_TB.append(pd.DataFrame(data = acc_IMS_B))

IAS_IMS_acc_TB['CuedHalf'] = ['Top']*62 + ['Bottom']*62
IAS_IMS_acc_TB['Alignment'] = ['Aligned']*31 + ['MisAligned']*31 + ['Aligned']*31 + ['MisAligned']*31
IAS_IMS_acc_TB['participant'] = list(range(1,32)) * 4

# ANOVA 
pandas2ri.activate()
r_IAS_IMAS_acc = pandas2ri.py2ri(IAS_IMS_acc_TB)

model = afex.aov_ez('participant', 'isCorrect', r_IAS_IMAS_acc, within = ['CuedHalf', 'Alignment'])
print(model)   # main effect of Cuedhalf, F(1,30) <1, p = .47; main effect of alignment, F(1,30) = 21.44, p < .0001  
# Interaction between Cuedhalf and alignment, F(1,30) = 15.13, p = .0005
 
# simple effect
inter_cuedhalf_align = emmeans.emmeans(model, 'Alignment', by = 'CuedHalf', contr = 'pairwise', adjust = 'bonferroni')
print(inter_cuedhalf_align)

# top target
IAS_IMS_acc_T = IAS_IMS_acc_TB[(IAS_IMS_acc_TB['CuedHalf'] == 'Top')] 
IAS_IMS_acc_T_stats = IAS_IMS_acc_T.groupby(['participant','CuedHalf','Alignment'])['isCorrect'].mean()
IAS_IMS_acc_T_stats0 = pairwise_ttests(dv='isCorrect', within='Alignment',
                            subject='participant', data= IAS_IMS_acc_T_stats.to_frame().reset_index(),
                            effsize='cohen')
# top: acc(aligned) < acc(misaligned), t(30)=-6.25, p<.0001, d = -.84


# bottom target
IAS_IMS_acc_B = IAS_IMS_acc_TB[(IAS_IMS_acc_TB['CuedHalf'] == 'Bottom')] 
IAS_IMS_acc_B_stats = IAS_IMS_acc_B.groupby(['participant','CuedHalf','Alignment'])['isCorrect'].mean()
IAS_IMS_acc_B_stats0 = pairwise_ttests(dv='isCorrect', within='Alignment',
                            subject='participant', data= IAS_IMS_acc_B_stats.to_frame().reset_index(),
                            effsize='cohen')
# top: acc(aligned) < acc(misaligned), t(30)=-1.56, p=.13, d = -.25

 
# descriptive
a = IAS_IMS_acc_TB.groupby(['Participant','CuedHalf','Alignment'])['isCorrect'].describe()    
a.groupby(['CuedHalf','Alignment'])['mean'].describe() 

"""
plot

"""
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
plt.savefig('acc_SameIncongruent a vs. m_cueTB_nr9 removed.tiff')
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
print(t,p)  # top: no difference between CAS and CMS, p = .124

Data_CAS_B = Data[(Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'A') & (Data['SameDifferent'] == 'S')]
Data_CMS_B = Data[(Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'M') & (Data['SameDifferent'] == 'S')]
acc_CAS_B = Data_CAS_B.groupby(['Participant'])['isCorrect'].mean()
acc_CMS_B = Data_CMS_B.groupby(['Participant'])['isCorrect'].mean()
t, p = stats.ttest_rel(acc_CAS_B, acc_CMS_B)
print(t,p)  # bottom: significant difference between CAS and CMS, t = 3.211, p = .0031

CAS_CMS_acc_TB = pd.DataFrame(data = acc_CAS_T)
CAS_CMS_acc_TB = CAS_CMS_acc_TB.append(pd.DataFrame(data = acc_CMS_T))
CAS_CMS_acc_TB = CAS_CMS_acc_TB.append(pd.DataFrame(data = acc_CAS_B))
CAS_CMS_acc_TB = CAS_CMS_acc_TB.append(pd.DataFrame(data = acc_CMS_B))

CAS_CMS_acc_TB['CuedHalf'] = ['Top']*62 + ['Bottom']*62
CAS_CMS_acc_TB['Alignment'] = ['Aligned']*31 + ['MisAligned']*31 + ['Aligned']*31 + ['MisAligned']*31
CAS_CMS_acc_TB['participant'] = list(range(1,32)) * 4

# ANOVA 
pandas2ri.activate()
r_CAS_CMS_acc_TB = pandas2ri.py2ri(CAS_CMS_acc_TB)

model = afex.aov_ez('participant', 'isCorrect', CAS_CMS_acc_TB, within = ['CuedHalf', 'Alignment'])
print(model)  # main effect of Cuedhalf, F(1,30) = 9.14, p = .005; main effect of alignment, F(1,30) = 11.23, p = .002  
# No iteraction between Cuedhalf and alignment, F(1,30) <1, p = .38



# descriptive
a = CAS_CMS_acc_TB.groupby(['Participant','Alignment',])['isCorrect'].describe()    
a.groupby(['Alignment'])['mean'].describe() 


a = CAS_CMS_acc_TB.groupby(['Participant','CuedHalf','Alignment',])['isCorrect'].describe()    
a.groupby(['CuedHalf','Alignment'])['mean'].mean() 
a.groupby(['CuedHalf','Alignment'])['mean'].std() 

"""
plot

"""
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
plt.savefig('acc_SameCongruent a vs. m_cueTB_nr9 removed.tiff')
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
plt.savefig('Traditional + Complementary_acc_nr9 removed.tiff')
plt.show()

# box plot ~ used in Ms
IAS_IMS_CAS_CMS_acc_TB = CAS_CMS_acc_TB.append(IAS_IMS_acc_TB)
IAS_IMS_CAS_CMS_acc_TB = IAS_IMS_CAS_CMS_acc_TB.rename(columns={"isCorrect" : "Accuracy"})
IAS_IMS_CAS_CMS_acc_TB["Condition"] = ["Same congruent"] * 124 + ["Same incongruent"] * 124


g = sns.catplot(x="CuedHalf", y="Accuracy", hue="Alignment", row="Condition",
                data=IAS_IMS_CAS_CMS_acc_TB,
                kind="box", showmeans = True, sharex = False, sharey="row", legend = False,
                height=4,
                aspect=1.3,
                dodge=True,
                palette = sns.cubehelix_palette(n_colors=2, rot = -.2, hue=1, light = 0.7, dark = 0.3, reverse = True))
plt.legend(loc='best')
g.set(ylim = (-0.05,1.05), xlabel="")
g.set_titles("{row_name}")
g.fig.subplots_adjust(hspace=.3, wspace=0.1)
plt.savefig('Traditional + Complementary_acc_boxplot_nr9 removed.tiff', bbox_inches = 'tight', dpi=100)
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
    for i in range(31):
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
#t1, p1 = stats.ttest_1samp(d_CA_T,0.0)
#t2, p2 = stats.ttest_1samp(d_CM_T,0.0) 
#t3, p3 = stats.ttest_1samp(d_IA_T,0.0)
#t4, p4 = stats.ttest_1samp(d_IM_T,0.0)
#t5, p5 = stats.ttest_1samp(d_CA_B,0.0)
#t6, p6 = stats.ttest_1samp(d_CM_B,0.0) 
#t7, p7 = stats.ttest_1samp(d_IA_B,0.0)
#t8, p8 = stats.ttest_1samp(d_IM_B,0.0) 

e1 = ttest(d_CA_T,0.0)
e2 = ttest(d_CM_T,0.0)
e3 = ttest(d_IA_T,0.0) 
e4 = ttest(d_IM_T,0.0)
[e1['p-val'],e2['p-val'],e3['p-val'],e4['p-val']]
[e1['cohen-d'],e2['cohen-d'],e3['cohen-d'],e4['cohen-d']]

e5 = ttest(d_CA_B,0.0)
e6 = ttest(d_CM_B,0.0)
e7 = ttest(d_IA_B,0.0)   
e8 = ttest(d_IM_B,0.0)   
[e5['p-val'],e6['p-val'],e7['p-val'],e8['p-val']]
[e5['cohen-d'],e6['cohen-d'],e7['cohen-d'],e8['cohen-d']]


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
dprime_long['CuedHalf'] = ['Top']*124 + ['Bottom']*124
dprime_long['Alignment'] = (['Aligned']*31 + ['MisAligned']*31) * 4 
dprime_long['Congruency'] = (['Congruent']*62 + ['Incongruent']*62) * 2
dprime_long['participant'] = list(range(1,32)) * 8

# ANOVA
pandas2ri.activate()
r_dprime_long = pandas2ri.py2ri(dprime_long)
model = afex.aov_ez('participant', 'd', r_dprime_long, within = ['CuedHalf','Alignment', 'Congruency'])
print(model)  # Effect of Top/Bottom, main effect of congruency, interaction between congruency and alignment, and interaction between cued half and alignment，
                       # the three way interaction is also significant

# main effect
main_prob = emmeans.emmeans(model, 'CuedHalf', contr = 'pairwise', adjust = 'bonferroni')
print(main_prob)

main_congr = emmeans.emmeans(model, 'Congruency', contr = 'pairwise', adjust = 'bonferroni')
print(main_congr)

# simple effect
inter_cuedhalf_align = emmeans.emmeans(model, 'Alignment', by = 'CuedHalf', contr = 'pairwise', adjust = 'bonferroni')
print(inter_cuedhalf_align)

inter_congr_align = emmeans.emmeans(model, 'Alignment', by = 'Congruency', contr = 'pairwise', adjust = 'bonferroni')
print(inter_congr_align)


# top
dprimeT = dprime_long[(dprime_long['CuedHalf'] == 'Top')]
r_dprime_T = pandas2ri.py2ri(dprimeT)
model_dprimeT = afex.aov_ez('participant', 'd', r_dprime_T, within = ['Alignment','Congruency'])
print(model_dprimeT) 


# bottom
dprimeB = dprime_long[(dprime_long['CuedHalf'] == 'Bottom')]
r_dprime_B = pandas2ri.py2ri(dprimeB)
model_dprimeB = afex.aov_ez('participant', 'd', r_dprime_B, within = ['Alignment','Congruency'])
print(model_dprimeB) 

"""
plot

"""
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
plt.savefig('d_cueT_nr9 removed.tiff')
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
plt.savefig('d_cueB_nr9 removed.tiff')
plt.show()


# Two ratio conditions in one plot, using subplot function
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
plt.savefig('Congruency by alignment_dprime_nr9 removed.tiff')
plt.show()

# plot, line graph ~~~~~~~~~~~ used in the Ms
dprime_long['d prime'] = dprime_long['d']

g = sns.catplot(x="Alignment", y="d prime", hue="Congruency", row="CuedHalf", 
               hue_order=["Congruent","Incongruent"], kind = "point",
               linestyles=["-", "--"], markers=["o", "^"], sharex=False, sharey="row", legend = False,
               data=dprime_long, palette = sns.cubehelix_palette(n_colors=2, hue=1, light = 0.7, dark = 0.3, reverse = True),   # https://seaborn.pydata.org/generated/seaborn.cubehelix_palette.html
               height = 4,
               aspect = 1.3)
plt.legend(loc='best')   # https://stackoverflow.com/questions/43151440/remove-seaborn-barplot-legend-title/47335905
#g.fig.suptitle('Experiment 2', y=1.05)
g.set(ylim = (0,3), xlabel="")
g.set_titles("{row_name}")
g.fig.subplots_adjust(hspace=.3, wspace=0.1)   #Increase space between rows and remove space between cols
#sns.set_context("paper", rc={'figure.figsize':(8,6)}) 
plt.savefig('d_congruency x alignment_line_nr9 removed.tiff', bbox_inches = 'tight', dpi=100)
plt.show()


"""
d prime, composite effect
[(d′aligned_congruent - d′aligned_incongruent) - (d′misaligned_congruent - d′misaligned_incongruent)] 
"""
dprime_long[(dprime_long['CuedHalf'] == 'Bottom')]


dprime_top = (dprime['Congruent_Aligned_Top'] - dprime['Incongruent_Aligned_Top']) - (dprime['Congruent_Misligned_Top'] - dprime['Incongruent_Misligned_Top'])
dprime_bottom = (dprime['Congruent_Aligned_Bottom'] - dprime['Incongruent_Aligned_Bottom']) - (dprime['Congruent_Misligned_Bottom'] - dprime['Incongruent_Misligned_Bottom'])

dprime_compositesize = dprime_top.append(dprime_bottom)
dprime_compositesize = dprime_compositesize.to_frame()
dprime_compositesize.columns = ['d']
dprime_compositesize['Participant'] = list(range(1,32))*2
dprime_compositesize['Cuedhalf'] = ['top'] * 31 + ['bottom'] * 31 

dprime_composite_t = pairwise_ttests(dv='d', within='Cuedhalf',
                            subject='Participant', data= dprime_compositesize.reset_index(),
                            effsize='cohen')

# descriptive stats
dprime_compositesize.groupby(['Cuedhalf'])['d'].describe()

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
corrRT_long['CuedHalf'] = ['Top']*124 + ['Bottom']*124
corrRT_long['Alignment'] = (['Aligned']*31 + ['MisAligned']*31) * 4 
corrRT_long['Congruency'] = (['Congruent']*62 + ['Incongruent']*62) * 2
corrRT_long['Participant'] = list(range(1,32)) * 8

# ANOVA
pandas2ri.activate()
r_corrRT_long = pandas2ri.py2ri(corrRT_long)

model = afex.aov_ez('Participant', 'corrRT', r_corrRT_long, within = ['CuedHalf', 'Alignment','Congruency'])
print(model)

# simple effect
inter_cong_align = emmeans.emmeans(model, 'Congruency', by = 'Alignment', contr = 'pairwise', adjust = 'bonferroni')
print(inter_cong_align)

corrRT_aligned = corrRT_long[(corrRT_long['Alignment'] == 'Aligned')] 
correctRT_alig_stats = corrRT_aligned.groupby(['Participant','Congruency'])['corrRT'].mean()
correctRT_alig_stats0 = pairwise_ttests(dv='corrRT', within='Congruency',
                            subject='Participant', data= correctRT_alig_stats.to_frame().reset_index(),
                            effsize='cohen')

corrRT_misaligned = corrRT_long[(corrRT_long['Alignment'] == 'MisAligned')] 
correctRT_mis_stats = corrRT_misaligned.groupby(['Participant','Congruency'])['corrRT'].mean()
correctRT_mis_stats0 = pairwise_ttests(dv='corrRT', within='Congruency',
                            subject='Participant', data= correctRT_mis_stats.to_frame().reset_index(),
                            effsize='cohen')

# descriptive
a = corrRT_long.groupby(['Participant','Congruency','Alignment'])['corrRT'].describe()    
a.groupby(['Congruency','Alignment'])['mean'].describe() 

"""
plot

"""
# Two cuedhalf conditions in one plot, using subplot function

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
plt.savefig('Congruency by alignment_RT_nr9 removed.tiff')
plt.show()


# plot, line graph ~~~~~~~~~~~ used in the Ms
RT_correct_long = pd.concat([RTT_CA*1000, RTT_CM*1000, RTB_CA*1000, RTB_CM*1000,
                RTT_IA*1000, RTT_IM*1000, RTB_IA*1000, RTB_IM*1000], ignore_index=True)

RT_correct_long = RT_correct_long.to_frame()    
RT_correct_long['Alignment'] = (["Aligned"] * 31 + ["MisAligned"] * 31) * 4
RT_correct_long['Congruency'] = (["Congruent"] * 124 + ["Incongruent"] * 124) 
RT_correct_long['CuedHalf'] = (["Top"] * 62 + ["Bottom"] * 62) * 2
RT_correct_long['Participant'] = list(range(1,32)) * 8
RT_correct_long = RT_correct_long.rename(columns={"reactionTime": "Response Time (ms)"})

g = sns.catplot(x="Alignment", y="Response Time (ms)", hue="Congruency", row='CuedHalf',
               hue_order=["Congruent","Incongruent"], kind = "point",
               linestyles=["-", "--"], markers=["o", "^"], sharex=False, sharey="row", legend = False,
               data=RT_correct_long, palette = sns.cubehelix_palette(n_colors=2, hue=1, light = 0.7, dark = 0.3, reverse = True),   # https://seaborn.pydata.org/generated/seaborn.cubehelix_palette.html
               height = 4,
               aspect = 1.3)
plt.legend(loc='best')   # https://stackoverflow.com/questions/43151440/remove-seaborn-barplot-legend-title/47335905
#g.fig.suptitle('Experiment 2', y=1.05)
g.set(ylim = (200,1200), xlabel="")
g.set_titles("{row_name}")
g.fig.subplots_adjust(hspace=.3, wspace=0.1)   #Increase space between rows and remove space between cols
#sns.set_context("paper", rc={'figure.figsize':(8,6)}) 
plt.savefig('RT_congruency x alignment_line_nr9 removed.tiff', bbox_inches = 'tight', dpi=100)
plt.show()



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

IAS_IMS_RT_TB['CuedHalf'] = ['Top']*62 + ['Bottom']*62
IAS_IMS_RT_TB['Alignment'] = ['Aligned']*31 + ['MisAligned']*31 + ['Aligned']*31 + ['MisAligned']*31
IAS_IMS_RT_TB['participant'] = list(range(1,32)) * 4

# ANOVA 
pandas2ri.activate()
r_IAS_IMAS_RT = pandas2ri.py2ri(IAS_IMS_RT_TB)

model = afex.aov_ez('participant', 'reactionTime', r_IAS_IMAS_RT, within = ['CuedHalf', 'Alignment'])
print(model)

# Descriptive
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
print(t,p)  # no difference between aligned and misalinged

DataBcor_CAS = DataB_cor[(DataB_cor['Congruency'] == 'C') & (DataB_cor['Alignment'] == 'A') & (DataB_cor['SameDifferent'] == 'S')]
DataBcor_CMS = DataB_cor[(DataB_cor['Congruency'] == 'C') & (DataB_cor['Alignment'] == 'M') & (DataB_cor['SameDifferent'] == 'S')]
RTB_CAS = DataBcor_CAS.groupby(['Participant'])['reactionTime'].mean()
RTB_CMS = DataBcor_CMS.groupby(['Participant'])['reactionTime'].mean()
t, p = stats.ttest_rel(RTB_CAS, RTB_CMS)  # RT longer in the misaligned condition
print(t,p) 

CAS_CMS_RT_TB = pd.DataFrame(data = RTT_CAS)
CAS_CMS_RT_TB = CAS_CMS_RT_TB.append(pd.DataFrame(data = RTT_CMS))
CAS_CMS_RT_TB = CAS_CMS_RT_TB.append(pd.DataFrame(data = RTB_CAS))
CAS_CMS_RT_TB = CAS_CMS_RT_TB.append(pd.DataFrame(data = RTB_CMS))

CAS_CMS_RT_TB['CuedHalf'] = ['Top']*62 + ['Bottom']*62
CAS_CMS_RT_TB['Alignment'] = ['Aligned']*31 + ['MisAligned']*31 + ['Aligned']*31 + ['MisAligned']*31
CAS_CMS_RT_TB['participant'] = list(range(1,32)) * 4


# ANOVA
pandas2ri.activate()
r_CAS_CMS_RT_TB = pandas2ri.py2ri(CAS_CMS_RT_TB)

model = afex.aov_ez('participant', 'reactionTime', r_CAS_CMS_RT_TB, within = ['CuedHalf', 'Alignment'])
print(model)

# descriptive
a = CAS_CMS_RT_TB.groupby(['participant','Alignment'])['reactionTime'].describe()    
a.groupby(['Alignment'])['mean'].describe() 


"""
plot

"""
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


# box plot ~ used in Ms
IAS_IMS_CAS_CMS_RT_TB = CAS_CMS_RT_TB.append(IAS_IMS_RT_TB)
IAS_IMS_CAS_CMS_RT_TB = IAS_IMS_CAS_CMS_RT_TB.rename(columns={"reactionTime" : "Response Time (ms)"})
IAS_IMS_CAS_CMS_RT_TB["Condition"] = ["Same congruent"] * 124 + ["Same incongruent"] * 124 
IAS_IMS_CAS_CMS_RT_TB["Response Time (ms)"] = IAS_IMS_CAS_CMS_RT_TB["Response Time (ms)"] * 1000

g = sns.catplot(x="CuedHalf", y="Response Time (ms)", hue="Alignment", row="Condition",
                data=IAS_IMS_CAS_CMS_RT_TB,
                kind="box", showmeans = True, sharex = False, sharey="row", legend = False,
                height=4,
                aspect=1.3,
                dodge=True,
                palette = sns.cubehelix_palette(n_colors=2, rot = -.2, hue=1, light = 0.7, dark = 0.3, reverse = True))
plt.legend(loc='best')
g.set(ylim = (200,2500), xlabel="")
g.set_titles("{row_name}")
g.fig.subplots_adjust(hspace=.3, wspace=0.1)
plt.savefig('Traditional + Complementary_RT_boxplot_nr9 removed.tiff', bbox_inches = 'tight', dpi=100)
plt.show()


"""
save data

"""
# save the file to csv file
Data.to_csv('composite task_cue.csv', sep='\t')
CAS_CMS_acc_TB.to_csv('acc_Same congruent_long.csv', sep='\t')
IAS_IMS_acc_TB.to_csv('acc_Same incongruent_long.csv', sep='\t')

dprime_long.to_csv('dprime_composite task_cue.csv', sep='\t')
