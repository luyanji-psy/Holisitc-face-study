# -*- coding: utf-8 -*-
"""
Created on 25/February/2020

Modified on 2/April/2020
* To change 0.25Top cue & 0.75Top cue to 0.25cueTarget & 0.75cueTarget

Updated on 8/June/2020
32 participants in total

@author: Ji

"""

"""
1. Top/Bottom, Probabiltiy of top cue (0.25/0.75), Congruency, Alignment as within-subject variable, put into the ANOVA
2. Use R packages to do ANOVA, with effect size output
3. Use pingouin packages (only two-way repeated measure ANOVA supported), provide more options of effect sizes

Traditional design: same incongruent trials, aligned vs. misaligned
Complementary design: same congruent trials, aligned vs. misaligned

"""

"""
    
updated on May 4, 2020
1. Check whether there are any outlier participants in this experiment
-- grand mean accuracy (collapsed across conditions) for each participant, whether any individual 
   had grand mean which was below or above 2.5sd of the grand mean of all participants

!!! Participant nr.4 is the outlier !!!   
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
import copy
import seaborn as sns

import pingouin as pg
from pingouin import ttest,pairwise_ttests

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import r, pandas2ri

packageNames = ('afex', 'emmeans')
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)
 
packnames_to_install = [x for x in packageNames if not rpackages.isinstalled(x)]
 
if len(packnames_to_install) > 0:
   utils.install_packages(StrVector(packnames_to_install))

afex = rpackages.importr('afex')
emmeans = rpackages.importr('emmeans', robject_translations = {"recover.data.call": "recover_data_call1"})

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 24

# reset the figure formats
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

raw_path = r'G:\Post-doc HKU\Haiyang\Cue_Complete_2575 proportion\CF_Complete_Cue_probability\Excel Data'
path = r'G:\Post-doc HKU\Haiyang\Cue_Complete_2575 proportion\CF_Complete_Cue_probability\Analysis'

# change the working folder to path
os.chdir(raw_path)
allFiles = glob.glob(raw_path + "/*.xlsx")
Data0 = pd.concat((pd.read_excel(f) for f in allFiles), sort = False)

os.chdir(path)

"""
rename condition levels 
Probability was recorded as "0.25Topcue" in both probability conditions

"""
Data = copy.copy(Data0)

def myfunc(x,y):
    if x == 'CF_Complete_75BottomCue':
        return y
    elif x == 'CF_Complete_75TopCue':
        return '0.75Topcue'

Data['Probability'] = Data.apply(lambda x:myfunc(x.Experiment,x.Probability), axis=1)

"""
rename condition levels 
Probability was recoded as "0.25cueTarget" and "0.75cueTarget"
When the cuedhalf was Top and Probility was 0.25Topcue, then it was changed to "0.25cueTarget"
When the cuedhalf was Top and Probility was 0.75Topcue, then it was changed to "0.75cueTarget"
When the cuedhalf was Bottom and Probility was 0.25Topcue, then it was changed to "0.75cueTarget"
When the cuedhalf was Bottom and Probility was 0.75Topcue, then it was changed to "0.25cueTarget" 

"""
def myfunc(a,b,y):
    if a == 'CF_Complete_75BottomCue' and b == 'B':
        return '0.75cueTarget'
    elif a == 'CF_Complete_75TopCue'and b == 'B':
        return '0.25cueTarget'
    elif a == 'CF_Complete_75BottomCue'and b == 'T':
        return '0.25cueTarget'
    elif a == 'CF_Complete_75TopCue'and b == 'T':
        return '0.75cueTarget'
    
Data['Probability'] = Data.apply(lambda x:myfunc(x.Experiment,x.CuedHalf,x.Probability), axis=1)


"""""
Top & bottom, acc
25%cueTarget & 75%cueTarget, separately

"""""
summary = Data.groupby(['Participant','Probability','CuedHalf','Congruency', 'Alignment'], as_index=False)['isCorrect'].mean()  

# grand mean of all conditions
summary_par = Data.groupby(['Participant'])['isCorrect'].mean()  
summary_par = summary_par.to_frame()
summary_par['z_summary_par'] = stats.zscore(summary_par)
summary_par['z_outlier'] = ['no' if (x > -2.5 and x < 2.5) else 'yes' for x in summary_par['z_summary_par']]
# !!! partipant nr.4 is the outlier!!!

# mean of each condition
summary_wide = pd.pivot_table(summary, index = "Participant", columns = ['Probability','CuedHalf','Congruency', 'Alignment'], aggfunc = max)
summary_wide_zscore = summary_wide.apply(zscore)
i= 0
for i in range(16):
    summary_wide_zscore[str(i)] = ['no' if (x > -2.5 and x < 2.5) else 'yes' for x in summary_wide_zscore.iloc[:,i]]
    i = i + 1
# !!! partipant nr.4 had outlier acc in some of the conditions !!!


# top cue
Data_CA_T_25tar = Data[(Data['Probability'] == '0.25cueTarget') & (Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'A')]
Data_CM_T_25tar = Data[(Data['Probability'] == '0.25cueTarget') & (Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'M')]
Data_IA_T_25tar = Data[(Data['Probability'] == '0.25cueTarget') & (Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'A')]
Data_IM_T_25tar = Data[(Data['Probability'] == '0.25cueTarget') & (Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'M')]

Data_CA_T_75tar = Data[(Data['Probability'] == '0.75cueTarget') & (Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'A')]
Data_CM_T_75tar = Data[(Data['Probability'] == '0.75cueTarget') & (Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'M')]
Data_IA_T_75tar = Data[(Data['Probability'] == '0.75cueTarget') & (Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'A')]
Data_IM_T_75tar = Data[(Data['Probability'] == '0.75cueTarget') & (Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'M')]

# bottom cue
Data_CA_B_75tar = Data[(Data['Probability'] == '0.75cueTarget') & (Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'A')]
Data_CM_B_75tar = Data[(Data['Probability'] == '0.75cueTarget') & (Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'M')]
Data_IA_B_75tar = Data[(Data['Probability'] == '0.75cueTarget') & (Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'A')]
Data_IM_B_75tar = Data[(Data['Probability'] == '0.75cueTarget') & (Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'M')]

Data_CA_B_25tar = Data[(Data['Probability'] == '0.25cueTarget') & (Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'A')]
Data_CM_B_25tar = Data[(Data['Probability'] == '0.25cueTarget') & (Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'M')]
Data_IA_B_25tar = Data[(Data['Probability'] == '0.25cueTarget') & (Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'A')]
Data_IM_B_25tar = Data[(Data['Probability'] == '0.25cueTarget') & (Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'M')]

# acc_top
acc_CA_T_25tar = Data_CA_T_25tar.groupby(['Participant'])['isCorrect'].mean()
acc_CM_T_25tar = Data_CM_T_25tar.groupby(['Participant'])['isCorrect'].mean()
acc_IA_T_25tar = Data_IA_T_25tar.groupby(['Participant'])['isCorrect'].mean()
acc_IM_T_25tar = Data_IM_T_25tar.groupby(['Participant'])['isCorrect'].mean()

acc_CA_T_75tar = Data_CA_T_75tar.groupby(['Participant'])['isCorrect'].mean()
acc_CM_T_75tar = Data_CM_T_75tar.groupby(['Participant'])['isCorrect'].mean()
acc_IA_T_75tar = Data_IA_T_75tar.groupby(['Participant'])['isCorrect'].mean()
acc_IM_T_75tar = Data_IM_T_75tar.groupby(['Participant'])['isCorrect'].mean()

# acc_bottom
acc_CA_B_75tar = Data_CA_B_75tar.groupby(['Participant'])['isCorrect'].mean()
acc_CM_B_75tar = Data_CM_B_75tar.groupby(['Participant'])['isCorrect'].mean()
acc_IA_B_75tar = Data_IA_B_75tar.groupby(['Participant'])['isCorrect'].mean()
acc_IM_B_75tar = Data_IM_B_75tar.groupby(['Participant'])['isCorrect'].mean()

acc_CA_B_25tar = Data_CA_B_25tar.groupby(['Participant'])['isCorrect'].mean()
acc_CM_B_25tar = Data_CM_B_25tar.groupby(['Participant'])['isCorrect'].mean()
acc_IA_B_25tar = Data_IA_B_25tar.groupby(['Participant'])['isCorrect'].mean()
acc_IM_B_25tar = Data_IM_B_25tar.groupby(['Participant'])['isCorrect'].mean()

# barplot
# two cue probabilities in one plot, top as target
barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Aligned','Misaligned','Aligned','Misaligned']
values1 = [np.mean(acc_CA_T_25tar), np.mean(acc_CM_T_25tar), np.mean(acc_CA_T_75tar), np.mean(acc_CM_T_75tar)]
errors1 = [stats.sem(acc_CA_T_25tar),stats.sem(acc_CM_T_25tar), stats.sem(acc_CA_T_75tar),stats.sem(acc_CM_T_75tar)]
values2 = [np.mean(acc_IA_T_25tar), np.mean(acc_IM_T_25tar), np.mean(acc_IA_T_75tar), np.mean(acc_IM_T_75tar)]
errors2 = [stats.sem(acc_IA_T_25tar),stats.sem(acc_IM_T_25tar), stats.sem(acc_IA_T_75tar),stats.sem(acc_IM_T_75tar)]

fig,ax = plt.subplots(figsize=(8,6),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, color='lightgrey', edgecolor='white', label = 'Congruent')
plt.bar(r2, values2, yerr=errors2, width=barWidth, color='black', edgecolor='white', label = 'Incongruent')
plt.ylim((0,1))
plt.ylabel('Accuracy')
plt.xticks([(r + barWidth/2) for r in range(4)], names)
plt.legend(loc = "upper right", bbox_to_anchor=(1.08, 1.05), labelspacing=0.1)  # bbox_to_anchor indicates the position of the right corner of the legend relative to the whole box
plt.text(1.75, 1.15, "Top", size=24,
         ha="center", va="center")
plt.text(0.75, 1.07, "25% cue target", size=20,
         ha="center", va="center")
plt.text(2.75, 1.07, "75% cue target", size=20,
         ha="center", va="center")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('Congruency by alignment_accT.tiff')
plt.show()


# two cue probabilities in one plot, bottom as target
barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Aligned','Misaligned','Aligned','Misaligned']
values1 = [np.mean(acc_CA_B_25tar), np.mean(acc_CM_B_25tar), np.mean(acc_CA_B_75tar), np.mean(acc_CM_B_75tar)]
errors1 = [stats.sem(acc_CA_B_25tar),stats.sem(acc_CM_B_25tar), stats.sem(acc_CA_B_75tar),stats.sem(acc_CM_B_75tar)]
values2 = [np.mean(acc_IA_B_25tar), np.mean(acc_IM_B_25tar), np.mean(acc_IA_B_75tar), np.mean(acc_IM_B_75tar)]
errors2 = [stats.sem(acc_IA_B_25tar),stats.sem(acc_IM_B_25tar), stats.sem(acc_IA_B_75tar),stats.sem(acc_IM_B_75tar)]

fig,ax = plt.subplots(figsize=(8,6),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, color='lightgrey', edgecolor='white', label = 'Congruent')
plt.bar(r2, values2, yerr=errors2, width=barWidth, color='black', edgecolor='white', label = 'Incongruent')
plt.ylim((0,1))
plt.ylabel('Accuracy')
plt.xticks([(r + barWidth/2) for r in range(4)], names)
plt.legend(loc = "upper right", bbox_to_anchor=(1.08, 1.05), labelspacing=0.1)  # bbox_to_anchor indicates the position of the right corner of the legend relative to the whole box
plt.text(1.75, 1.15, "Bottom", size=24,
         ha="center", va="center")
plt.text(0.75, 1.07, "25% cue target", size=20,
         ha="center", va="center")
plt.text(2.75, 1.07, "75% cue target", size=20,
         ha="center", va="center")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('Congruency by alignment_accB.tiff')
plt.show()


# anova
# four way: cue part, cue probability, congruency, alignment
pandas2ri.activate()
r_Data = pandas2ri.py2ri(Data)

model_acc = afex.aov_ez('Participant', 'isCorrect', r_Data, within = ['CuedHalf','Probability','Congruency', 'Alignment'])  
print(model_acc)
# Main effect of probability, congruency
# No four-way interaction
# Three-way interaction: cuedHalf x Congruency x Alignment
# Two-way interaction: Congruency x Alignment, Cuedhalf * Alignment, Probability x Congruency

# To examine three way: cuedHalf, congruency, alignment
# Top
r_Data_Top = pandas2ri.py2ri(Data[Data['CuedHalf'] == 'T'])

model_accT = afex.aov_ez('Participant', 'isCorrect', r_Data_Top, within = ['Congruency', 'Alignment'])  
print(model_accT)
# Main effect of Congruency, Main effect of Alignment
# Interaction between congruency and alignment


# To examine three way: cuedHalf, congruency, alignment
# Bottom
r_Data_Bottom = pandas2ri.py2ri(Data[Data['CuedHalf'] == 'B'])

model_accB = afex.aov_ez('Participant', 'isCorrect', r_Data_Bottom, within = ['Congruency', 'Alignment'])  
print(model_accB)
# Main effect of Congruency. No main effect of alignment!
# No interaction between congruency and alignment!! , p = .061


"""""""""""""""""""""""""""""""""""""""
Traditional design: same incongruent

"""""""""""""""""""""""""""""""""""""""

"""""""""""
#descriptive

"""""""'"""
# acc, 25% cue target
Data_IAS_T_25tar = Data[(Data['Probability'] == '0.25cueTarget') & (Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'A') & (Data['SameDifferent'] == 'S')]
Data_IMS_T_25tar = Data[(Data['Probability'] == '0.25cueTarget') & (Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'M') & (Data['SameDifferent'] == 'S')]
acc_IAS_T_25tar = Data_IAS_T_25tar.groupby(['Participant'])['isCorrect'].mean()
acc_IMS_T_25tar = Data_IMS_T_25tar.groupby(['Participant'])['isCorrect'].mean()

Data_IAS_B_25tar = Data[(Data['Probability'] == '0.25cueTarget') & (Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'A') & (Data['SameDifferent'] == 'S')]
Data_IMS_B_25tar = Data[(Data['Probability'] == '0.25cueTarget') & (Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'M') & (Data['SameDifferent'] == 'S')]
acc_IAS_B_25tar = Data_IAS_B_25tar.groupby(['Participant'])['isCorrect'].mean()
acc_IMS_B_25tar = Data_IMS_B_25tar.groupby(['Participant'])['isCorrect'].mean()

# acc, 75% cue target
Data_IAS_T_75tar = Data[(Data['Probability'] == '0.75cueTarget') & (Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'A') & (Data['SameDifferent'] == 'S')]
Data_IMS_T_75tar = Data[(Data['Probability'] == '0.75cueTarget') & (Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'M') & (Data['SameDifferent'] == 'S')]
acc_IAS_T_75tar = Data_IAS_T_75tar.groupby(['Participant'])['isCorrect'].mean()
acc_IMS_T_75tar = Data_IMS_T_75tar.groupby(['Participant'])['isCorrect'].mean()


Data_IAS_B_75tar = Data[(Data['Probability'] == '0.75cueTarget') & (Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'A') & (Data['SameDifferent'] == 'S')]
Data_IMS_B_75tar = Data[(Data['Probability'] == '0.75cueTarget') & (Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'I') & (Data['Alignment'] == 'M') & (Data['SameDifferent'] == 'S')]
acc_IAS_B_75tar = Data_IAS_B_75tar.groupby(['Participant'])['isCorrect'].mean()
acc_IMS_B_75tar = Data_IMS_B_75tar.groupby(['Participant'])['isCorrect'].mean()

"""""""""""""""""""""""""""""""""""""""
#three-way ANOVA: cuedHalf, probability, alignment

"""""""""""""""""""""""""""""""""""""""
# acc, 25% cue target
IAS_IMS_acc_TB_25tar = pd.DataFrame(data = acc_IAS_T_25tar)
IAS_IMS_acc_TB_25tar = IAS_IMS_acc_TB_25tar.append(pd.DataFrame(data = acc_IMS_T_25tar))
IAS_IMS_acc_TB_25tar = IAS_IMS_acc_TB_25tar.append(pd.DataFrame(data = acc_IAS_B_25tar))
IAS_IMS_acc_TB_25tar = IAS_IMS_acc_TB_25tar.append(pd.DataFrame(data = acc_IMS_B_25tar))

IAS_IMS_acc_TB_25tar['CuedHalf'] = ['Top']*64 + ['Bottom']*64
IAS_IMS_acc_TB_25tar['Alignment'] = ['Aligned']*32 + ['MisAligned']*32 + ['Aligned']*32 + ['MisAligned']*32
IAS_IMS_acc_TB_25tar['participant'] = list(range(1,33)) * 4


# acc, 75% cue target
IAS_IMS_acc_TB_75tar = pd.DataFrame(data = acc_IAS_T_75tar)
IAS_IMS_acc_TB_75tar = IAS_IMS_acc_TB_75tar.append(pd.DataFrame(data = acc_IMS_T_75tar))
IAS_IMS_acc_TB_75tar = IAS_IMS_acc_TB_75tar.append(pd.DataFrame(data = acc_IAS_B_75tar))
IAS_IMS_acc_TB_75tar = IAS_IMS_acc_TB_75tar.append(pd.DataFrame(data = acc_IMS_B_75tar))

IAS_IMS_acc_TB_75tar['CuedHalf'] = ['Top']*64 + ['Bottom']*64
IAS_IMS_acc_TB_75tar['Alignment'] = ['Aligned']*32 + ['MisAligned']*32 + ['Aligned']*32 + ['MisAligned']*32
IAS_IMS_acc_TB_75tar['participant'] = list(range(1,33)) * 4

# combined acc data
IAS_IMS_acc_TB = pd.DataFrame(data = IAS_IMS_acc_TB_25tar)
IAS_IMS_acc_TB = IAS_IMS_acc_TB.append(pd.DataFrame(data = IAS_IMS_acc_TB_75tar))
IAS_IMS_acc_TB['Probability'] = ['25cueTarget']*128 + ['75cueTarget']*128 

# ANOVA 
pandas2ri.activate()
r_IAS_IMS_acc = pandas2ri.py2ri(IAS_IMS_acc_TB)

model = afex.aov_ez('participant', 'isCorrect', r_IAS_IMS_acc, within = ['Probability','CuedHalf', 'Alignment'])
print(model)
# Main effect of probability, alignment
# Interaction bteween cuedhalf and alignment; no three-way interaction

# main effect
main_prob = emmeans.emmeans(model, 'Probability', contr = 'pairwise', adjust = 'bonferroni')
print(main_prob)

main_align = emmeans.emmeans(model, 'Alignment', contr = 'pairwise', adjust = 'bonferroni')
print(main_align)

# simple effect
inter_cuedhalf_align = emmeans.emmeans(model, 'Alignment', by = 'CuedHalf', contr = 'pairwise', adjust = 'bonferroni')
print(inter_cuedhalf_align)

# top target
IAS_IMS_acc_T = IAS_IMS_acc_TB[(IAS_IMS_acc_TB['CuedHalf'] == 'Top')] 
IAS_IMS_acc_T_stats = IAS_IMS_acc_T.groupby(['participant','CuedHalf','Alignment'])['isCorrect'].mean()
IAS_IMS_acc_T_stats0 = pairwise_ttests(dv='isCorrect', within='Alignment',
                            subject='participant', data= IAS_IMS_acc_T_stats.to_frame().reset_index(),
                            effsize='cohen')
# top: acc(aligned) < acc(misaligned), t(31)=-6.08, p<.0001, d = -.6


# bottom target
IAS_IMS_acc_B = IAS_IMS_acc_TB[(IAS_IMS_acc_TB['CuedHalf'] == 'Bottom')] 
IAS_IMS_acc_B_stats = IAS_IMS_acc_B.groupby(['participant','CuedHalf','Alignment'])['isCorrect'].mean()
IAS_IMS_acc_B_stats0 = pairwise_ttests(dv='isCorrect', within='Alignment',
                            subject='participant', data= IAS_IMS_acc_B_stats.to_frame().reset_index(),
                            effsize='cohen')
# top: acc(aligned) < acc(misaligned), t(31)=-.59, p=.56, d = -.05


# descriptive
a = IAS_IMS_acc_TB.groupby(['participant','CuedHalf','Alignment'])['isCorrect'].describe()
a.groupby(['CuedHalf','Alignment'])['mean'].describe()

a = IAS_IMS_acc_TB.groupby(['participant','Probability'])['isCorrect'].describe()
a.groupby(['Probability'])['mean'].describe()


"""""""""""
Plot: two probabilities separately

"""""""'"""
# 25% cue target
means_acc_IAS_TB_25tar = [np.mean(acc_IAS_T_25tar), np.mean(acc_IAS_B_25tar)]; errors_IAS_TB_25tar = [stats.sem(acc_IAS_T_25tar),stats.sem(acc_IAS_B_75tar)]
means_acc_IMS_TB_25tar = [np.mean(acc_IMS_T_25tar), np.mean(acc_IMS_B_25tar)]; errors_IMS_TB_25tar = [stats.sem(acc_IMS_T_25tar),stats.sem(acc_IMS_B_75tar)]

index = np.arange(2)
bar_width = 0.35
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
rects1 = ax.bar(index,means_acc_IAS_TB_25tar, bar_width, yerr = errors_IAS_TB_25tar, label = 'Aligned')
rects1 = ax.bar(index + bar_width,means_acc_IMS_TB_25tar, bar_width, yerr = errors_IMS_TB_25tar, label = 'Misaligned')
ax.set_ylabel('Acurracy')
ax.set_ylim((0,1))
ax.set_title('Incongruent Same Conditions, 25% cue', fontweight="bold", fontsize=24)
ax.set_xticks(index + bar_width/ 2)
ax.set_xticklabels(('Top', 'Bottom'))
ax.legend()
plt.savefig('acc_SameIncongruent a vs. m_cueTB_25tar.tiff')
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(12,10),dpi=100)
plt.title('Incongruent Same, 25% cue', fontweight="bold", fontsize=24)
plt.ylabel('Accuracy')
plt.ylim(0,1)
sns.boxplot(data=[acc_IAS_T_25tar, acc_IMS_T_25tar, acc_IAS_B_25tar, acc_IMS_B_25tar], palette = "Blues", showmeans=True)
sns.swarmplot(data=[acc_IAS_T_25tar, acc_IMS_T_25tar, acc_IAS_B_25tar, acc_IMS_B_25tar], color = ".25")
sns.despine()
plt.xticks([0,1,2,3], ['Aligned Top','Misaligned Top','Aligned Bottom','Misaligned Bottom'])
plt.savefig('acc_SameIncongruent a vs. m_boxplot_cueTB_25tar.tiff')
plt.show()

# 75% cue target
means_acc_IAS_TB_75tar = [np.mean(acc_IAS_T_75tar), np.mean(acc_IAS_B_75tar)]; errors_IAS_TB_75tar = [stats.sem(acc_IAS_T_75tar),stats.sem(acc_IAS_B_75tar)]
means_acc_IMS_TB_75tar = [np.mean(acc_IMS_T_75tar), np.mean(acc_IMS_B_75tar)]; errors_IMS_TB_75tar = [stats.sem(acc_IMS_T_75tar),stats.sem(acc_IMS_B_75tar)]

index = np.arange(2)
bar_width = 0.35
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
rects1 = ax.bar(index,means_acc_IAS_TB_75tar, bar_width, yerr = errors_IAS_TB_75tar, label = 'Aligned')
rects1 = ax.bar(index + bar_width,means_acc_IMS_TB_75tar, bar_width, yerr = errors_IMS_TB_75tar, label = 'Misaligned')
ax.set_ylabel('Acurracy')
ax.set_ylim((0,1))
ax.set_title('Incongruent Same Conditions, 75% cue', fontweight="bold", fontsize=24)
ax.set_xticks(index + bar_width/ 2)
ax.set_xticklabels(('Top', 'Bottom'))
ax.legend()
plt.savefig('acc_SameIncongruent a vs. m_cueTB_75tar.tiff')
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(12,10),dpi=100)
plt.title('Incongruent Same, 75% cue', fontweight="bold", fontsize=24)
plt.ylabel('Accuracy')
plt.ylim(0,1)
sns.boxplot(data=[acc_IAS_T_75tar, acc_IMS_T_75tar, acc_IAS_B_75tar, acc_IMS_B_75tar], palette = "Blues", showmeans=True)
sns.swarmplot(data=[acc_IAS_T_75tar, acc_IMS_T_75tar, acc_IAS_B_75tar, acc_IMS_B_75tar], color = ".25")
sns.despine()
plt.xticks([0,1,2,3], ['Aligned Top','Misaligned Top','Aligned Bottom','Misaligned Bottom'])
plt.savefig('acc_SameIncongruent a vs. m_boxplot_cueTB_75tar.tiff')
plt.show()


"""""""""""""""""""""""""""""""""""""""
Complementary design (part-whole like): same congruent

"""""""""""""""""""""""""""""""""""""""
"""""""""""
#descriptive

"""""""'"""

# acc, 25% cue target
Data_CAS_T_25tar = Data[(Data['Probability'] == '0.25cueTarget') & (Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'A') & (Data['SameDifferent'] == 'S')]
Data_CMS_T_25tar = Data[(Data['Probability'] == '0.25cueTarget') & (Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'M') & (Data['SameDifferent'] == 'S')]
acc_CAS_T_25tar = Data_CAS_T_25tar.groupby(['Participant'])['isCorrect'].mean()
acc_CMS_T_25tar = Data_CMS_T_25tar.groupby(['Participant'])['isCorrect'].mean()

Data_CAS_B_25tar = Data[(Data['Probability'] == '0.25cueTarget') & (Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'A') & (Data['SameDifferent'] == 'S')]
Data_CMS_B_25tar = Data[(Data['Probability'] == '0.25cueTarget') & (Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'M') & (Data['SameDifferent'] == 'S')]
acc_CAS_B_25tar = Data_CAS_B_25tar.groupby(['Participant'])['isCorrect'].mean()
acc_CMS_B_25tar = Data_CMS_B_25tar.groupby(['Participant'])['isCorrect'].mean()

# acc, 75% cue target
Data_CAS_T_75tar = Data[(Data['Probability'] == '0.75cueTarget') & (Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'A') & (Data['SameDifferent'] == 'S')]
Data_CMS_T_75tar = Data[(Data['Probability'] == '0.75cueTarget') & (Data['CuedHalf'] == 'T') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'M') & (Data['SameDifferent'] == 'S')]
acc_CAS_T_75tar = Data_CAS_T_75tar.groupby(['Participant'])['isCorrect'].mean()
acc_CMS_T_75tar = Data_CMS_T_75tar.groupby(['Participant'])['isCorrect'].mean()

Data_CAS_B_75tar = Data[(Data['Probability'] == '0.75cueTarget') & (Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'A') & (Data['SameDifferent'] == 'S')]
Data_CMS_B_75tar = Data[(Data['Probability'] == '0.75cueTarget') & (Data['CuedHalf'] == 'B') & (Data['Congruency'] == 'C') & (Data['Alignment'] == 'M') & (Data['SameDifferent'] == 'S')]
acc_CAS_B_75tar = Data_CAS_B_75tar.groupby(['Participant'])['isCorrect'].mean()
acc_CMS_B_75tar = Data_CMS_B_75tar.groupby(['Participant'])['isCorrect'].mean()


"""""""""""""""""""""""""""""""""""""""
#three-way ANOVA: cuedHalf, probability, alignment

"""""""""""""""""""""""""""""""""""""""
# acc, 25% cue target
CAS_CMS_acc_TB_25tar = pd.DataFrame(data = acc_CAS_T_25tar)
CAS_CMS_acc_TB_25tar = CAS_CMS_acc_TB_25tar.append(pd.DataFrame(data = acc_CMS_T_25tar))
CAS_CMS_acc_TB_25tar = CAS_CMS_acc_TB_25tar.append(pd.DataFrame(data = acc_CAS_B_25tar))
CAS_CMS_acc_TB_25tar = CAS_CMS_acc_TB_25tar.append(pd.DataFrame(data = acc_CMS_B_25tar))

CAS_CMS_acc_TB_25tar['CuedHalf'] = ['Top']*64 + ['Bottom']*64
CAS_CMS_acc_TB_25tar['Alignment'] = ['Aligned']*32 + ['MisAligned']*32 + ['Aligned']*32 + ['MisAligned']*32
CAS_CMS_acc_TB_25tar['participant'] = list(range(1,33)) * 4

# acc, 75% cue target
CAS_CMS_acc_TB_75tar = pd.DataFrame(data = acc_CAS_T_75tar)
CAS_CMS_acc_TB_75tar = CAS_CMS_acc_TB_75tar.append(pd.DataFrame(data = acc_CMS_T_75tar))
CAS_CMS_acc_TB_75tar = CAS_CMS_acc_TB_75tar.append(pd.DataFrame(data = acc_CAS_B_75tar))
CAS_CMS_acc_TB_75tar = CAS_CMS_acc_TB_75tar.append(pd.DataFrame(data = acc_CMS_B_75tar))

CAS_CMS_acc_TB_75tar['CuedHalf'] = ['Top']*64 + ['Bottom']*64
CAS_CMS_acc_TB_75tar['Alignment'] = ['Aligned']*32 + ['MisAligned']*32 + ['Aligned']*32 + ['MisAligned']*32
CAS_CMS_acc_TB_75tar['participant'] = list(range(1,33)) * 4

# combined acc data
CAS_CMS_acc_TB = pd.DataFrame(data = CAS_CMS_acc_TB_25tar)
CAS_CMS_acc_TB = CAS_CMS_acc_TB.append(pd.DataFrame(data = CAS_CMS_acc_TB_75tar))
CAS_CMS_acc_TB['Probability'] = ['25cueTarget']*128 + ['75cueTarget']*128 

# ANOVA 
pandas2ri.activate()
r_CAS_CMS_acc = pandas2ri.py2ri(CAS_CMS_acc_TB)

model = afex.aov_ez('participant', 'isCorrect', r_CAS_CMS_acc, within = ['Probability','CuedHalf', 'Alignment'])
print(model)
# Main effect of probability, alignment
# No interaction bteween cuedhalf and alignment; no three-way interaction

# main effect
main_prob = emmeans.emmeans(model, 'Probability', contr = 'pairwise', adjust = 'bonferroni')
print(main_prob)

main_align = emmeans.emmeans(model, 'Alignment', contr = 'pairwise', adjust = 'bonferroni')
print(main_align)

# simple effect
#inter_cuedhalf_align = emmeans.emmeans(model, 'Alignment', by = 'CuedHalf', contr = 'pairwise', adjust = 'bonferroni')
#print(inter_cuedhalf_align)

# top target
CAS_CMS_acc_T = CAS_CMS_acc_TB[(CAS_CMS_acc_TB['CuedHalf'] == 'Top')] 
CAS_CMS_acc_T_stats = CAS_CMS_acc_T.groupby(['participant','CuedHalf','Alignment'])['isCorrect'].mean()
CAS_CMS_acc_T_stats0 = pairwise_ttests(dv='isCorrect', within='Alignment',
                            subject='participant', data= CAS_CMS_acc_T_stats.to_frame().reset_index(),
                            effsize='cohen')
# top: acc(aligned) > acc(misaligned), t(31)=2.20, p=.035, d = .24


# bottom target
CAS_CMS_acc_B = CAS_CMS_acc_TB[(CAS_CMS_acc_TB['CuedHalf'] == 'Bottom')] 
CAS_CMS_acc_B_stats = CAS_CMS_acc_B.groupby(['participant','CuedHalf','Alignment'])['isCorrect'].mean()
CAS_CMS_acc_B_stats0 = pairwise_ttests(dv='isCorrect', within='Alignment',
                            subject='participant', data= CAS_CMS_acc_B_stats.to_frame().reset_index(),
                            effsize='cohen')
# top: acc(aligned) > acc(misaligned), t(31)=2.36, p=.025, d = .24


# descriptive
a = CAS_CMS_acc_TB.groupby(['participant','CuedHalf','Alignment'])['isCorrect'].describe()
a.groupby(['CuedHalf','Alignment'])['mean'].describe()

a = CAS_CMS_acc_TB.groupby(['participant','Probability'])['isCorrect'].describe()
a.groupby(['Probability'])['mean'].describe()


"""""""""""
Plot: two probabilities separately

"""""""'"""
# 25% cue target
means_acc_CAS_TB_25tar = [np.mean(acc_CAS_T_25tar), np.mean(acc_CAS_B_25tar)]; errors_CAS_TB_25tar = [stats.sem(acc_CAS_T_25tar),stats.sem(acc_CAS_B_25tar)]
means_acc_CMS_TB_25tar = [np.mean(acc_CMS_T_25tar), np.mean(acc_CMS_B_25tar)]; errors_CMS_TB_25tar = [stats.sem(acc_CMS_T_25tar),stats.sem(acc_CMS_B_25tar)]

index = np.arange(2)
bar_width = 0.35
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
rects1 = ax.bar(index,means_acc_CAS_TB_25tar, bar_width, yerr = errors_CAS_TB_25tar, label = 'Aligned')
rects1 = ax.bar(index + bar_width,means_acc_CMS_TB_25tar, bar_width, yerr = errors_CMS_TB_25tar, label = 'Misaligned')
ax.set_ylabel('Acurracy')
ax.set_ylim((0,1))
ax.set_title('Congruent Same Conditions, 25% cue', fontweight="bold", fontsize=24)
ax.set_xticks(index + bar_width/ 2)
ax.set_xticklabels(('Top', 'Bottom'))
ax.legend()
plt.savefig('acc_SameCongruent a vs. m_cueTB_25tar.tiff')
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(12,10),dpi=100)
plt.title('Congruent Same Conditions, 25% cue', fontweight="bold", fontsize=24)
plt.ylabel('Accuracy')
plt.ylim(0,1)
sns.boxplot(data=[acc_CAS_T_25tar, acc_CMS_T_25tar, acc_CAS_B_25tar, acc_CMS_B_25tar], palette = "Blues", showmeans=True)
sns.swarmplot(data=[acc_CAS_T_25tar, acc_CMS_T_25tar, acc_CAS_B_25tar, acc_CMS_B_25tar], color = ".25")
sns.despine()
plt.xticks([0,1,2,3], ['Aligned Top','Misaligned Top','Aligned Bottom','Misaligned Bottom'])
plt.savefig('acc_SameCongruent a vs. m_boxplot_cueTB_25tar.tiff')
plt.show()


# 75% top cue
means_acc_CAS_TB_75tar = [np.mean(acc_CAS_T_75tar), np.mean(acc_CAS_B_75tar)]; errors_CAS_TB_75tar = [stats.sem(acc_CAS_T_75tar),stats.sem(acc_CAS_B_75tar)]
means_acc_CMS_TB_75tar = [np.mean(acc_CMS_T_75tar), np.mean(acc_CMS_B_75tar)]; errors_CMS_TB_75tar = [stats.sem(acc_CMS_T_75tar),stats.sem(acc_CMS_B_75tar)]

index = np.arange(2)
bar_width = 0.35
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
rects1 = ax.bar(index,means_acc_CAS_TB_75tar, bar_width, yerr = errors_CAS_TB_75tar, label = 'Aligned')
rects1 = ax.bar(index + bar_width,means_acc_CMS_TB_75tar, bar_width, yerr = errors_CMS_TB_75tar, label = 'Misaligned')
ax.set_ylabel('Acurracy')
ax.set_ylim((0,1))
ax.set_title('Congruent Same Conditions, 75% cue', fontweight="bold", fontsize=24)
ax.set_xticks(index + bar_width/ 2)
ax.set_xticklabels(('Top', 'Bottom'))
ax.legend()
plt.savefig('acc_SameCongruent a vs. m_cueTB_75tar.tiff')
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(12,10),dpi=100)
plt.title('Congruent Same Conditions, 75% cue', fontweight="bold", fontsize=24)
plt.ylabel('Accuracy')
plt.ylim(0,1)
sns.boxplot(data=[acc_CAS_T_75tar, acc_CMS_T_75tar, acc_CAS_B_75tar, acc_CMS_B_75tar], palette = "Blues", showmeans=True)
sns.swarmplot(data=[acc_CAS_T_75tar, acc_CMS_T_75tar, acc_CAS_B_75tar, acc_CMS_B_75tar], color = ".25")
sns.despine()
plt.xticks([0,1,2,3], ['Aligned Top','Misaligned Top','Aligned Bottom','Misaligned Bottom'])
plt.savefig('acc_SameCongruent a vs. m_boxplot_cueTB_75tar.tiff')
plt.show()


"""
plot: traditional design + part-whole (complementary) design
same incongruent condition + same congruent condition
Two conditions in one plot

"""

# 25% cue target
barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Top','Bottom','Top','Bottom']
values1 = [np.mean(acc_CAS_T_25tar), np.mean(acc_CAS_B_25tar), np.mean(acc_IAS_T_25tar), np.mean(acc_IAS_B_25tar)]
errors1 = [stats.sem(acc_CAS_T_25tar),stats.sem(acc_CAS_B_25tar), stats.sem(acc_IAS_T_25tar),stats.sem(acc_IAS_B_25tar)]
values2 = [np.mean(acc_CMS_T_25tar), np.mean(acc_CMS_B_25tar), np.mean(acc_IMS_T_25tar), np.mean(acc_IMS_B_25tar)]
errors2 = [stats.sem(acc_CMS_T_25tar),stats.sem(acc_CMS_B_25tar), stats.sem(acc_IMS_T_25tar),stats.sem(acc_IMS_B_25tar)]

fig,ax = plt.subplots(figsize=(8,6),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, color='lightgrey', edgecolor='white', label = 'Aligned')
plt.bar(r2, values2, yerr=errors2, width=barWidth, color='black', edgecolor='white', label = 'Misaligned')
plt.ylim((0,1))
plt.ylabel('Accuracy')
plt.xticks([(r + barWidth/2) for r in range(4)], names)
plt.legend(loc = "upper right", bbox_to_anchor=(1.08, 1.09), labelspacing=0.1)  # bbox_to_anchor indicates the position of the right corner of the legend relative to the whole box
plt.text(1.75, 1.2, "25% cue", size=20,
         ha="center", va="center")
plt.text(0.75, 1.1, "Same congruent", size=20,
         ha="center", va="center")
plt.text(2.75, 1.1, "Same incongruent", size=20,
         ha="center", va="center")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('Traditional + Complementary_acc_25tar.tiff')
plt.show()


# 75% cue target
barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Top','Bottom','Top','Bottom']
values1 = [np.mean(acc_CAS_T_75tar), np.mean(acc_CAS_B_75tar), np.mean(acc_IAS_T_75tar), np.mean(acc_IAS_B_75tar)]
errors1 = [stats.sem(acc_CAS_T_75tar),stats.sem(acc_CAS_B_75tar), stats.sem(acc_IAS_T_75tar),stats.sem(acc_IAS_B_75tar)]
values2 = [np.mean(acc_CMS_T_75tar), np.mean(acc_CMS_B_75tar), np.mean(acc_IMS_T_75tar), np.mean(acc_IMS_B_75tar)]
errors2 = [stats.sem(acc_CMS_T_75tar),stats.sem(acc_CMS_B_75tar), stats.sem(acc_IMS_T_75tar),stats.sem(acc_IMS_B_75tar)]

fig,ax = plt.subplots(figsize=(8,6),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, color='lightgrey', edgecolor='white', label = 'Aligned')
plt.bar(r2, values2, yerr=errors2, width=barWidth, color='black', edgecolor='white', label = 'Misaligned')
plt.ylim((0,1))
plt.ylabel('Accuracy')
plt.xticks([(r + barWidth/2) for r in range(4)], names)
plt.legend(loc = "upper right", bbox_to_anchor=(1.08, 1.09), labelspacing=0.1)  # bbox_to_anchor indicates the position of the right corner of the legend relative to the whole box
plt.text(1.75, 1.2, "75% cue", size=20,
         ha="center", va="center")
plt.text(0.75, 1.1, "Same congruent", size=20,
         ha="center", va="center")
plt.text(2.75, 1.1, "Same incongruent", size=20,
         ha="center", va="center")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('Traditional + Complementary_acc_75tar.tiff')
plt.show()


""""""""""
d prime
Top, bottom, separately

Adjust the extreme values.
Rates of 0 are replaced with 0.5/n, and rates of 1 are replaced with (n-0.5)/n,
where n is the number of signal or noise trials (Macmillan & Kaplan, 1985, see Stanisla & Todorov, 1999)

n = 40 in the present study 

""""""""""
n = 40
def SDT (hits, fas):
    for i in range(32):
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


"""""
25% cue target
"""""
# top
hit_CA_T_25tar = Data_CA_T_25tar[(Data_CA_T_25tar['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_CA_T_25tar = 1 - Data_CA_T_25tar[(Data_CA_T_25tar['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
#d_CA_T = Z(hit_CA_T) - Z(fa_CA_T)
d_CA_T_25tar = SDT(hit_CA_T_25tar.tolist(),fa_CA_T_25tar.tolist())

hit_CM_T_25tar = Data_CM_T_25tar[(Data_CM_T_25tar['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_CM_T_25tar = 1- Data_CM_T_25tar[(Data_CM_T_25tar['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_CM_T_25tar = SDT(hit_CM_T_25tar.tolist(),fa_CM_T_25tar.tolist())

hit_IA_T_25tar = Data_IA_T_25tar[(Data_IA_T_25tar['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_IA_T_25tar = 1 - Data_IA_T_25tar[(Data_IA_T_25tar['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_IA_T_25tar = SDT(hit_IA_T_25tar.tolist(),fa_IA_T_25tar.tolist())

hit_IM_T_25tar = Data_IM_T_25tar[(Data_IM_T_25tar['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_IM_T_25tar = 1 - Data_IM_T_25tar[(Data_IM_T_25tar['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_IM_T_25tar = SDT(hit_IM_T_25tar.tolist(),fa_IM_T_25tar.tolist())

# bottom
hit_CA_B_25tar = Data_CA_B_25tar[(Data_CA_B_25tar['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean() 
fa_CA_B_25tar = 1 - Data_CA_B_25tar[(Data_CA_B_25tar['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_CA_B_25tar = SDT(hit_CA_B_25tar.tolist(),fa_CA_B_25tar.tolist())

hit_CM_B_25tar = Data_CM_B_25tar[(Data_CM_B_25tar['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_CM_B_25tar = 1- Data_CM_B_25tar[(Data_CM_B_25tar['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_CM_B_25tar = SDT(hit_CM_B_25tar.tolist(),fa_CM_B_25tar.tolist())

hit_IA_B_25tar = Data_IA_B_25tar[(Data_IA_B_25tar['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_IA_B_25tar = 1 - Data_IA_B_25tar[(Data_IA_B_25tar['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_IA_B_25tar = SDT(hit_IA_B_25tar.tolist(),fa_IA_B_25tar.tolist())

hit_IM_B_25tar = Data_IM_B_25tar[(Data_IM_B_25tar['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_IM_B_25tar = 1 - Data_IM_B_25tar[(Data_IM_B_25tar['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_IM_B_25tar = SDT(hit_IM_B_25tar.tolist(),fa_IM_B_25tar.tolist())

## one-sample t test, whether d prime is different from zero
# in all conditions, the d prime was significant larger than zero, except p8!!!
#t1, p1 = stats.ttest_1samp(d_CA_T_25tar,0.0)
#t2, p2 = stats.ttest_1samp(d_CM_T_25tar,0.0) 
#t3, p3 = stats.ttest_1samp(d_IA_T_25tar,0.0)  
#t4, p4 = stats.ttest_1samp(d_IM_T_25tar,0.0)
#t5, p5 = stats.ttest_1samp(d_CA_B_25tar,0.0)
#t6, p6 = stats.ttest_1samp(d_CM_B_25tar,0.0) 
#t7, p7 = stats.ttest_1samp(d_IA_B_25tar,0.0) 
#t8, p8 = stats.ttest_1samp(d_IM_B_25tar,0.0) 

e1 = ttest(d_CA_T_25tar,0.0)
e2 = ttest(d_CM_T_25tar,0.0)
e3 = ttest(d_IA_T_25tar,0.0)  # incongruent aligned bottom, t(31) = .28, p3 = .78, d=.05
e4 = ttest(d_IM_T_25tar,0.0)
[e1['p-val'],e2['p-val'],e3['p-val'],e4['p-val']]
[e1['cohen-d'],e2['cohen-d'],e3['cohen-d'],e4['cohen-d']]

e5 = ttest(d_CA_B_25tar,0.0)
e6 = ttest(d_CM_B_25tar,0.0)
e7 = ttest(d_IA_B_25tar,0.0)   # incongruent aligned bottom, t(31) = .73, p7 = .47
e8 = ttest(d_IM_B_25tar,0.0)   # incongruent misaligned bottom, t(31) = .30, p8 = .77
[e5['p-val'],e6['p-val'],e7['p-val'],e8['p-val']]
[e5['cohen-d'],e6['cohen-d'],e7['cohen-d'],e8['cohen-d']]


dprime_25tar = pd.DataFrame(
    {'Congruent_Aligned_Top': d_CA_T_25tar,
     'Congruent_Misligned_Top': d_CM_T_25tar,
     'Incongruent_Aligned_Top': d_IA_T_25tar,
     'Incongruent_Misligned_Top': d_IM_T_25tar,
     'Congruent_Aligned_Bottom': d_CA_B_25tar,
     'Congruent_Misligned_Bottom': d_CM_B_25tar,
     'Incongruent_Aligned_Bottom': d_IA_B_25tar,
     'Incongruent_Misligned_Bottom': d_IM_B_25tar
    })

dprime25tar_long = pd.melt(dprime_25tar, value_name='d')
dprime25tar_long['CuedHalf'] = ['Top']*128 + ['Bottom']*128
dprime25tar_long['Alignment'] = (['Aligned']*32 + ['MisAligned']*32) * 4 
dprime25tar_long['Congruency'] = (['Congruent']*64 + ['Incongruent']*64) * 2
dprime25tar_long['participant'] = list(range(1,33)) * 8
dprime25tar_long['ProbabilityCue'] = ['25cueTarget']*256


"""""
75% cue target
"""""
# top
hit_CA_T_75tar = Data_CA_T_75tar[(Data_CA_T_75tar['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_CA_T_75tar = 1 - Data_CA_T_75tar[(Data_CA_T_75tar['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
#d_CA_T = Z(hit_CA_T) - Z(fa_CA_T)
d_CA_T_75tar = SDT(hit_CA_T_75tar.tolist(),fa_CA_T_75tar.tolist())

hit_CM_T_75tar = Data_CM_T_75tar[(Data_CM_T_75tar['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_CM_T_75tar = 1- Data_CM_T_75tar[(Data_CM_T_75tar['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_CM_T_75tar = SDT(hit_CM_T_75tar.tolist(),fa_CM_T_75tar.tolist())

hit_IA_T_75tar = Data_IA_T_75tar[(Data_IA_T_75tar['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_IA_T_75tar = 1 - Data_IA_T_75tar[(Data_IA_T_75tar['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_IA_T_75tar = SDT(hit_IA_T_75tar.tolist(),fa_IA_T_75tar.tolist())

hit_IM_T_75tar = Data_IM_T_75tar[(Data_IM_T_75tar['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_IM_T_75tar = 1 - Data_IM_T_75tar[(Data_IM_T_75tar['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_IM_T_75tar = SDT(hit_IM_T_75tar.tolist(),fa_IM_T_75tar.tolist())

# bottom
hit_CA_B_75tar = Data_CA_B_75tar[(Data_CA_B_75tar['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean() 
fa_CA_B_75tar = 1 - Data_CA_B_75tar[(Data_CA_B_75tar['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_CA_B_75tar = SDT(hit_CA_B_75tar.tolist(),fa_CA_B_75tar.tolist())

hit_CM_B_75tar = Data_CM_B_75tar[(Data_CM_B_75tar['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_CM_B_75tar = 1- Data_CM_B_75tar[(Data_CM_B_75tar['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_CM_B_75tar = SDT(hit_CM_B_75tar.tolist(),fa_CM_B_75tar.tolist())

hit_IA_B_75tar = Data_IA_B_75tar[(Data_IA_B_75tar['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_IA_B_75tar = 1 - Data_IA_B_75tar[(Data_IA_B_75tar['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_IA_B_75tar = SDT(hit_IA_B_75tar.tolist(),fa_IA_B_75tar.tolist())

hit_IM_B_75tar = Data_IM_B_75tar[(Data_IM_B_75tar['SameDifferent'] == 'S')].groupby(['Participant'])['isCorrect'].mean()
fa_IM_B_75tar = 1 - Data_IM_B_75tar[(Data_IM_B_75tar['SameDifferent'] == 'D')].groupby(['Participant'])['isCorrect'].mean()
d_IM_B_75tar = SDT(hit_IM_B_75tar.tolist(),fa_IM_B_75tar.tolist())

## one-sample t test, whether d prime is different from zero
# in all conditions, the d prime was significant larger than zero, except p8!!!
#t1, p1 = stats.ttest_1samp(d_CA_T_75tar,0.0)
#t2, p2 = stats.ttest_1samp(d_CM_T_75tar,0.0) 
#t3, p3 = stats.ttest_1samp(d_IA_T_75tar,0.0)
#t4, p4 = stats.ttest_1samp(d_IM_T_75tar,0.0)
#t5, p5 = stats.ttest_1samp(d_CA_B_75tar,0.0)
#t6, p6 = stats.ttest_1samp(d_CM_B_75tar,0.0) 
#t7, p7 = stats.ttest_1samp(d_IA_B_75tar,0.0)
#t8, p8 = stats.ttest_1samp(d_IM_B_75tar,0.0) 

e1 = ttest(d_CA_T_75tar,0.0)
e2 = ttest(d_CM_T_75tar,0.0)
e3 = ttest(d_IA_T_75tar,0.0) 
e4 = ttest(d_IM_T_75tar,0.0)
[e1['p-val'],e2['p-val'],e3['p-val'],e4['p-val']]
[e1['cohen-d'],e2['cohen-d'],e3['cohen-d'],e4['cohen-d']]

e5 = ttest(d_CA_B_75tar,0.0)
e6 = ttest(d_CM_B_75tar,0.0)
e7 = ttest(d_IA_B_75tar,0.0)   
e8 = ttest(d_IM_B_75tar,0.0)   
[e5['p-val'],e6['p-val'],e7['p-val'],e8['p-val']]
[e5['cohen-d'],e6['cohen-d'],e7['cohen-d'],e8['cohen-d']]

dprime_75tar = pd.DataFrame(
    {'Congruent_Aligned_Top': d_CA_T_75tar,
     'Congruent_Misligned_Top': d_CM_T_75tar,
     'Incongruent_Aligned_Top': d_IA_T_75tar,
     'Incongruent_Misligned_Top': d_IM_T_75tar,
     'Congruent_Aligned_Bottom': d_CA_B_75tar,
     'Congruent_Misligned_Bottom': d_CM_B_75tar,
     'Incongruent_Aligned_Bottom': d_IA_B_75tar,
     'Incongruent_Misligned_Bottom': d_IM_B_75tar
    })

dprime75tar_long = pd.melt(dprime_75tar, value_name='d')
dprime75tar_long['CuedHalf'] = ['Top']*128 + ['Bottom']*128
dprime75tar_long['Alignment'] = (['Aligned']*32 + ['MisAligned']*32) * 4 
dprime75tar_long['Congruency'] = (['Congruent']*64 + ['Incongruent']*64) * 2
dprime75tar_long['participant'] = list(range(1,33)) * 8
dprime75tar_long['ProbabilityCue'] = ['75cueTarget']*256


dprime_long = dprime25tar_long
dprime_long = dprime_long.append(pd.DataFrame(data = dprime75tar_long))


# anova
# four way: cue part, cue probability, congruency, alignment
pandas2ri.activate()
r_dprime = pandas2ri.py2ri(dprime_long)

model = afex.aov_ez('participant', 'd', r_dprime, within = ['ProbabilityCue','CuedHalf', 'Alignment','Congruency'])
print(model)
# main effect of probability, and congruency
# three-way interaction between cuedhalf, congruency and alignment
# two-way interaction between congruency and alignment, probability and congruency, cued half and alignment
# no four-way interaction

# main effect
main_prob = emmeans.emmeans(model, 'ProbabilityCue', contr = 'pairwise', adjust = 'bonferroni')
print(main_prob)

main_congr = emmeans.emmeans(model, 'Congruency', contr = 'pairwise', adjust = 'bonferroni')
print(main_congr)

# simple effect
inter_cuedhalf_align = emmeans.emmeans(model, 'Alignment', by = 'CuedHalf', contr = 'pairwise', adjust = 'bonferroni')
print(inter_cuedhalf_align)

inter_congr_align = emmeans.emmeans(model, 'Alignment', by = 'Congruency', contr = 'pairwise', adjust = 'bonferroni')
print(inter_congr_align)

inter_congr_prob = emmeans.emmeans(model, 'ProbabilityCue', by = 'Congruency', contr = 'pairwise', adjust = 'bonferroni')
print(inter_congr_prob)

# top target
d_T = dprime_long[(dprime_long['CuedHalf'] == 'Top')] 
r_dprime_T = pandas2ri.py2ri(d_T)
model_dprimeT = afex.aov_ez('participant', 'd', r_dprime_T, within = ['Alignment','Congruency'])
print(model_dprimeT) 
# interaction between alignment and congruency
inter_congr_align = emmeans.emmeans(model_dprimeT, 'Alignment', by = 'Congruency', contr = 'pairwise', adjust = 'bonferroni')
print(inter_congr_align)

inter_congr_align = emmeans.emmeans(model_dprimeT, 'Congruency', by = 'Alignment', contr = 'pairwise', adjust = 'bonferroni')
print(inter_congr_align)


# bottom target
d_B = dprime_long[(dprime_long['CuedHalf'] == 'Bottom')] 
r_dprime_B = pandas2ri.py2ri(d_B)
model_dprimeB = afex.aov_ez('participant', 'd', r_dprime_B, within = ['Alignment','Congruency'])
print(model_dprimeB) 
# interaction between alignment and congruency
inter_congr_align2 = emmeans.emmeans(model_dprimeB, 'Alignment', by = 'Congruency', contr = 'pairwise', adjust = 'bonferroni')
print(inter_congr_align2)

inter_congr_align2 = emmeans.emmeans(model_dprimeB, 'Congruency', by = 'Alignment', contr = 'pairwise', adjust = 'bonferroni')
print(inter_congr_align2)

# descriptive    


a = dprime_long.groupby(['participant','ProbabilityCue'])['d'].describe()    
a.groupby(['ProbabilityCue'])['mean'].describe() 

"""""
plot
"""""

# 25% cue target, top and bottom in one plot, using subplot function

barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Aligned','Misaligned','Aligned','Misaligned']
values1 = [np.mean(d_CA_T_25tar), np.mean(d_CM_T_25tar), np.mean(d_CA_B_25tar), np.mean(d_CM_B_25tar)]
errors1 = [stats.sem(d_CA_T_25tar),stats.sem(d_CM_T_25tar), stats.sem(d_CA_B_25tar),stats.sem(d_CM_B_25tar)]
values2 = [np.mean(d_IA_T_25tar), np.mean(d_IM_T_25tar), np.mean(d_IA_B_25tar), np.mean(d_IM_B_25tar)]
errors2 = [stats.sem(d_IA_T_25tar),stats.sem(d_IM_T_25tar), stats.sem(d_IA_B_25tar),stats.sem(d_IM_B_25tar)]

fig,ax = plt.subplots(figsize=(8,6),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, color='lightgrey', edgecolor='white', label = 'Congruent')
plt.bar(r2, values2, yerr=errors2, width=barWidth, color='black', edgecolor='white', label = 'Incongruent')
plt.ylim((0,3.5))
plt.ylabel('d prime')
plt.xticks([(r + barWidth/2) for r in range(4)], names)
plt.legend(loc = "upper right", bbox_to_anchor=(1.08, 1.05), labelspacing=0.1)  # bbox_to_anchor indicates the position of the right corner of the legend relative to the whole box
plt.text(1.75, 3.9, "25% Cue", size=20,
         ha="center", va="center")
plt.text(0.75, 3.75, "Top", size=20,
         ha="center", va="center")
plt.text(2.75, 3.75, "Bottom", size=20,
         ha="center", va="center")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('Congruency by alignment_dprime25tar.tiff')
plt.show()


# boxplot
sns.set_style("whitegrid")
plt.figure(figsize=(12,10),dpi=100)
plt.title('Top, 25% cue', fontweight="bold", fontsize=24)
plt.ylabel('d prime')
plt.ylim(-2,4)
sns.boxplot(data=[d_CA_T_25tar, d_CM_T_25tar, d_IA_T_25tar, d_IM_T_25tar], palette = "Blues", showmeans=True)
sns.swarmplot(data=[d_CA_T_25tar, d_CM_T_25tar, d_IA_T_25tar, d_IM_T_25tar], color = ".25")
sns.despine()
plt.xticks([0,1,2,3], ['CA','CM','IA','IM'])
plt.savefig('d_congruency x alignment_boxplot_cueT_25tar.tiff')
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(12,10),dpi=100)
plt.title('Bottom, 25% cue', fontweight="bold", fontsize=24)
plt.ylabel('d prime')
plt.ylim(-2,4)
sns.boxplot(data=[d_CA_B_25tar, d_CM_B_25tar, d_IA_B_25tar, d_IM_B_25tar], palette = "Blues", showmeans=True)
sns.swarmplot(data=[d_CA_B_25tar, d_CM_B_25tar, d_IA_B_25tar, d_IM_B_25tar], color = ".25")
sns.despine()
plt.xticks([0,1,2,3], ['CA','CM','IA','IM'])
plt.savefig('d_congruency x alignment_boxplot_cueB_25tar.tiff')
plt.show()


# 75% cue target, top and bottom in one plot, using subplot function

barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Aligned','Misaligned','Aligned','Misaligned']
values1 = [np.mean(d_CA_T_75tar), np.mean(d_CM_T_75tar), np.mean(d_CA_B_75tar), np.mean(d_CM_B_75tar)]
errors1 = [stats.sem(d_CA_T_75tar),stats.sem(d_CM_T_75tar), stats.sem(d_CA_B_75tar),stats.sem(d_CM_B_75tar)]
values2 = [np.mean(d_IA_T_75tar), np.mean(d_IM_T_75tar), np.mean(d_IA_B_75tar), np.mean(d_IM_B_75tar)]
errors2 = [stats.sem(d_IA_T_75tar),stats.sem(d_IM_T_75tar), stats.sem(d_IA_B_75tar),stats.sem(d_IM_B_75tar)]

fig,ax = plt.subplots(figsize=(8,6),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, color='lightgrey', edgecolor='white', label = 'Congruent')
plt.bar(r2, values2, yerr=errors2, width=barWidth, color='black', edgecolor='white', label = 'Incongruent')
plt.ylim((0,3.5))
plt.ylabel('d prime')
plt.xticks([(r + barWidth/2) for r in range(4)], names)
plt.legend(loc = "upper right", bbox_to_anchor=(1.08, 1.05), labelspacing=0.1)  # bbox_to_anchor indicates the position of the right corner of the legend relative to the whole box
plt.text(1.75, 3.9, "75% Cue", size=20,
         ha="center", va="center")
plt.text(0.75, 3.75, "Top", size=20,
         ha="center", va="center")
plt.text(2.75, 3.75, "Bottom", size=20,
         ha="center", va="center")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('Congruency by alignment_dprime75tar.tiff')
plt.show()

# boxplot
sns.set_style("whitegrid")
plt.figure(figsize=(12,10),dpi=100)
plt.title('Top, 75% cue', fontweight="bold", fontsize=24)
plt.ylabel('d prime')
plt.ylim(-2,4)
sns.boxplot(data=[d_CA_T_75tar, d_CM_T_75tar, d_IA_T_75tar, d_IM_T_75tar], palette = "Blues", showmeans=True)
sns.swarmplot(data=[d_CA_T_75tar, d_CM_T_75tar, d_IA_T_75tar, d_IM_T_75tar], color = ".25")
sns.despine()
plt.xticks([0,1,2,3], ['CA','CM','IA','IM'])
plt.savefig('d_congruency x alignment_boxplot_cueT_75tar.tiff')
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(12,10),dpi=100)
plt.title('Bottom, 75% cue', fontweight="bold", fontsize=24)
plt.ylabel('d prime')
plt.ylim(-2,4)
sns.boxplot(data=[d_CA_B_75tar, d_CM_B_75tar, d_IA_B_75tar, d_IM_B_75tar], palette = "Blues", showmeans=True)
sns.swarmplot(data=[d_CA_B_75tar, d_CM_B_75tar, d_IA_B_75tar, d_IM_B_75tar], color = ".25")
sns.despine()
plt.xticks([0,1,2,3], ['CA','CM','IA','IM'])
plt.savefig('d_congruency x alignment_boxplot_cueB_75tar.tiff')
plt.show()




"""""
RT: response time for correct choices
"""""
## Top

DataT = Data[(Data['CuedHalf'] == 'T')]
DataT_cor = DataT[(DataT['isCorrect'] == 1)]

DataT_cor.groupby(['Congruency', 'Alignment'])['reactionTime'].mean()  

# 25% cue target
DataTcor_CA_25tar = DataT_cor[(DataT_cor['Congruency'] == 'C') & (DataT_cor['Alignment'] == 'A') & (DataT_cor['Probability'] == '0.25cueTarget')]
DataTcor_CM_25tar = DataT_cor[(DataT_cor['Congruency'] == 'C') & (DataT_cor['Alignment'] == 'M') & (DataT_cor['Probability'] == '0.25cueTarget')]
DataTcor_IA_25tar = DataT_cor[(DataT_cor['Congruency'] == 'I') & (DataT_cor['Alignment'] == 'A') & (DataT_cor['Probability'] == '0.25cueTarget')]
DataTcor_IM_25tar = DataT_cor[(DataT_cor['Congruency'] == 'I') & (DataT_cor['Alignment'] == 'M') & (DataT_cor['Probability'] == '0.25cueTarget')]

RTT_CA_25tar = DataTcor_CA_25tar.groupby(['Participant'])['reactionTime'].mean()
RTT_CM_25tar = DataTcor_CM_25tar.groupby(['Participant'])['reactionTime'].mean()
RTT_IA_25tar = DataTcor_IA_25tar.groupby(['Participant'])['reactionTime'].mean()
RTT_IM_25tar = DataTcor_IM_25tar.groupby(['Participant'])['reactionTime'].mean()

# 75% cue target
DataTcor_CA_75tar = DataT_cor[(DataT_cor['Congruency'] == 'C') & (DataT_cor['Alignment'] == 'A') & (DataT_cor['Probability'] == '0.75cueTarget')]
DataTcor_CM_75tar = DataT_cor[(DataT_cor['Congruency'] == 'C') & (DataT_cor['Alignment'] == 'M') & (DataT_cor['Probability'] == '0.75cueTarget')]
DataTcor_IA_75tar = DataT_cor[(DataT_cor['Congruency'] == 'I') & (DataT_cor['Alignment'] == 'A') & (DataT_cor['Probability'] == '0.75cueTarget')]
DataTcor_IM_75tar = DataT_cor[(DataT_cor['Congruency'] == 'I') & (DataT_cor['Alignment'] == 'M') & (DataT_cor['Probability'] == '0.75cueTarget')]

RTT_CA_75tar = DataTcor_CA_75tar.groupby(['Participant'])['reactionTime'].mean()
RTT_CM_75tar = DataTcor_CM_75tar.groupby(['Participant'])['reactionTime'].mean()
RTT_IA_75tar = DataTcor_IA_75tar.groupby(['Participant'])['reactionTime'].mean()
RTT_IM_75tar = DataTcor_IM_75tar.groupby(['Participant'])['reactionTime'].mean()

## Bottom
DataB = Data[(Data['CuedHalf'] == 'B')]
DataB_cor = DataB[(DataB['isCorrect'] == 1)]

DataB_cor.groupby(['Congruency', 'Alignment'])['reactionTime'].mean()  

# 25% cue target
DataBcor_CA_25tar = DataB_cor[(DataB_cor['Congruency'] == 'C') & (DataB_cor['Alignment'] == 'A') & (DataB_cor['Probability'] == '0.25cueTarget')]
DataBcor_CM_25tar = DataB_cor[(DataB_cor['Congruency'] == 'C') & (DataB_cor['Alignment'] == 'M') & (DataB_cor['Probability'] == '0.25cueTarget')]
DataBcor_IA_25tar = DataB_cor[(DataB_cor['Congruency'] == 'I') & (DataB_cor['Alignment'] == 'A') & (DataB_cor['Probability'] == '0.25cueTarget')]
DataBcor_IM_25tar = DataB_cor[(DataB_cor['Congruency'] == 'I') & (DataB_cor['Alignment'] == 'M') & (DataB_cor['Probability'] == '0.25cueTarget')]

RTB_CA_25tar = DataBcor_CA_25tar.groupby(['Participant'])['reactionTime'].mean()
RTB_CM_25tar = DataBcor_CM_25tar.groupby(['Participant'])['reactionTime'].mean()
RTB_IA_25tar = DataBcor_IA_25tar.groupby(['Participant'])['reactionTime'].mean()
RTB_IM_25tar = DataBcor_IM_25tar.groupby(['Participant'])['reactionTime'].mean()

# 75% cue target
DataBcor_CA_75tar = DataB_cor[(DataB_cor['Congruency'] == 'C') & (DataB_cor['Alignment'] == 'A') & (DataB_cor['Probability'] == '0.75cueTarget')]
DataBcor_CM_75tar = DataB_cor[(DataB_cor['Congruency'] == 'C') & (DataB_cor['Alignment'] == 'M') & (DataB_cor['Probability'] == '0.75cueTarget')]
DataBcor_IA_75tar = DataB_cor[(DataB_cor['Congruency'] == 'I') & (DataB_cor['Alignment'] == 'A') & (DataB_cor['Probability'] == '0.75cueTarget')]
DataBcor_IM_75tar = DataB_cor[(DataB_cor['Congruency'] == 'I') & (DataB_cor['Alignment'] == 'M') & (DataB_cor['Probability'] == '0.75cueTarget')]

RTB_CA_75tar = DataBcor_CA_75tar.groupby(['Participant'])['reactionTime'].mean()
RTB_CM_75tar = DataBcor_CM_75tar.groupby(['Participant'])['reactionTime'].mean()
RTB_IA_75tar = DataBcor_IA_75tar.groupby(['Participant'])['reactionTime'].mean()
RTB_IM_75tar = DataBcor_IM_75tar.groupby(['Participant'])['reactionTime'].mean()


# barplot
# two cue probabilities in one plot, top as target
barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Aligned','Misaligned','Aligned','Misaligned']
values1 = [np.mean(RTT_CA_25tar*1000), np.mean(RTT_CM_25tar*1000), np.mean(RTT_CA_75tar*1000), np.mean(RTT_CM_75tar*1000)]
errors1 = [stats.sem(RTT_CA_25tar*1000),stats.sem(RTT_CM_25tar*1000), stats.sem(RTT_CA_75tar*1000),stats.sem(RTT_CM_75tar*1000)]
values2 = [np.mean(RTT_IA_25tar*1000), np.mean(RTT_IM_25tar*1000), np.mean(RTT_IA_75tar*1000), np.mean(RTT_IM_75tar*1000)]
errors2 = [stats.sem(RTT_IA_25tar*1000),stats.sem(RTT_IM_25tar*1000), stats.sem(RTT_IA_75tar*1000),stats.sem(RTT_IM_75tar*1000)]

fig,ax = plt.subplots(figsize=(8,6),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, color='lightgrey', edgecolor='white', label = 'Congruent')
plt.bar(r2, values2, yerr=errors2, width=barWidth, color='black', edgecolor='white', label = 'Incongruent')
plt.ylim((0,1050))
plt.ylabel('RT(ms)')
plt.xticks([(r + barWidth/2) for r in range(4)], names)
plt.legend(loc = "upper right", bbox_to_anchor=(1.08, 1.05), labelspacing=0.1)  # bbox_to_anchor indicates the position of the right corner of the legend relative to the whole box
plt.text(1.75, 1200, "Top", size=24,
         ha="center", va="center")
plt.text(0.75, 1120, "25% cue target", size=20,
         ha="center", va="center")
plt.text(2.75, 1120, "75% cue target", size=20,
         ha="center", va="center")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('Congruency by alignment_RTT.tiff')
plt.show()


# two cue probabilities in one plot, bottom as target
barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Aligned','Misaligned','Aligned','Misaligned']
values1 = [np.mean(RTB_CA_25tar*1000), np.mean(RTB_CM_25tar*1000), np.mean(RTB_CA_75tar*1000), np.mean(RTB_CM_75tar*1000)]
errors1 = [stats.sem(RTB_CA_25tar*1000),stats.sem(RTB_CM_25tar*1000), stats.sem(RTB_CA_75tar*1000),stats.sem(RTB_CM_75tar*1000)]
values2 = [np.mean(RTB_IA_25tar*1000), np.mean(RTB_IM_25tar*1000), np.mean(RTB_IA_75tar*1000), np.mean(RTB_IM_75tar*1000)]
errors2 = [stats.sem(RTB_IA_25tar*1000),stats.sem(RTB_IM_25tar*1000), stats.sem(RTB_IA_75tar*1000),stats.sem(RTB_IM_75tar*1000)]

fig,ax = plt.subplots(figsize=(8,6),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, color='lightgrey', edgecolor='white', label = 'Congruent')
plt.bar(r2, values2, yerr=errors2, width=barWidth, color='black', edgecolor='white', label = 'Incongruent')
plt.ylim((0,1050))
plt.ylabel('RT(ms)')
plt.xticks([(r + barWidth/2) for r in range(4)], names)
plt.legend(loc = "upper right", bbox_to_anchor=(1.08, 1.05), labelspacing=0.1)  # bbox_to_anchor indicates the position of the right corner of the legend relative to the whole box
plt.text(1.75, 1200, "Bottom", size=24,
         ha="center", va="center")
plt.text(0.75, 1120, "25% cue target", size=20,
         ha="center", va="center")
plt.text(2.75, 1120, "75% cue target", size=20,
         ha="center", va="center")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('Congruency by alignment_RTB.tiff')
plt.show()


# barplot
# two cued half in one plot, cue probability 25%
barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Aligned','Misaligned','Aligned','Misaligned']
values1 = [np.mean(RTT_CA_25tar*1000), np.mean(RTT_CM_25tar*1000), np.mean(RTB_CA_25tar*1000), np.mean(RTB_CM_25tar*1000)]
errors1 = [stats.sem(RTT_CA_25tar*1000),stats.sem(RTT_CM_25tar*1000), stats.sem(RTB_CA_25tar*1000),stats.sem(RTB_CM_25tar*1000)]
values2 = [np.mean(RTT_IA_25tar*1000), np.mean(RTT_IM_25tar*1000), np.mean(RTB_IA_25tar*1000), np.mean(RTB_IM_25tar*1000)]
errors2 = [stats.sem(RTT_IA_25tar*1000),stats.sem(RTT_IM_25tar*1000), stats.sem(RTB_IA_25tar*1000),stats.sem(RTB_IM_25tar*1000)]

fig,ax = plt.subplots(figsize=(8,6),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, color='lightgrey', edgecolor='white', label = 'Congruent')
plt.bar(r2, values2, yerr=errors2, width=barWidth, color='black', edgecolor='white', label = 'Incongruent')
plt.ylim((0,1050))
plt.ylabel('RT(ms)')
plt.xticks([(r + barWidth/2) for r in range(4)], names)
plt.legend(loc = "upper right", bbox_to_anchor=(1.08, 1.05), labelspacing=0.1)  # bbox_to_anchor indicates the position of the right corner of the legend relative to the whole box
plt.text(1.75, 1200, "25% cue", size=24,
         ha="center", va="center")
plt.text(0.75, 1120, "Top", size=20,
         ha="center", va="center")
plt.text(2.75, 1120, "Bottom", size=20,
         ha="center", va="center")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('Congruency by alignment_RT25tar.tiff')
plt.show()


# two cued half in one plot, cue probability 75%
barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Aligned','Misaligned','Aligned','Misaligned']
values1 = [np.mean(RTT_CA_75tar*1000), np.mean(RTT_CM_75tar*1000), np.mean(RTB_CA_75tar*1000), np.mean(RTB_CM_75tar*1000)]
errors1 = [stats.sem(RTT_CA_75tar*1000),stats.sem(RTT_CM_75tar*1000), stats.sem(RTB_CA_75tar*1000),stats.sem(RTB_CM_75tar*1000)]
values2 = [np.mean(RTT_IA_75tar*1000), np.mean(RTT_IM_75tar*1000), np.mean(RTB_IA_75tar*1000), np.mean(RTB_IM_75tar*1000)]
errors2 = [stats.sem(RTT_IA_75tar*1000),stats.sem(RTT_IM_75tar*1000), stats.sem(RTB_IA_75tar*1000),stats.sem(RTB_IM_75tar*1000)]

fig,ax = plt.subplots(figsize=(8,6),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, color='lightgrey', edgecolor='white', label = 'Congruent')
plt.bar(r2, values2, yerr=errors2, width=barWidth, color='black', edgecolor='white', label = 'Incongruent')
plt.ylim((0,1050))
plt.ylabel('RT(ms)')
plt.xticks([(r + barWidth/2) for r in range(4)], names)
plt.legend(loc = "upper right", bbox_to_anchor=(1.08, 1.05), labelspacing=0.1)  # bbox_to_anchor indicates the position of the right corner of the legend relative to the whole box
plt.text(1.75, 1200, "75% cue", size=24,
         ha="center", va="center")
plt.text(0.75, 1120, "Top", size=20,
         ha="center", va="center")
plt.text(2.75, 1120, "Bottom", size=20,
         ha="center", va="center")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('Congruency by alignment_RT75tar.tiff')
plt.show()

# anova
# four way: cue part, cue probability, congruency, alignment
pandas2ri.activate()
r_corRT = pandas2ri.py2ri(Data[(Data['isCorrect'] == 1)])

model = afex.aov_ez('Participant', 'reactionTime', r_corRT, within = ['Probability','CuedHalf', 'Alignment','Congruency'])
print(model)

# Main effect of probability, alignment, congruency
# No four-way or three-way interaction
# !!! Two-way interaction: Congruency x Alignment, p=.001

# main effect
main_prob = emmeans.emmeans(model, 'Probability', contr = 'pairwise', adjust = 'bonferroni')
print(main_prob)

main_align = emmeans.emmeans(model, 'Alignment', contr = 'pairwise', adjust = 'bonferroni')
print(main_align)

# simple effect
inter_congr_align = emmeans.emmeans(model, 'Alignment', by = 'Congruency', contr = 'pairwise', adjust = 'bonferroni')
print(inter_congr_align)

# descriptive
a = Data[(Data['isCorrect'] == 1)].groupby(['Participant','Probability'])['reactionTime'].describe()    
a.groupby(['Probability'])['mean'].describe() 

a = Data[(Data['isCorrect'] == 1)].groupby(['Participant','Alignment','Congruency'])['reactionTime'].describe()    
a.groupby(['Alignment','Congruency'])['mean'].describe() 


"""""""""""""""""""""""""""""""""""""""
RT: Traditional design: same incongruent

"""""""""""""""""""""""""""""""""""""""

"""""""""""
#descriptive

"""""""'"""

# 25% cue target
DataTcor_IAS_25tar = DataT_cor[(DataT_cor['Congruency'] == 'I') & (DataT_cor['Alignment'] == 'A') & (DataT_cor['SameDifferent'] == 'S') & (DataT_cor['Probability'] == '0.25cueTarget')]
DataTcor_IMS_25tar = DataT_cor[(DataT_cor['Congruency'] == 'I') & (DataT_cor['Alignment'] == 'M') & (DataT_cor['SameDifferent'] == 'S') & (DataT_cor['Probability'] == '0.25cueTarget')]
RTT_IAS_25tar = DataTcor_IAS_25tar.groupby(['Participant'])['reactionTime'].mean()
RTT_IMS_25tar = DataTcor_IMS_25tar.groupby(['Participant'])['reactionTime'].mean()
#stats.ttest_rel(RTT_IAS_25tar, RTT_IMS_25tar, nan_policy='omit')  # pair-wise difference: not significant

DataBcor_IAS_25tar = DataB_cor[(DataB_cor['Congruency'] == 'I') & (DataB_cor['Alignment'] == 'A') & (DataB_cor['SameDifferent'] == 'S') & (DataB_cor['Probability'] == '0.25cueTarget')]
DataBcor_IMS_25tar = DataB_cor[(DataB_cor['Congruency'] == 'I') & (DataB_cor['Alignment'] == 'M') & (DataB_cor['SameDifferent'] == 'S') & (DataB_cor['Probability'] == '0.25cueTarget')]
RTB_IAS_25tar = DataBcor_IAS_25tar.groupby(['Participant'])['reactionTime'].mean()
RTB_IMS_25tar = DataBcor_IMS_25tar.groupby(['Participant'])['reactionTime'].mean()
stats.ttest_rel(RTB_IAS_25tar, RTB_IMS_25tar, nan_policy='omit')  # pair-wise difference: not significant

# 75% cue target
DataTcor_IAS_75tar = DataT_cor[(DataT_cor['Congruency'] == 'I') & (DataT_cor['Alignment'] == 'A') & (DataT_cor['SameDifferent'] == 'S') & (DataT_cor['Probability'] == '0.75cueTarget')]
DataTcor_IMS_75tar = DataT_cor[(DataT_cor['Congruency'] == 'I') & (DataT_cor['Alignment'] == 'M') & (DataT_cor['SameDifferent'] == 'S') & (DataT_cor['Probability'] == '0.75cueTarget')]
RTT_IAS_75tar = DataTcor_IAS_75tar.groupby(['Participant'])['reactionTime'].mean()
RTT_IMS_75tar = DataTcor_IMS_75tar.groupby(['Participant'])['reactionTime'].mean()
#stats.ttest_rel(RTT_IAS_75tar, RTT_IMS_75tar, nan_policy='omit')  # pair-wise difference: not significant

DataBcor_IAS_75tar = DataB_cor[(DataB_cor['Congruency'] == 'I') & (DataB_cor['Alignment'] == 'A') & (DataB_cor['SameDifferent'] == 'S') & (DataB_cor['Probability'] == '0.75cueTarget')]
DataBcor_IMS_75tar = DataB_cor[(DataB_cor['Congruency'] == 'I') & (DataB_cor['Alignment'] == 'M') & (DataB_cor['SameDifferent'] == 'S') & (DataB_cor['Probability'] == '0.75cueTarget')]
RTB_IAS_75tar = DataBcor_IAS_75tar.groupby(['Participant'])['reactionTime'].mean()
RTB_IMS_75tar = DataBcor_IMS_75tar.groupby(['Participant'])['reactionTime'].mean()


"""""""""""""""""""""""""""""""""""""""
#three-way ANOVA: cuedHalf, probability, alignment

"""""""""""""""""""""""""""""""""""""""

# acc, 25% cue target
RTT_IAS_25tar[9] = np.nan  # sub no.9, acc=0 in this condition (Top_IAS_25tar)
RTT_IAS_25tar = RTT_IAS_25tar.sort_index()

IAS_IMS_RT_TB_25tar = pd.DataFrame(data = RTT_IAS_25tar)  
IAS_IMS_RT_TB_25tar = IAS_IMS_RT_TB_25tar.append(pd.DataFrame(data = RTT_IMS_25tar))
IAS_IMS_RT_TB_25tar = IAS_IMS_RT_TB_25tar.append(pd.DataFrame(data = RTB_IAS_25tar))  
IAS_IMS_RT_TB_25tar = IAS_IMS_RT_TB_25tar.append(pd.DataFrame(data = RTB_IMS_25tar))

IAS_IMS_RT_TB_25tar['CuedHalf'] = ['Top']*64 + ['Bottom']*64
IAS_IMS_RT_TB_25tar['Alignment'] = ['Aligned']*32 + ['MisAligned']*32 + ['Aligned']*32 + ['MisAligned']*32
IAS_IMS_RT_TB_25tar['participant'] = list(range(1,33)) * 4

# acc, 75% cue target
IAS_IMS_RT_TB_75tar = pd.DataFrame(data = RTT_IAS_75tar)
IAS_IMS_RT_TB_75tar = IAS_IMS_RT_TB_75tar.append(pd.DataFrame(data = RTT_IMS_75tar))
IAS_IMS_RT_TB_75tar = IAS_IMS_RT_TB_75tar.append(pd.DataFrame(data = RTB_IAS_75tar))
IAS_IMS_RT_TB_75tar = IAS_IMS_RT_TB_75tar.append(pd.DataFrame(data = RTB_IMS_75tar))

IAS_IMS_RT_TB_75tar['CuedHalf'] = ['Top']*64 + ['Bottom']*64
IAS_IMS_RT_TB_75tar['Alignment'] = ['Aligned']*32 + ['MisAligned']*32 + ['Aligned']*32 + ['MisAligned']*32
IAS_IMS_RT_TB_75tar['participant'] = list(range(1,33)) * 4

# combined acc data
IAS_IMS_RT_TB = pd.DataFrame(data = IAS_IMS_RT_TB_25tar)
IAS_IMS_RT_TB = IAS_IMS_RT_TB.append(pd.DataFrame(data = IAS_IMS_RT_TB_75tar))
IAS_IMS_RT_TB['Probability'] = ['25cueTarget']*128 + ['75cueTarget']*128 

# anova
pandas2ri.activate()
r_IAS_IMS_RT = pandas2ri.py2ri(IAS_IMS_RT_TB)

model = afex.aov_ez('participant', 'reactionTime', IAS_IMS_RT_TB, within = ['Probability','CuedHalf', 'Alignment'])
print(model)
# Main effect of probability, no effect of alignment
# No interactions

# main effect 
main_prob = emmeans.emmeans(model, 'Probability', contr = 'pairwise', adjust = 'bonferroni')
print(main_prob)

# descriptive
a = IAS_IMS_RT_TB.groupby(['participant','Probability'])['reactionTime'].describe()    
a.groupby(['Probability'])['mean'].describe() 


"""""""""""
Plot: two probabilities separately

"""""""'"""
# 25% cue target
means_RT_IAS_TB_25tar = [np.nanmean(RTT_IAS_25tar), np.mean(RTB_IAS_25tar)]; errors_IAS_TB_25tar = [stats.sem(RTT_IAS_25tar,nan_policy='omit'),stats.sem(RTB_IAS_25tar)]
means_RT_IMS_TB_25tar = [np.mean(RTT_IMS_25tar), np.mean(RTB_IMS_25tar)]; errors_IMS_TB_25tar = [stats.sem(RTT_IMS_25tar),stats.sem(RTB_IMS_25tar)]

index = np.arange(2)
bar_width = 0.35
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
rects1 = ax.bar(index,means_RT_IAS_TB_25tar, bar_width, yerr = errors_IAS_TB_25tar, label = 'Aligned')
rects1 = ax.bar(index + bar_width,means_RT_IMS_TB_25tar, bar_width, yerr = errors_IMS_TB_25tar, label = 'Misaligned')
ax.set_ylabel('RT(s)')
ax.set_ylim((0,1.05))
ax.set_title('Incongruent Same Conditions, 25% cue', fontweight="bold", fontsize=24)
ax.set_xticks(index + bar_width/ 2)
ax.set_xticklabels(('Top', 'Bottom'))
ax.legend()
plt.savefig('RT_SameIncongruent a vs. m_cueTB_25tar.tiff')
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(12,10),dpi=100)
plt.title('Incongruent Same Conditions, 25% cue', fontweight="bold", fontsize=24)
plt.ylabel('RT(s)')
plt.ylim(0,1.8)
sns.boxplot(data=[RTT_IAS_25tar, RTT_IMS_25tar, RTB_IAS_25tar, RTB_IMS_25tar], palette = "Blues", showmeans=True)
sns.swarmplot(data=[RTT_IAS_25tar, RTT_IMS_25tar, RTB_IAS_25tar, RTB_IMS_25tar], color = ".25")
sns.despine()
plt.xticks([0,1,2,3], ['Aligned Top','Misaligned Top','Aligned Bottom','Misaligned Bottom'])
plt.savefig('RT_SameIncongruent a vs. m_boxplot_cueTB_25tar.tiff')
plt.show()


# 75% top cue
means_RT_IAS_TB_75tar = [np.mean(RTT_IAS_75tar), np.mean(RTB_IAS_75tar)]; errors_IAS_TB_75tar = [stats.sem(RTT_IAS_75tar),stats.sem(RTB_IAS_75tar)]
means_RT_IMS_TB_75tar = [np.mean(RTT_IMS_75tar), np.mean(RTB_IMS_75tar)]; errors_IMS_TB_75tar = [stats.sem(RTT_IMS_75tar),stats.sem(RTB_IMS_75tar)]

index = np.arange(2)
bar_width = 0.35
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
rects1 = ax.bar(index,means_RT_IAS_TB_75tar, bar_width, yerr = errors_IAS_TB_75tar, label = 'Aligned')
rects1 = ax.bar(index + bar_width,means_RT_IMS_TB_75tar, bar_width, yerr = errors_IMS_TB_75tar, label = 'Misaligned')
ax.set_ylabel('RT(s)')
ax.set_ylim((0,1.05))
ax.set_title('Incongruent Same Conditions, 75% cue', fontweight="bold", fontsize=24)
ax.set_xticks(index + bar_width/ 2)
ax.set_xticklabels(('Top', 'Bottom'))
ax.legend()
plt.savefig('RT_SameIncongruent a vs. m_cueTB_75tar.tiff')
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(12,10),dpi=100)
plt.title('Incongruent Same Conditions, 75% cue', fontweight="bold", fontsize=24)
plt.ylabel('RT(s)')
plt.ylim(0,1.8)
sns.boxplot(data=[RTT_IAS_75tar, RTT_IMS_75tar, RTB_IAS_75tar, RTB_IMS_75tar], palette = "Blues", showmeans=True)
sns.swarmplot(data=[RTT_IAS_75tar, RTT_IMS_75tar, RTB_IAS_75tar, RTB_IMS_75tar], color = ".25")
sns.despine()
plt.xticks([0,1,2,3], ['Aligned Top','Misaligned Top','Aligned Bottom','Misaligned Bottom'])
plt.savefig('RT_SameIncongruent a vs. m_boxplot_cueTB_75tar.tiff')
plt.show()


"""""""""""""""""""""""""""""""""""""""
RT: Complementary design (part-whole like): same congruent

"""""""""""""""""""""""""""""""""""""""
"""""""""""
#descriptive

"""""""'"""

# 25% cue target
DataTcor_CAS_25tar = DataT_cor[(DataT_cor['Congruency'] == 'C') & (DataT_cor['Alignment'] == 'A') & (DataT_cor['SameDifferent'] == 'S') & (DataT_cor['Probability'] == '0.25cueTarget')]
DataTcor_CMS_25tar = DataT_cor[(DataT_cor['Congruency'] == 'C') & (DataT_cor['Alignment'] == 'M') & (DataT_cor['SameDifferent'] == 'S') & (DataT_cor['Probability'] == '0.25cueTarget')]
RTT_CAS_25tar = DataTcor_CAS_25tar.groupby(['Participant'])['reactionTime'].mean()
RTT_CMS_25tar = DataTcor_CMS_25tar.groupby(['Participant'])['reactionTime'].mean()

DataBcor_CAS_25tar = DataB_cor[(DataB_cor['Congruency'] == 'C') & (DataB_cor['Alignment'] == 'A') & (DataB_cor['SameDifferent'] == 'S') & (DataB_cor['Probability'] == '0.25cueTarget')]
DataBcor_CMS_25tar = DataB_cor[(DataB_cor['Congruency'] == 'C') & (DataB_cor['Alignment'] == 'M') & (DataB_cor['SameDifferent'] == 'S') & (DataB_cor['Probability'] == '0.25cueTarget')]
RTB_CAS_25tar = DataBcor_CAS_25tar.groupby(['Participant'])['reactionTime'].mean()
RTB_CMS_25tar = DataBcor_CMS_25tar.groupby(['Participant'])['reactionTime'].mean()

# 75% cue target
DataTcor_CAS_75tar = DataT_cor[(DataT_cor['Congruency'] == 'C') & (DataT_cor['Alignment'] == 'A') & (DataT_cor['SameDifferent'] == 'S') & (DataT_cor['Probability'] == '0.75cueTarget')]
DataTcor_CMS_75tar = DataT_cor[(DataT_cor['Congruency'] == 'C') & (DataT_cor['Alignment'] == 'M') & (DataT_cor['SameDifferent'] == 'S') & (DataT_cor['Probability'] == '0.75cueTarget')]
RTT_CAS_75tar = DataTcor_CAS_75tar.groupby(['Participant'])['reactionTime'].mean()
RTT_CMS_75tar = DataTcor_CMS_75tar.groupby(['Participant'])['reactionTime'].mean()

DataBcor_CAS_75tar = DataB_cor[(DataB_cor['Congruency'] == 'C') & (DataB_cor['Alignment'] == 'A') & (DataB_cor['SameDifferent'] == 'S') & (DataB_cor['Probability'] == '0.75cueTarget')]
DataBcor_CMS_75tar = DataB_cor[(DataB_cor['Congruency'] == 'C') & (DataB_cor['Alignment'] == 'M') & (DataB_cor['SameDifferent'] == 'S') & (DataB_cor['Probability'] == '0.75cueTarget')]
RTB_CAS_75tar = DataBcor_CAS_75tar.groupby(['Participant'])['reactionTime'].mean()
RTB_CMS_75tar = DataBcor_CMS_75tar.groupby(['Participant'])['reactionTime'].mean()


"""""""""""""""""""""""""""""""""""""""
#three-way ANOVA: cuedHalf, probability, alignment

"""""""""""""""""""""""""""""""""""""""

# acc, 25% cue target
CAS_CMS_RT_TB_25tar = pd.DataFrame(data = RTT_CAS_25tar)
CAS_CMS_RT_TB_25tar = CAS_CMS_RT_TB_25tar.append(pd.DataFrame(data = RTT_CMS_25tar))
CAS_CMS_RT_TB_25tar = CAS_CMS_RT_TB_25tar.append(pd.DataFrame(data = RTB_CAS_25tar))
CAS_CMS_RT_TB_25tar = CAS_CMS_RT_TB_25tar.append(pd.DataFrame(data = RTB_CMS_25tar))

CAS_CMS_RT_TB_25tar['CuedHalf'] = ['Top']*64 + ['Bottom']*64
CAS_CMS_RT_TB_25tar['Alignment'] = ['Aligned']*32 + ['MisAligned']*32 + ['Aligned']*32 + ['MisAligned']*32
CAS_CMS_RT_TB_25tar['participant'] = list(range(1,33)) * 4

# acc, 75% cue target
CAS_CMS_RT_TB_75tar = pd.DataFrame(data = RTT_CAS_75tar)
CAS_CMS_RT_TB_75tar = CAS_CMS_RT_TB_75tar.append(pd.DataFrame(data = RTT_CMS_75tar))
CAS_CMS_RT_TB_75tar = CAS_CMS_RT_TB_75tar.append(pd.DataFrame(data = RTB_CAS_75tar))
CAS_CMS_RT_TB_75tar = CAS_CMS_RT_TB_75tar.append(pd.DataFrame(data = RTB_CMS_75tar))

CAS_CMS_RT_TB_75tar['CuedHalf'] = ['Top']*64 + ['Bottom']*64
CAS_CMS_RT_TB_75tar['Alignment'] = ['Aligned']*32 + ['MisAligned']*32 + ['Aligned']*32 + ['MisAligned']*32
CAS_CMS_RT_TB_75tar['participant'] = list(range(1,33)) * 4

# combined acc data
CAS_CMS_RT_TB = pd.DataFrame(data = CAS_CMS_RT_TB_25tar)
CAS_CMS_RT_TB = CAS_CMS_RT_TB.append(pd.DataFrame(data = CAS_CMS_RT_TB_75tar))
CAS_CMS_RT_TB['Probability'] = ['25cueTarget']*128 + ['75cueTarget']*128 

# anova
pandas2ri.activate()
r_CAS_CMS_RT = pandas2ri.py2ri(CAS_CMS_RT_TB)

model = afex.aov_ez('participant', 'reactionTime', CAS_CMS_RT_TB, within = ['Probability','CuedHalf', 'Alignment'])
print(model)
# Main effect of probability and alignment
# Interactions between probability and alignment

# main effect 
main_prob = emmeans.emmeans(model, 'Probability', contr = 'pairwise', adjust = 'bonferroni')
print(main_prob)

main_align = emmeans.emmeans(model, 'Alignment', contr = 'pairwise', adjust = 'bonferroni')
print(main_align)

# simple effect
inter_prob_align = emmeans.emmeans(model, 'Alignment', by = 'Probability', contr = 'pairwise', adjust = 'bonferroni')
print(inter_prob_align)

# descriptive
a = CAS_CMS_RT_TB.groupby(['participant','Alignment','Probability'])['reactionTime'].describe()    
a.groupby(['Alignment','Probability'])['mean'].describe() 


"""""""""""
Plot: two probabilities separately

"""""""'"""
# 25% cue target
means_RT_CAS_TB_25tar = [np.mean(RTT_CAS_25tar), np.mean(RTB_CAS_25tar)]; errors_CAS_TB_25tar = [stats.sem(RTT_CAS_25tar),stats.sem(RTB_CAS_25tar)]
means_RT_CMS_TB_25tar = [np.mean(RTT_CMS_25tar), np.mean(RTB_CMS_25tar)]; errors_CMS_TB_25tar = [stats.sem(RTT_CMS_25tar),stats.sem(RTB_CMS_25tar)]

index = np.arange(2)
bar_width = 0.35
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
rects1 = ax.bar(index,means_RT_CAS_TB_25tar, bar_width, yerr = errors_CAS_TB_25tar, label = 'Aligned')
rects1 = ax.bar(index + bar_width,means_RT_CMS_TB_25tar, bar_width, yerr = errors_CMS_TB_25tar, label = 'Misaligned')
ax.set_ylabel('RT(s)')
ax.set_ylim((0,1.05))
ax.set_title('Congruent Same Conditions, 25% cue', fontweight="bold", fontsize=24)
ax.set_xticks(index + bar_width/ 2)
ax.set_xticklabels(('Top', 'Bottom'))
ax.legend()
plt.savefig('RT_SameCongruent a vs. m_cueTB_25tar.tiff')
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(12,10),dpi=100)
plt.title('Congruent Same Conditions, 25% cue', fontweight="bold", fontsize=24)
plt.ylabel('RT(s)')
plt.ylim(0,1.8)
sns.boxplot(data=[RTT_CAS_25tar, RTT_CMS_25tar, RTB_CAS_25tar, RTB_CMS_25tar], palette = "Blues", showmeans=True)
sns.swarmplot(data=[RTT_CAS_25tar, RTT_CMS_25tar, RTB_CAS_25tar, RTB_CMS_25tar], color = ".25")
sns.despine()
plt.xticks([0,1,2,3], ['Aligned Top','Misaligned Top','Aligned Bottom','Misaligned Bottom'])
plt.savefig('RT_SameCongruent a vs. m_boxplot_cueTB_25tar.tiff')
plt.show()


# 75% top cue
means_RT_CAS_TB_75tar = [np.mean(RTT_CAS_75tar), np.mean(RTB_CAS_75tar)]; errors_CAS_TB_75tar = [stats.sem(RTT_CAS_75tar),stats.sem(RTB_CAS_75tar)]
means_RT_CMS_TB_75tar = [np.mean(RTT_CMS_75tar), np.mean(RTB_CMS_75tar)]; errors_CMS_TB_75tar = [stats.sem(RTT_CMS_75tar),stats.sem(RTB_CMS_75tar)]

index = np.arange(2)
bar_width = 0.35
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
rects1 = ax.bar(index,means_RT_CAS_TB_75tar, bar_width, yerr = errors_CAS_TB_75tar, label = 'Aligned')
rects1 = ax.bar(index + bar_width,means_RT_CMS_TB_75tar, bar_width, yerr = errors_CMS_TB_75tar, label = 'Misaligned')
ax.set_ylabel('RT(s)')
ax.set_ylim((0,1.05))
ax.set_title('Congruent Same Conditions, 75% cue', fontweight="bold", fontsize=24)
ax.set_xticks(index + bar_width/ 2)
ax.set_xticklabels(('Top', 'Bottom'))
ax.legend()
plt.savefig('RT_SameCongruent a vs. m_cueTB_75tar.tiff')
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(12,10),dpi=100)
plt.title('Congruent Same Conditions, 75% cue', fontweight="bold", fontsize=24)
plt.ylabel('RT(s)')
plt.ylim(0,1.8)
sns.boxplot(data=[RTT_CAS_75tar, RTT_CMS_75tar, RTB_CAS_75tar, RTB_CMS_75tar], palette = "Blues", showmeans=True)
sns.swarmplot(data=[RTT_CAS_75tar, RTT_CMS_75tar, RTB_CAS_75tar, RTB_CMS_75tar], color = ".25")
sns.despine()
plt.xticks([0,1,2,3], ['Aligned Top','Misaligned Top','Aligned Bottom','Misaligned Bottom'])
plt.savefig('RT_SameCongruent a vs. m_boxplot_cueTB_75tar.tiff')
plt.show()


"""
plot: traditional design + part-whole (complementary) design
same incongruent condition + same congruent condition
Two conditions in one plot

"""

# 25% cue target
barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Top','Bottom','Top','Bottom']
values1 = [np.mean(RTT_CAS_25tar*1000), np.mean(RTB_CAS_25tar*1000), np.nanmean(RTT_IAS_25tar*1000), np.mean(RTB_IAS_25tar*1000)]
errors1 = [stats.sem(RTT_CAS_25tar*1000),stats.sem(RTB_CAS_25tar*1000), stats.sem(RTT_IAS_25tar*1000,nan_policy='omit'),stats.sem(RTB_IAS_25tar*1000)]
values2 = [np.mean(RTT_CMS_25tar*1000), np.mean(RTB_CMS_25tar*1000), np.mean(RTT_IMS_25tar*1000), np.mean(RTB_IMS_25tar*1000)]
errors2 = [stats.sem(RTT_CMS_25tar*1000),stats.sem(RTB_CMS_25tar*1000), stats.sem(RTT_IMS_25tar*1000),stats.sem(RTB_IMS_25tar*1000)]

fig,ax = plt.subplots(figsize=(8,6),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, color='lightgrey', edgecolor='white', label = 'Aligned')
plt.bar(r2, values2, yerr=errors2, width=barWidth, color='black', edgecolor='white', label = 'Misaligned')
plt.ylim((0,1050))
plt.ylabel('RT(ms)')
plt.xticks([(r + barWidth/2) for r in range(4)], names)
plt.legend(loc = "upper right", bbox_to_anchor=(1.08, 1.09), labelspacing=0.1)  # bbox_to_anchor indicates the position of the right corner of the legend relative to the whole box
plt.text(1.75, 1250, "25% cue", size=20,
         ha="center", va="center")
plt.text(0.75, 1150, "Same congruent", size=20,
         ha="center", va="center")
plt.text(2.75, 1150, "Same incongruent", size=20,
         ha="center", va="center")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('Traditional + Complementary_RT_25tar.tiff')
plt.show()


# 75% cue target
barWidth = 0.4
r1 = np.arange(4)
r2 = [x + barWidth for x in r1]
names = ['Top','Bottom','Top','Bottom']
values1 = [np.mean(RTT_CAS_75tar*1000), np.mean(RTB_CAS_75tar*1000), np.mean(RTT_IAS_75tar*1000), np.mean(RTB_IAS_75tar*1000)]
errors1 = [stats.sem(RTT_CAS_75tar*1000),stats.sem(RTB_CAS_75tar*1000), stats.sem(RTT_IAS_75tar*1000),stats.sem(RTB_IAS_75tar*1000)]
values2 = [np.mean(RTT_CMS_75tar*1000), np.mean(RTB_CMS_75tar*1000), np.mean(RTT_IMS_75tar*1000), np.mean(RTB_IMS_75tar*1000)]
errors2 = [stats.sem(RTT_CMS_75tar*1000),stats.sem(RTB_CMS_75tar*1000), stats.sem(RTT_IMS_75tar*1000),stats.sem(RTB_IMS_75tar*1000)]

fig,ax = plt.subplots(figsize=(8,6),dpi=100)
plt.bar(r1, values1, yerr=errors1, width=barWidth, color='lightgrey', edgecolor='white', label = 'Aligned')
plt.bar(r2, values2, yerr=errors2, width=barWidth, color='black', edgecolor='white', label = 'Misaligned')
plt.ylim((0,1050))
plt.ylabel('RT(ms)')
plt.xticks([(r + barWidth/2) for r in range(4)], names)
plt.legend(loc = "upper right", bbox_to_anchor=(1.08, 1.09), labelspacing=0.1)  # bbox_to_anchor indicates the position of the right corner of the legend relative to the whole box
plt.text(1.75, 1250, "75% cue", size=20,
         ha="center", va="center")
plt.text(0.75, 1150, "Same congruent", size=20,
         ha="center", va="center")
plt.text(2.75, 1150, "Same incongruent", size=20,
         ha="center", va="center")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('Traditional + Complementary_RT_75tar.tiff')
plt.show()


"""
save data

"""
# save the file to csv file
Data.to_csv('composite task_cue probability.csv', sep='\t')

dprime_long.to_csv('dprime_composite task.csv', sep='\t')

CAS_CMS_acc_TB.to_csv('acc_Same congruent_long.csv', sep='\t')
IAS_IMS_acc_TB.to_csv('acc_Same incongruent_long.csv', sep='\t')

CAS_CMS_RT_TB.to_csv('RT_Same congruent_long.csv', sep='\t')
IAS_IMS_RT_TB.to_csv('RT_Same incongruent_long.csv', sep='\t')