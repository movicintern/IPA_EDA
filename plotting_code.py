import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_excel("/home/movic/True_NAS2/EDA_IPA_01_(MG_JY)/Excel/base_watergate.xlsx")

df_ = df[~df['prob_organizer_E#1'].isna()]

df_['시각'] = pd.to_datetime(df['시각'])

fig = plt.figure(figsize=(12, 12))

ax1 = fig.add_subplot(211)
ax1.plot(np.arange(0, len(df_['시각'])), df_['prob_organizer_E#1'])
ax1.set_title("Rate of Change Transition E#1")


ax2 = fig.add_subplot(212)
ax2.plot(np.arange(0, len(df_['시각'])), df_['prob_organizer_E#2'])
ax2.set_title("Rate of Change Transition E#2")
plt.tight_layout()
plt.show()

df = pd.read_excel("/home/movic/True_NAS2/EDA_IPA_01_(MG_JY)/Excel/base_watergate.xlsx")
df_ = df[~df['prob_organizer_E#1'].isna()]

data = np.load('/home/movic/JY/EDA/20241004151055.npy')

df_['prob_organizer_E#1'] = np.clip(df_['prob_organizer_E#1'], np.min(data[0]), np.max(df_['prob_organizer_E#1']))
df_['prob_organizer_E#2'] = np.clip(df_['prob_organizer_E#2'], np.min(data[1]), np.max(df_['prob_organizer_E#2']))

min1 = np.min(df_['prob_organizer_E#1'])
max1 = np.max(df_['prob_organizer_E#1'])
median1 = np.median(df_['prob_organizer_E#1'])

threshold1 = (min1+max1+median1)/3

min2 = np.min(df_['prob_organizer_E#2'])
max2 = np.max(df_['prob_organizer_E#2'])
median2 = np.median(df_['prob_organizer_E#2'])

threshold2 = (min2+max2+median2)/3

fig = plt.figure(figsize=(12, 12))

ax2 = fig.add_subplot(211)
ax2.plot(df_['시각'], df_['prob_organizer_E#1'])
ax2.set_xticks(np.arange(0, len(df_['시각']))[::10])
ax2.set_xticklabels(np.arange(0, len(df_['시각']))[::10], rotation=270)
ax2.set_title('Rate of Change Transition E#1')
plt.axhline(threshold1, 0, 1, color='red', linestyle='-', linewidth=2)
plt.tight_layout()

ax2 = fig.add_subplot(212)
ax2.plot(df_['시각'], df_['prob_organizer_E#2'])
ax2.set_xticks(np.arange(0, len(df_['시각']))[::10])
ax2.set_xticklabels(np.arange(0, len(df_['시각']))[::10], rotation=270)
ax2.set_title('Rate of Change Transition E#2')
plt.axhline(threshold2, 0, 1, color='red', linestyle='-', linewidth=2)
plt.tight_layout()
plt.show()