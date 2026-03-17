import numpy as np
import pandas as pd

from percolator_RESET import utility_functions as uf
import percolator_functions_ju as pfju


#search_file = "/mnt/data/pipeline-of-identifications/publication/PXD023217-CAMPI/DB1/percolator-reset-test/S07.mzid.psm_utils.pin"
search_file = "/mnt/data/percolator-reset-test/LFQ_Orbitrap_DDA_Human_03.mzid.psm_utils.enhanced.tsv.features.oktoberfest.specid_corrected.pin"
dynamic_competition = True
pair = False
stem_mod = False

initial_dir = 'absdM'
#sort_score = 'neg_ln_comet_expectation_value'
sort_score = 'ln_msgf_specevalue'
mult = 1
p_init = 0.5
total_iter = 10
FDR_threshold = 0.01
train_FDR_threshold = 0.01
folds = 3
remove = ['enzInt']#,'ExpMass', 'CalcMass']

 #read in search file(s)
data_dfs = []

data_df = uf.read_pin(search_file)

# cast SPecId to int
data_df['SpecId'] = data_df['SpecId'].apply(lambda x: int(x))

#removing flanking aa
data_df['Peptide'] = data_df['Peptide'].str.extract(r'^[^.]*\.(.*?)\.[^.]*$', expand=False).fillna(data_df['Peptide'])
data_df['Proteins'] = data_df['Proteins'].str.replace('\t', ',')


df_all = pfju.PSM_level(data_df.copy(), None, top=1, sortscore=sort_score)


PSMs = data_df.copy()
PSMs['rank'] = 1
PSMs = PSMs[PSMs['rank'] == 1].reset_index(drop=True)
PSMs.drop('rank', inplace = True, axis = 1)

#applying scaling
df_all_scale, scale = pfju.do_scale(df_all.copy())

#create target-decoys at pseudolevel
rand_indxs = np.random.choice([True, False], replace = True, size = sum((df_all_scale['Label'] == -1)), p = [p_init, 1 - p_init])

train_all = df_all_scale.loc[(df_all_scale['Label'] == -1)].copy()
train_all = train_all.loc[rand_indxs].copy()

train_all_unscale = df_all.loc[(df_all['Label'] == -1)].copy()
train_all_unscale = train_all_unscale.loc[rand_indxs].copy()

#report number of target and decoy peptides
print("There are %s target peptides. \n" %(sum(df_all.Label == 1)))
print("There are %s decoy peptides. \n" %(sum(df_all.Label == -1)))

#do SVM
df_new, train_all_new, model, columns_trained = pfju.do_svm(df_all_scale.copy(), train_all.copy(), df_all.copy(), folds=folds,
        p=p_init, total_iter=total_iter, alpha=FDR_threshold, train_alpha=train_FDR_threshold, remove = remove, mult=mult, initial_dir=initial_dir, n_jobs=30)

df_new = df_new.loc[(df_new.q_val <= FDR_threshold) | (df_new.Label == -1)]

coefficients = np.concatenate((model.best_estimator_.coef_[0], model.best_estimator_.intercept_))
coefficients = pd.Series(coefficients)
coefficients.index = columns_trained.append(pd.Index(['intercept']))

print('Final SVM feature coefficients:')
print(coefficients.to_string())
print('Features that are constant are dropped.')

filtered = df_new[df_new['Label'] == 1]
filtered = filtered[filtered['q_val'] <= 0.01]

print(f"Filtered: {filtered.shape}")
filtered.to_csv("./psms.percolator-reset.txt", header=True, index=False, sep='\t')
