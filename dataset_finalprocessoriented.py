import os
import pandas as pd

# data inladen
path_to_file = "/Users/thibaudseveri/PycharmProjects/pythonProject/importdata/dagmaandjaar.xlsx"
dagmaandjaar = pd.read_excel(path_to_file)

#maak 10 kopies van dagmaandjaar dataset
copy1 = dagmaandjaar.copy()
copy2 = dagmaandjaar.copy()
copy3 = dagmaandjaar.copy()
copy4 = dagmaandjaar.copy()
copy5 = dagmaandjaar.copy()
copy6 = dagmaandjaar.copy()
copy7 = dagmaandjaar.copy()
copy8 = dagmaandjaar.copy()
copy9 = dagmaandjaar.copy()
copy10 = dagmaandjaar.copy()

#dataset 2021 maken
dataset2021 = copy1
dataset2021 = dataset2021.loc[(dataset2021['Year_time_start'] == 2021) | ((dataset2021['Year_time_start'] < 2021) & (dataset2021['Year_time_end'] >= 2021))]
dataset2021 = dataset2021.drop_duplicates(subset=['id', 'Year_time_start'], keep='last') #als er twee functies gedaan werden in hetzelfde jaar houden we de meest recente
dataset2021 = dataset2021.drop(['time_start', 'time_end', 'next_act', 'contract_start', 'contract_end', 'Year_time_start',	'Year_time_end'], axis=1).copy()
dataset2021 = dataset2021.rename(columns={'act': 'act_2021', 'prev_act': 'prev_act_2021', 'V01': 'V01_2021', 'V02':'V02_2021', 'V03': 'V03_2021', 'V04':'V04_2021', 'V05': 'V05_2021', 'V06': 'V06_2021', 'V07': 'V07_2021', 'V08': 'V08_2021', 'V09': 'V09_2021', 'V10': 'V10_2021', 'V11': 'V11_2021'})

#dataset2020 maken
dataset2020 = copy2
dataset2020 = dataset2020.loc[(dataset2020['Year_time_start'] == 2020) | ((dataset2020['Year_time_start'] < 2020) & (dataset2020['Year_time_end'] >= 2020))]
dataset2020 = dataset2020.drop_duplicates(subset=['id', 'Year_time_start'], keep='last')
dataset2020 = dataset2020.drop(['time_start', 'time_end', 'next_act', 'contract_start', 'contract_end', 'Year_time_start',	'Year_time_end'], axis=1).copy()
dataset2020 = dataset2020.rename(columns={'act': 'act_2020', 'prev_act': 'prev_act_2020', 'V01': 'V01_2020', 'V02':'V02_2020', 'V03': 'V03_2020', 'V04':'V04_2020', 'V05': 'V05_2020', 'V06': 'V06_2020', 'V07': 'V07_2020', 'V08': 'V08_2020', 'V09': 'V09_2020', 'V10': 'V10_2020', 'V11': 'V11_2020'})

#dataset2019 maken
dataset2019 = copy3
dataset2019 = dataset2019.loc[(dataset2019['Year_time_start'] == 2019) | ((dataset2019['Year_time_start'] < 2019) & (dataset2019['Year_time_end'] >= 2019))]
dataset2019 = dataset2019.drop_duplicates(subset=['id', 'Year_time_start'], keep='last')
dataset2019 = dataset2019.drop(['time_start', 'time_end', 'next_act', 'contract_start', 'contract_end', 'Year_time_start',	'Year_time_end'], axis=1).copy()
dataset2019 = dataset2019.rename(columns={'act': 'act_2019', 'prev_act': 'prev_act_2019', 'V01': 'V01_2019', 'V02':'V02_2019', 'V03': 'V03_2019', 'V04':'V04_2019', 'V05': 'V05_2019', 'V06': 'V06_2019', 'V07': 'V07_2019', 'V08': 'V08_2019', 'V09': 'V09_2019', 'V10': 'V10_2019', 'V11': 'V11_2019'})

#dataset2018 maken
dataset2018 = copy4
dataset2018 = dataset2018.loc[(dataset2018['Year_time_start'] == 2018) | ((dataset2018['Year_time_start'] < 2018) & (dataset2018['Year_time_end'] >= 2018))]
dataset2018 = dataset2018.drop_duplicates(subset=['id', 'Year_time_start'], keep='last')
dataset2018 = dataset2018.drop(['time_start', 'time_end', 'next_act', 'contract_start', 'contract_end', 'Year_time_start',	'Year_time_end'], axis=1).copy()
dataset2018 = dataset2018.rename(columns={'act': 'act_2018', 'prev_act': 'prev_act_2018', 'V01': 'V01_2018', 'V02':'V02_2018', 'V03': 'V03_2018', 'V04':'V04_2018', 'V05': 'V05_2018', 'V06': 'V06_2018', 'V07': 'V07_2018', 'V08': 'V08_2018', 'V09': 'V09_2018', 'V10': 'V10_2018', 'V11': 'V11_2018'})

#dataset2017 maken
dataset2017 = copy5
dataset2017 = dataset2017.loc[(dataset2017['Year_time_start'] == 2017) | ((dataset2017['Year_time_start'] < 2017) & (dataset2017['Year_time_end'] >= 2017))]
dataset2017 = dataset2017.drop_duplicates(subset=['id', 'Year_time_start'], keep='last')
dataset2017 = dataset2017.drop(['time_start', 'time_end', 'next_act', 'contract_start', 'contract_end', 'Year_time_start',	'Year_time_end'], axis=1).copy()
dataset2017 = dataset2017.rename(columns={'act': 'act_2017', 'prev_act': 'prev_act_2017', 'V01': 'V01_2017', 'V02':'V02_2017', 'V03': 'V03_2017', 'V04':'V04_2017', 'V05': 'V05_2017', 'V06': 'V06_2017', 'V07': 'V07_2017', 'V08': 'V08_2017', 'V09': 'V09_2017', 'V10': 'V10_2017', 'V11': 'V11_2017'})

#dataset2016 maken
dataset2016 = copy6
dataset2016 = dataset2016.loc[(dataset2016['Year_time_start'] == 2016) | ((dataset2016['Year_time_start'] < 2016) & (dataset2016['Year_time_end'] >= 2016))]
dataset2016 = dataset2016.drop_duplicates(subset=['id', 'Year_time_start'], keep='last')
dataset2016 = dataset2016.drop(['time_start', 'time_end', 'next_act', 'contract_start', 'contract_end', 'Year_time_start',	'Year_time_end'], axis=1).copy()
dataset2016 = dataset2016.rename(columns={'act': 'act_2016', 'prev_act': 'prev_act_2016', 'V01': 'V01_2016', 'V02':'V02_2016', 'V03': 'V03_2016', 'V04':'V04_2016', 'V05': 'V05_2016', 'V06': 'V06_2016', 'V07': 'V07_2016', 'V08': 'V08_2016', 'V09': 'V09_2016', 'V10': 'V10_2016', 'V11': 'V11_2016'})

#dataset2015 maken
dataset2015 = copy7
dataset2015 = dataset2015.loc[(dataset2015['Year_time_start'] == 2015) | ((dataset2015['Year_time_start'] < 2015) & (dataset2015['Year_time_end'] >= 2015))]
dataset2015 = dataset2015.drop_duplicates(subset=['id', 'Year_time_start'], keep='last')
dataset2015 = dataset2015.drop(['time_start', 'time_end', 'next_act', 'contract_start', 'contract_end', 'Year_time_start',	'Year_time_end'], axis=1).copy()
dataset2015 = dataset2015.rename(columns={'act': 'act_2015', 'prev_act': 'prev_act_2015', 'V01': 'V01_2015', 'V02':'V02_2015', 'V03': 'V03_2015', 'V04':'V04_2015', 'V05': 'V05_2015', 'V06': 'V06_2015', 'V07': 'V07_2015', 'V08': 'V08_2015', 'V09': 'V09_2015', 'V10': 'V10_2015', 'V11': 'V11_2015'})

#dataset2014 maken
dataset2014 = copy8
dataset2014 = dataset2014.loc[(dataset2014['Year_time_start'] == 2014) | ((dataset2014['Year_time_start'] < 2014) & (dataset2014['Year_time_end'] >= 2014))]
dataset2014 = dataset2014.drop_duplicates(subset=['id', 'Year_time_start'], keep='last')
dataset2014 = dataset2014.drop(['time_start', 'time_end', 'next_act', 'contract_start', 'contract_end', 'Year_time_start',	'Year_time_end'], axis=1).copy()
dataset2014 = dataset2014.rename(columns={'act': 'act_2014', 'prev_act': 'prev_act_2014', 'V01': 'V01_2014', 'V02':'V02_2014', 'V03': 'V03_2014', 'V04':'V04_2014', 'V05': 'V05_2014', 'V06': 'V06_2014', 'V07': 'V07_2014', 'V08': 'V08_2014', 'V09': 'V09_2014', 'V10': 'V10_2014', 'V11': 'V11_2014'})

#dataset2013 maken
dataset2013 = copy9
dataset2013 = dataset2013.loc[(dataset2013['Year_time_start'] == 2013) | ((dataset2013['Year_time_start'] < 2013) & (dataset2013['Year_time_end'] >= 2013))]
dataset2013 = dataset2013.drop_duplicates(subset=['id', 'Year_time_start'], keep='last')
dataset2013 = dataset2013.drop(['time_start', 'time_end', 'next_act', 'contract_start', 'contract_end', 'Year_time_start',	'Year_time_end'], axis=1).copy()
dataset2013 = dataset2013.rename(columns={'act': 'act_2013', 'prev_act': 'prev_act_2013', 'V01': 'V01_2013', 'V02':'V02_2013', 'V03': 'V03_2013', 'V04':'V04_2013', 'V05': 'V05_2013', 'V06': 'V06_2013', 'V07': 'V07_2013', 'V08': 'V08_2013', 'V09': 'V09_2013', 'V10': 'V10_2013', 'V11': 'V11_2013'})

#dataset2012 maken
dataset2012 = copy10
dataset2012 = dataset2012.loc[dataset2012['Year_time_start'] == 2012]
dataset2012 = dataset2012.drop_duplicates(subset=['id', 'Year_time_start'], keep='last')
dataset2012 = dataset2012.drop(['time_start', 'time_end', 'next_act', 'contract_start', 'contract_end', 'Year_time_start',	'Year_time_end'], axis=1).copy()
dataset2012 = dataset2012.rename(columns={'act': 'act_2012', 'prev_act': 'prev_act_2012', 'V01': 'V01_2012', 'V02':'V02_2012', 'V03': 'V03_2012', 'V04':'V04_2012', 'V05': 'V05_2012', 'V06': 'V06_2012', 'V07': 'V07_2012', 'V08': 'V08_2012', 'V09': 'V09_2012', 'V10': 'V10_2012', 'V11': 'V11_2012'})

#merge all datasets together on 'id' in new_dataset
merged_df = pd.merge(dataset2012, dataset2013, on='id', how='outer')
merged_df = pd.merge(merged_df, dataset2014, on='id', how='outer')
merged_df = pd.merge(merged_df, dataset2015, on='id', how='outer')
merged_df = pd.merge(merged_df, dataset2016, on='id', how='outer')
merged_df = pd.merge(merged_df, dataset2017, on='id', how='outer')
merged_df = pd.merge(merged_df, dataset2018, on='id', how='outer')
merged_df = pd.merge(merged_df, dataset2019, on='id', how='outer')
merged_df = pd.merge(merged_df, dataset2020, on='id', how='outer')
merged_df = pd.merge(merged_df, dataset2021, on='id', how='outer')

#maak dataset uniek op basis van juiste rijen
merged_df = merged_df.drop_duplicates(subset=['id'], keep='last')

# Create a new merged_df with the columns in the desired order
merged_df = merged_df[['id', 'prev_act_2012', 'act_2012', 'V01_2012', 'V02_2012', 'V03_2012', 'V04_2012', 'V05_2012', 'V06_2012', 'V07_2012', 'V08_2012', 'V09_2012', 'V10_2012', 'V11_2012',
'prev_act_2013', 'act_2013', 'V01_2013', 'V02_2013', 'V03_2013', 'V04_2013', 'V05_2013', 'V06_2013', 'V07_2013', 'V08_2013', 'V09_2013', 'V10_2013', 'V11_2013',
'prev_act_2014', 'act_2014', 'V01_2014', 'V02_2014', 'V03_2014', 'V04_2014', 'V05_2014', 'V06_2014', 'V07_2014', 'V08_2014', 'V09_2014', 'V10_2014', 'V11_2014',
'prev_act_2015', 'act_2015', 'V01_2015', 'V02_2015', 'V03_2015', 'V04_2015', 'V05_2015', 'V06_2015', 'V07_2015', 'V08_2015', 'V09_2015', 'V10_2015', 'V11_2015',
'prev_act_2016', 'act_2016', 'V01_2016', 'V02_2016', 'V03_2016', 'V04_2016', 'V05_2016', 'V06_2016', 'V07_2016', 'V08_2016', 'V09_2016', 'V10_2016', 'V11_2016',
'prev_act_2017', 'act_2017', 'V01_2017', 'V02_2017', 'V03_2017', 'V04_2017', 'V05_2017', 'V06_2017', 'V07_2017', 'V08_2017', 'V09_2017', 'V10_2017', 'V11_2017',
'prev_act_2018', 'act_2018', 'V01_2018', 'V02_2018', 'V03_2018', 'V04_2018', 'V05_2018', 'V06_2018', 'V07_2018', 'V08_2018', 'V09_2018', 'V10_2018', 'V11_2018',
'prev_act_2019', 'act_2019', 'V01_2019', 'V02_2019', 'V03_2019', 'V04_2019', 'V05_2019', 'V06_2019', 'V07_2019', 'V08_2019', 'V09_2019', 'V10_2019', 'V11_2019',
'prev_act_2020', 'act_2020', 'V01_2020', 'V02_2020', 'V03_2020', 'V04_2020', 'V05_2020', 'V06_2020', 'V07_2020', 'V08_2020', 'V09_2020', 'V10_2020', 'V11_2020',
'prev_act_2021', 'act_2021', 'V01_2021', 'V02_2021', 'V03_2021', 'V04_2021', 'V05_2021', 'V06_2021', 'V07_2021', 'V08_2021', 'V09_2021', 'V10_2021', 'V11_2021']]

#print nieuwe dataframe finalprocessorienteddataset in excel bestand
with pd.ExcelWriter('finalprocessorienteddataset.xlsx') as writer:
    merged_df.to_excel(writer, sheet_name='Sheet1', index=False)
os.system('open ' + 'finalprocessorienteddataset.xlsx')