import os
import pandas as pd

# data inladen
path_to_file = "/Users/thibaudseveri/PycharmProjects/pythonProject/importdata/HR_log_data.xlsx"
df = pd.read_excel(path_to_file)

# extra kolommen maken voor het jaar van time_start en het jaar van time_end
date_col = pd.DatetimeIndex(df['time_start'])
date_col1 = pd.DatetimeIndex(df['time_end'])
df["Year_time_start"] = date_col.year
df["Year_time_end"] = date_col1.year

#delete all rows with 'before_data_capture' because it is not registered which activity the employee did
df = df[df['act'] != 'before_data_capture']

#print nieuwe dataframe dagmaandjaar in excel bestand
with pd.ExcelWriter('dagmaandjaar.xlsx') as writer:
    df.to_excel(writer, sheet_name='Sheet1', index=False)
os.system('open ' + 'dagmaandjaar.xlsx')