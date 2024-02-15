import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# This script reads csv data and saves it to joblib-model file "euriborModel.joblib"
# Data source (CSV): https://www.suomenpankki.fi/fi/Tilastot/korot/kuviot/korot_kuviot/euriborkorot_pv_chrt_fi/
# You can update the data by downloading the new .csv data (no need to edit anything in the file)

data = pd.read_csv('euriborkorot_pv_chrt_fi.csv')
x = data.drop(columns=[
                       'dundasChartControl1_DRG_DataRowGrouping1_dundasChartControl1_DCG_Period1_label',
                       'dundasChartControl1_DRG_DataRowGrouping1_dundasChartControl1_DCG_Period1_Value_Y',       
                       ]
                       )
y = data['dundasChartControl1_DRG_DataRowGrouping1_dundasChartControl1_DCG_Period1_Value_Y']

x.columns = x.columns.str.replace('dundasChartControl1_DRG_DataRowGrouping1_label', 'euribor')
x.columns = x.columns.str.replace('dundasChartControl1_DRG_DataRowGrouping1_dundasChartControl1_DCG_Period1_Value_X', 'date')
y.replace('dundasChartControl1_DRG_DataRowGrouping1_dundasChartControl1_DCG_Period1_Value_Y', 'value')
rows = 0
x_new = []
for i in x.values:
    arr = []
    if i[0] == 'txtb_row_title' or len(i[0]) == 0:
        break
    elif i[0] == '1 vko':
        arr.append(0.2)
    else:
        arr.append(float(i[0][0]+i[0][1]))

    day = i[1][3:5]
    month = i[1][0:2]
    year = i[1][6:10]
    arr.append(float(day))
    arr.append(float(month))
    arr.append(float(year))   
    x_new.append(arr)

y_new = []
for i in y.values:
    if i == 'txtb_value':
        break
    y_new.append(i)

model = DecisionTreeClassifier()
model.fit(pd.DataFrame(x_new), pd.DataFrame(y_new))

joblib.dump(model, 'euriborModel.joblib')