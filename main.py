import pandas as pd
import numpy as np

titanic_df = pd.read_csv(r'C:\Users\mona\PycharmProjects\titanic\Data\train.csv')
titanic_df = titanic_df.fillna(value={'Age':titanic_df['Age'].mean()},inplace=False)
titanic_df =titanic_df.drop(['Name','Ticket','Fare','PassengerId'] , axis=1)

def handle_non_numrical_data(df):
    columns = df.columns.values

    for column in columns :
        text_digit_values = {}
        def convert_to_int(val):
            return text_digit_values[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_values:
                    text_digit_values[unique] = x
                    x = x+1
            df[column] = list(map(convert_to_int , df[column]))
    return df

titanic_df = handle_non_numrical_data(titanic_df)

# titanis_df =titanis_df.fillna(value={'Cabin':'Nan'})
# print(titanis_df.Age[0:6])
# cabin_full = titanis_df[titanis_df.Cabin == 'Nan']
# cabin_full = titanis_df['Cabin'].isnull()
# print(cabin_full)
# x = titanis_df['Pclass','Sex','Age',]
print(titanic_df.head())