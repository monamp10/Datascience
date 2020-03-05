import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def handle_non_numrical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_values = {}

        def convert_to_int(val):
            return text_digit_values[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_values:
                    text_digit_values[unique] = x
                    x = x + 1
            df[column] = list(map(convert_to_int, df[column]))
    return df


titanic_train = pd.read_csv(r'C:\Users\mona\PycharmProjects\titanic\Data\train.csv')
titanic_test = pd.read_csv(r'C:\Users\mona\PycharmProjects\titanic\Data\test.csv')
titanic_train = titanic_train.fillna(value={'Age': titanic_train['Age'].mean()}, inplace=False)
titanic_train = titanic_train.drop(['Name', 'Ticket', 'Fare', 'PassengerId', 'Cabin'], axis=1)

titanic_train = handle_non_numrical_data(titanic_train)
X_train = titanic_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
Y_train = titanic_train[['Survived']]

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, Y_train.values.ravel())
Y_predict = classifier.predict(X_train)

Y_train_np = Y_train.to_numpy()

result = []
count_T = 0
count_F = 0
for i in range(len(Y_predict)):
    if Y_predict[i] == Y_train_np[i]:
        result.append('True')
        count_T = count_T + 1
    else:
        result.append('False')
        count_F = count_F + 1
print(result)
print(count_T)
print(count_F)

# print(X_train.head())
# print(Y_train.head())
# titanis_df =titanis_df.fillna(value={'Cabin':'Nan'})
# print(titanis_df.Age[0:6])
# cabin_full = titanic_train[titanic_train.Cabin == 0 ]
# cabin_full = titanis_df['Cabin'].isnull()
# print(len(cabin_full))
# print(len(titanic_train))
# x = titanis_df['Pclass','Sex','Age',]
# print(titanic_train.head())
