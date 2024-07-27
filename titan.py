from pandas import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree
from loguru import logger
logger.success('Программа запущена')
age=[]
df=read_csv('C:/Users/andre/Desktop/Rebotica/titanic.csv')
logger.info('Датафрейм создан')
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df=df.fillna(df['Age'].mean())
df=get_dummies(df, drop_first=True, dtype=int)
df=df.rename(columns={'Sex_male': 'Sex'})
logger.info('Датафрейм отсортирован')
target=df['Survived']
fatures=df.drop('Survived', axis=1)
logger.info('target и futures выбраны')
target_train, target_test_=train_test_split(target, test_size=0.25, random_state=322)
fatures_train, fatures_test=train_test_split(fatures, test_size=0.25, random_state=322)
tree_=tree.DecisionTreeClassifier(criterion='entropy')
logger.info('Дерево создано')
way=tree_.fit(fatures_train, target_train)
predict=way.predict(fatures)
logger.info("predict создан")

sir=df['Pclass'].value_counts(1)*100#процентное количечество людей, поезхавших 3 классом
sir=sir.head(1)
print(sir)
logger.success('процентное количечество людей, поезхавших 3 классом посчитано')

sir2=df.Sex.value_counts(1)*100#процентное количество мужчин на борту
sir2=sir2.head(1)
print(sir2)
logger.success('процентное количество мужчин на борту посчитано')


sir3=df['SibSp']+df['Parch']#количество людей , у котрыч больше одного родственника
sir3=sir3.value_counts()
sir4=sir3.tail(7)
sir4=sir4.sum()
print(sir4)
logger.success('количество людей , у котрыч больше одного родственника посчитано')

#print(target_test_.shape) # показывает количество
print(tree.plot_tree(way))

