import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"D:\airepos\forked_ai_roadmap\ai-roadmap\content\ml\introduction\data\titanic.csv")
# data = sns.load_dataset('titanic')


# Understand the Dataset
print(data.head(7))
print(data.shape)
print(data.dtypes)
print(data.columns)

numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns
print(numerical_columns)

categorical_columns = data.select_dtypes(include=['object']).columns
print(categorical_columns)

# Summary Statistics
print(data.describe())
print("---------")
print(data['Sex'].value_counts())
print(data['Sex'].unique())
print(data['Sex'].nunique())
print("---------")
print(data.groupby('Sex')['Survived'].mean())
print("---------------")
print(data.groupby('Pclass')['Survived'].mean())
print("--------------")
print(data.groupby(["Sex", 'Pclass'])['Survived'].mean())
print("----------------")
pivot = data.pivot_table(values='Survived', index='Sex', columns='Pclass', aggfunc='mean')
print(pivot)

# Missing Data Analysis
print(data.isna().sum().sort_values(ascending=False))
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Cabin'] = data['Cabin'].fillna('Unknown')
print("---------------CLEANED DATA-----------------")
print(data.info())

# Data Visualization
sns.histplot(data['Age'], kde=True)
plt.show()
sns.boxplot(x='Sex', y='Age', data=data)
plt.show()

# A bar chart of Sex to show the male/female ratio.
sns.barplot(x='Sex', y='Age', data=data, color='red')
plt.show()

# A pie chart or bar plot showing the distribution of Survived (target variable).
sns.countplot(x='Survived', data=data)
plt.show()
plt.pie(data['Survived'].value_counts(), startangle=90, labels=['Did Not Survive', 'Survived'], autopct='%1.1f%%')
plt.show()

# Compare survival rates based on Pclass, Sex, or Embarked using grouped bar plots or heatmaps.
sns.catplot(x='Pclass', y="Survived", hue='Sex', data=data, kind='violin')
plt.show()

# Data Cleaning
duplicate_count = data.duplicated().sum()
print("Number of duplicates is ", duplicate_count)
data.drop_duplicates(inplace=True)
data['Age'] = data['Age'].fillna(data['Age'].median())

# Male - 1, Female - 0
data['Sex']= data['Sex'].map({'male': 1, 'female': 0})

#  Basic Insights

# What percentage of passengers survived?
print(data['Survived'].sum() / len(data) * 100)

# Did passengers in higher ticket classes (Pclass) have a higher survival rate?

print(data.groupby('Pclass')['Survived'].mean())

# Did age or gender influence survival?
data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# print(data.groupby(['AgeGroup', 'Sex'])['Survived'].mean())
print(data.pivot_table(values='Survived', index='Sex', columns='AgeGroup', aggfunc='sum'))

# Bonus Challenge
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1)
print(data[['SibSp', 'Parch', 'FamilySize', 'IsAlone']].head())


# Advanced Plot
numeric_cols = ['Age', 'Fare', 'Parch', 'SibSp', 'FamilySize']
sns.pairplot(data[numeric_cols + ['Survived']], hue='Survived', diag_kind='kde')
plt.suptitle('Pair Plot of Numerical Features Colored by Survival', y=1.02)
plt.show()

corr = data[['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'Survived']].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()