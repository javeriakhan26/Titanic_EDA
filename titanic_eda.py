import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 1. Basic inspection
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# 2. Survival by Sex
plt.figure(figsize=(8,5))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Sex')
plt.savefig('survival_by_sex.png')
plt.show()

# 3. Survival by Class
plt.figure(figsize=(8,5))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Class')
plt.savefig('survival_by_class.png')
plt.show()

# 4. Age distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution')
plt.savefig('age_distribution.png')
plt.show()

# 5. Survival by Age bucket
df['AgeBucket'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100],
                          labels=['Child','Teen','Adult','Middle','Senior'])
plt.figure(figsize=(8,5))
sns.barplot(x='AgeBucket', y='Survived', data=df)
plt.title('Survival Rate by Age Group')
plt.savefig('survival_by_age.png')
plt.show()

print("\nInsight Report:")
print("- Females had much higher survival rate than males")
print("- 1st class passengers survived more than 2nd and 3rd class")
print("- Children had relatively higher survival rate")
print("- Most passengers were adults aged 20-40")
print("- About 20% of Age data is missing")