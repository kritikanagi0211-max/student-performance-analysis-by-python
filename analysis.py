import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

# ---------------- LOAD DATA ----------------
data = pd.read_excel(r"C:\Users\Kritika\Downloads\student_data.xlsx")
data.columns = data.columns.str.strip()

print("Columns:", data.columns)
print("Rows BEFORE cleaning:", len(data))

# ---------------- CLEANING FUNCTIONS ----------------

def clean_hours(x):
    try:
        x = str(x).lower()
        if '2-3' in x: return 2.5
        if '3-4' in x: return 3.5
        if '1-2' in x: return 1.5
        if 'half' in x: return 0.5

        num = re.findall(r'\d+\.?\d*', x)
        if num:
            value = float(num[0])
            if value > 15:
                return None
            return value
        return None
    except:
        return None

def clean_marks(x):
    try:
        x = str(x).lower()
        if 'no' in x or 'not' in x or 'result' in x:
            return None

        x = x.replace('%','').replace('percent','')
        num = re.findall(r'\d+\.?\d*', x)

        if num:
            value = float(num[0])
            if value < 20 or value > 100:
                return None
            return value
        return None
    except:
        return None

def clean_attendance(x):
    try:
        x = str(x).lower().replace('%','')

        if '-' in x:
            nums = re.findall(r'\d+', x)
            if len(nums) == 2:
                return (float(nums[0]) + float(nums[1])) / 2

        num = re.findall(r'\d+\.?\d*', x)
        if num:
            value = float(num[0])
            if value < 50 or value > 100:
                return None
            return value
        return None
    except:
        return None

# ---------------- APPLY CLEANING ----------------
data['study_hours'] = data['How many hours do you study daily?'].apply(clean_hours)
data['marks'] = data['What were your MST - 1 marks (%) in current semester?'].apply(clean_marks)
data['attendance'] = data['What is your attendance percentage?'].apply(clean_attendance)

# ---------------- ADD MORE FEATURES ----------------

def clean_revision(x):
    x = str(x).lower()
    if 'daily' in x: return 4
    elif 'weekly' in x: return 3
    elif 'sometimes' in x: return 2
    elif 'rare' in x: return 1
    else: return 2

def clean_homework(x):
    x = str(x).lower()
    if 'always' in x or 'yes' in x: return 2
    elif 'sometimes' in x: return 1
    elif 'no' in x: return 0
    else: return 1

def clean_study_method(x):
    x = str(x).lower()
    if 'self' in x: return 1
    elif 'tuition' in x or 'coaching' in x: return 2
    elif 'online' in x or 'youtube' in x: return 3
    else: return 2

def clean_study_increase(x):
    x = str(x).lower()
    if 'yes' in x: return 2
    elif 'no' in x: return 0
    else: return 1

data['revision'] = data['How often do you revise your subjects?'].apply(clean_revision)
data['homework'] = data['Do you complete your homework on time?'].apply(clean_homework)
data['study_method'] = data['Which study method do you mostly use?'].apply(clean_study_method)
data['study_increase'] = data['Has your study time increased compared to last semester?'].apply(clean_study_increase)

# ---------------- ADD PREVIOUS MARKS ----------------
data['previous_marks'] = data['What were your MST-1 marks (%) in last semester?'].apply(clean_marks)

# ---------------- HANDLE MISSING ----------------
for col in ['study_hours','marks','attendance','revision','homework','study_method','study_increase','previous_marks']:
    data[col] = data[col].fillna(data[col].mean())

# ---------------- FEATURE ENGINEERING ----------------

# Improvement
data['improvement'] = data['marks'] - data['previous_marks']

# Performance ratio (avoid divide by zero)
data['performance_ratio'] = data['marks'] / (data['previous_marks'] + 1)

# Study efficiency
data['study_efficiency'] = data['marks'] / (data['study_hours'] + 1)

# Remove infinity / invalid
data = data.replace([float('inf'), -float('inf')], None)
data = data.dropna()

print("Rows AFTER cleaning:", len(data))

# ---------------- DESCRIPTIVE ----------------
print("\nAverage Study Hours:", data['study_hours'].mean())
print("Average Marks:", data['marks'].mean())

print("\n=== STATISTICAL SUMMARY ===")
print(data[['study_hours','marks','attendance']].describe())

# ---------------- CORRELATION ----------------
print("\n=== CORRELATION MATRIX ===")
corr = data[['study_hours','attendance','revision','homework','study_method',
             'study_increase','previous_marks','improvement',
             'performance_ratio','study_efficiency','marks']].corr()
print(corr)

# ---------------- HYPOTHESIS TESTING ----------------
print("\n=== HYPOTHESIS TESTING ===")

corr_value, p_value = pearsonr(data['study_hours'], data['marks'])

print("Correlation (Study Hours vs Marks):", corr_value)
print("P-value:", p_value)

if p_value < 0.05:
    print("Result: Significant relationship (Reject Null Hypothesis)")
else:
    print("Result: No significant relationship (Accept Null Hypothesis)")

# ---------------- REGRESSION ----------------
X = data[['study_hours']]
y = data['marks']

model = LinearRegression()
model.fit(X, y)

print("\n=== REGRESSION ===")
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

# ---------------- ML MODELS ----------------


# ---------------- ML MODELS ----------------

features = ['study_hours','attendance','revision','homework','study_method',
            'study_increase','previous_marks','improvement',
            'performance_ratio','study_efficiency']

X = data[features]
y = data['marks']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree
dt = DecisionTreeRegressor(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\n=== DECISION TREE ===")
print("Sample Predictions:", dt_pred[:5])

# Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n=== RANDOM FOREST ===")
print("Sample Predictions:", rf_pred[:5])
print("R2 Score:", r2_score(y_test, rf_pred))
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("Feature Importance:", rf.feature_importances_)
print("\n=== FEATURE IMPORTANCE (CLEAR VIEW) ===")
for name, value in zip(features, rf.feature_importances_):
    print(name, ":", round(value, 3))
# ---------------- GRAPHS ----------------

plt.scatter(data['study_hours'], data['marks'])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()

plt.scatter(data['attendance'], data['marks'])
plt.xlabel("Attendance")
plt.ylabel("Marks")
plt.title("Attendance vs Marks")
plt.show()

sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()

data['If your marks decreased, what is the main reason?'].value_counts().plot(kind='bar')
plt.title("Reasons for Low Performance")
plt.show()

sns.histplot(data['marks'], bins=10)
plt.title("Marks Distribution")
plt.show()

# ---------------- ML GRAPHS ----------------

plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, filled=True)
plt.title("Decision Tree")
plt.show()

plt.scatter(y_test, dt_pred)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Decision Tree: Actual vs Predicted")
plt.show()

plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Random Forest: Actual vs Predicted")
plt.show()

plt.scatter(y_test, dt_pred, label="Decision Tree")
plt.scatter(y_test, rf_pred, label="Random Forest")
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.legend()
plt.title("Model Comparison")
plt.show()