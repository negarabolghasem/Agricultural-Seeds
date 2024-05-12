#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


df= pd.read_csv('Case_2_Dataset_DecTree.csv')


# ## Dataset is provided for the data about an agricultural seeds that contains some Dimensional factors and some shape factors. The purpose is to identify the class of the product

# In[3]:


df.head(5)


# #### Different type of seeds:

# In[4]:


classes = df['Class'].unique()
print("Unique classes:", classes)


# #### Finding the non-value itmes and treat them

# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[4]:


df.shape


# In[5]:


import numpy as np
df = df.drop_duplicates()
df.drop_duplicates(inplace=True)
df.shape


# #### Summary view of data and related distributions 

# In[9]:


df.describe()


# #### Scatter Plots for different Classes

# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue='Class', vars=[ 'DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor5', 'DFactor6', 'DFactor7', 'DFactor8', 'DFactor9'])
plt.savefig('pairplotDF.png')
plt.show()


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue='Class', vars=[ 'Area', 'Perimeter', 'MajorAxisLength'])
plt.savefig('pairplototh.png')
plt.show()


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue='Class', vars=[ 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3','ShapeFactor4'])
plt.savefig('pairplotSha.png')
plt.show()


# #### Distribution of different factors

# In[13]:


df.hist(['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor5', 'DFactor6', 'DFactor7', 'DFactor8', 'DFactor9'], figsize=(18,10))


# In[14]:


df.hist(['Area', 'Perimeter', 'MajorAxisLength'], figsize=(18,10))


# In[15]:


df.hist([ 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3','ShapeFactor4'], figsize=(18,10))


# In[16]:


features = ['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor5', 'DFactor6', 'DFactor7', 'DFactor8', 'DFactor9']
grouped = df.groupby('Class')

# Create subplots with three rows, one for each feature
fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(10, 35))

# Set different colors for each class
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# Iterate over features
for i, feature in enumerate(features):
    ax = axes[i]
    
    # Check if this is the top row of subplots
    if i == 0:
        ax.set_title("Feature Histograms")
    else:
        ax.set_title("")  # Clear title for other subplots
    
    # Iterate over all classes and plot histograms
    for (class_name, group), color in zip(grouped, colors):
        group[feature].plot(kind='hist', ax=ax, color=color, alpha=0.6)
    
    ax.set_xlabel(feature)
    
    # Check if a legend exists and remove it if present
    if ax.get_legend():
        ax.get_legend().remove()
plt.savefig('freq.png')
plt.show()


# In[17]:


# Define the features you want to create histograms for
features = ['Area', 'Perimeter', 'MajorAxisLength']

# Create subplots with three rows, one for each feature
fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(6, 15))

# Set different colors for each class
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# Iterate over features
for i, feature in enumerate(features):
    ax = axes[i]
    ax.set_title(feature)
    
    # Iterate over all classes and plot histograms
    for (class_name, group), color in zip(grouped, colors):
        group[feature].plot(kind='hist', ax=ax, label=class_name, color=color, alpha=0.6)
    
    ax.set_xlabel(feature)
    ax.legend()
plt.savefig('freqaother.png')
plt.show()


# In[18]:


# Define the features you want to create histograms for
features = [ 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3','ShapeFactor4']

# Create subplots with three rows, one for each feature
fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(6, 15))

# Set different colors for each class
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# Iterate over features
for i, feature in enumerate(features):
    ax = axes[i]
    ax.set_title(feature)
    
    # Iterate over all classes and plot histograms
    for (class_name, group), color in zip(grouped, colors):
        group[feature].plot(kind='hist', ax=ax, label=class_name, color=color, alpha=0.6)
    
    ax.set_xlabel(feature)
    ax.legend()
plt.savefig('freqshape.png')
plt.show()


# #### Outlier treatment

# In[19]:


# Define the list of features you want to create box plots for
features = ['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor5', 'DFactor6', 'DFactor7', 'DFactor8', 'DFactor9']

# Calculate the number of rows needed based on the number of features (2 features per row)
num_rows = len(features) // 2 + (len(features) % 2)  # Add 1 row if there's an odd number of features

# Create subplots with the calculated number of rows
fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 10))

# Iterate over features and create box-and-whisker plots, arranging them in pairs in different rows
for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    sns.boxplot(x='Class', y=feature, data=df, ax=ax)
    ax.set_title(f'Box Plot for {feature}')
    ax.set_ylabel('')
    ax.set_xlabel('Class')

# Remove any empty subplots
for i in range(len(features), num_rows * 2):
    fig.delaxes(axes[i // 2, i % 2])

# Adjust the layout
plt.tight_layout()
plt.savefig('outdf.png')
plt.show()



# In[20]:


# Define the list of features you want to create box plots for
features = ['Area', 'Perimeter', 'MajorAxisLength']
# Calculate the number of rows needed based on the number of features (2 features per row)
num_rows = len(features) // 2 + (len(features) % 2)  # Add 1 row if there's an odd number of features

# Create subplots with the calculated number of rows
fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 10))

# Iterate over features and create box-and-whisker plots, arranging them in pairs in different rows
for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    sns.boxplot(x='Class', y=feature, data=df, ax=ax)
    ax.set_title(f'Box Plot for {feature}')
    ax.set_ylabel('')
    ax.set_xlabel('Class')

# Remove any empty subplots
for i in range(len(features), num_rows * 2):
    fig.delaxes(axes[i // 2, i % 2])

# Adjust the layout
plt.tight_layout()
plt.savefig('outA.png')
plt.show()


# In[21]:


# Define the list of features you want to create box plots for
features = [ 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3','ShapeFactor4']
# Calculate the number of rows needed based on the number of features (2 features per row)
num_rows = len(features) // 2 + (len(features) % 2)  # Add 1 row if there's an odd number of features

# Create subplots with the calculated number of rows
fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 10))

# Iterate over features and create box-and-whisker plots, arranging them in pairs in different rows
for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    sns.boxplot(x='Class', y=feature, data=df, ax=ax)
    ax.set_title(f'Box Plot for {feature}')
    ax.set_ylabel('')
    ax.set_xlabel('Class')

# Remove any empty subplots
for i in range(len(features), num_rows * 2):
    fig.delaxes(axes[i // 2, i % 2])

# Adjust the layout
plt.tight_layout()
plt.savefig('outsh.png')
plt.show()


# In[23]:


# Define the columns want to check for outliers
columns_to_check = ['Area', 'Perimeter', 'MajorAxisLength', 'DFactor1', 'DFactor2',
       'DFactor3', 'DFactor4', 'DFactor5', 'DFactor7', 'DFactor6','DFactor8',
       'DFactor9', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3',
       'ShapeFactor4']
# Define the remove_outliers function
def remove_outliers(data, columns):
    data_no_outliers = data.copy()
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        # Adjust this multiplier to control the outlier range
        IQR_multiplier = 1.5

        lower_bound = Q1 - IQR_multiplier * IQR
        upper_bound = Q3 + IQR_multiplier * IQR

        # Filter and remove outliers for the current column
        data_no_outliers = data_no_outliers[(data_no_outliers[column] >= lower_bound) &
                                            (data_no_outliers[column] <= upper_bound)]

    return data_no_outliers

# Remove outliers from specific columns
df_no_outliers = remove_outliers(df, columns_to_check)

# Optional: Save the cleaned dataset
df_no_outliers.to_csv('3rdversion_nooutliers_Final.csv', index=False)


# In[13]:


# Define a function to impute outliers with the mean
def impute_outliers_with_mean(series):
    mean = series.mean()
    std = series.std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    series = series.apply(lambda x: x if lower_bound <= x <= upper_bound else mean)
    return series

# Columns to impute outliers
columns_to_impute = ["Area", "Perimeter", "MajorAxisLength", "DFactor1", "DFactor2", "DFactor3",
                     "DFactor4", "DFactor5", "DFactor6", "DFactor7", "DFactor8", "DFactor9",
                     "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"]

# Impute outliers for the specified columns
for col in columns_to_impute:
    df[col] = impute_outliers_with_mean(df[col])

# Save the new DataFrame with imputed outliers to a new CSV file
output_file = "df-impute_outliers03.csv"
df.to_csv(output_file, index=False)

print(f"Data with imputed outliers saved to {output_file}")


# In[14]:


df_impute_outliers= pd.read_csv('df-impute_outliers03.csv')


# ## Removing Redundant features by analyzing their correlation with other features and Targer variable:
# 
# - Correlation of different features and target variable

# In[18]:


from scipy.stats import f_oneway, kruskal
df_impute_outliers['Class'] = df_impute_outliers['Class'].astype('category').cat.codes

# Group your data by the target variable
grouped_data = [df_impute_outliers[df_impute_outliers['Class'] == category] for category in df_impute_outliers['Class'].unique()]

# Create a dictionary to map class indices to class names
class_names = {0: 'SE', 1: 'BA', 2: 'BO', 3: 'CA', 4: 'HO', 5: 'SI', 6: 'DE'}

# Perform ANOVA or Kruskal-Wallis test for each feature
features = ['Area', 'Perimeter', 'MajorAxisLength', 'DFactor1', 'DFactor2',
       'DFactor3', 'DFactor4', 'DFactor5', 'DFactor6', 'DFactor7', 'DFactor8',
       'DFactor9', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3',
       'ShapeFactor4']

for feature in features:
    print(f"Feature: {feature}")
    for category, group in zip(df_impute_outliers['Class'].unique(), grouped_data):
        values = group[feature]
        if len(df['Class'].unique()) > 2:
            # Use Kruskal-Wallis test for non-parametric data
            stat, p_value = kruskal(*[values for values in grouped_data])
        else:
            # Use ANOVA for parametric data
            stat, p_value = f_oneway(*[values for values in grouped_data])
        
        # Convert the p-value from an array to a scalar and then format it
        p_value_scalar = p_value[0]  # Assuming p_value is a 1-element numpy array
        class_name = class_names[category]
        print(f"{class_name}: p-value = {p_value_scalar:.4f}")


# - Correlation between different features

# In[19]:


# Select the numerical features for which we want to calculate the correlation
numerical_features = ['Area', 'Perimeter', 'MajorAxisLength', 'DFactor1', 'DFactor2',
                      'DFactor3', 'DFactor4', 'DFactor5', 'DFactor6', 'DFactor7', 'DFactor8',
                      'DFactor9', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']

# Calculate the correlation matrix
correlation_matrix = df_impute_outliers[numerical_features].corr()

# Display the correlation matrix
print(correlation_matrix)


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt
correlation_matrix = df_impute_outliers[numerical_features].corr()
# Set the figure size to make the heatmap larger
plt.figure(figsize=(12, 10))

# Create a heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# Customize the heatmap, if needed
plt.title("Correlation Heatmap")
plt.show()


# In[22]:


SelectedColumns=['Area', 'DFactor1', 'DFactor2',
                      'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8',
                      'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4','Class']

# Selecting final columns
DataForML=df_impute_outliers[SelectedColumns]
DataForML.head()


# In[23]:


DataForML.to_csv('FinalVer_imputeoutliersandcorrelation.csv', index=False)


# In[24]:


DataForML.head(5)


# In[25]:


newdata=pd.read_csv('FinalVer_imputeoutliersandcorrelation.csv')
newdata.head(5)


# #### Scaling Data

# In[26]:


from sklearn.preprocessing import StandardScaler

# Create a StandardScaler instance
scaler = StandardScaler()

features = ['Area', 'DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']

# Fit and transform the selected features using the scaler
scaled_features = scaler.fit_transform(newdata[features])

# Use .loc to assign the scaled values back to the DataFrame
newdata.loc[:, features] = scaled_features


# ## Decision Tree
# #### Finding the best combination of feautres for Decition Tree :

# In[29]:


from itertools import combinations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

X = newdata[['Area','DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']]
y = newdata['Class'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=428)

# Create lists of different feature sets (example combinations)
feature_sets = [
    ['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 'DFactor9'],
    ['Area','ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'],
    ['ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'],
    ['Area','DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 'DFactor9'],
    ['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 'DFactor9','ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4','Area']
]

# Define a list of different hyperparameters to try
hyperparameters = [
    {'max_depth': 5, 'min_samples_split': 2},
    {'max_depth': 10, 'min_samples_split': 2},
    {'max_depth': 7, 'min_samples_split': 2},
    {'max_depth': 8, 'min_samples_split': 3},
    {'max_depth': 8, 'min_samples_split': 2}
]

best_model = None
best_feature_set = None
best_score = 0

# Iterate over feature sets
for features in feature_sets:
    X_train_subset = X_train[features]
    
    # Iterate over hyperparameters
    for params in hyperparameters:
        clf = DecisionTreeClassifier(**params)
        
        # Evaluate the model using cross-validation
        scores = cross_val_score(clf, X_train_subset, y_train, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = clf
            best_feature_set = features

# Train the best model on the entire training set with the best feature set
best_model.fit(X_train[best_feature_set], y_train)

# Evaluate the model on the test set
y_pred = best_model.predict(X_test[best_feature_set])
test_accuracy = accuracy_score(y_test, y_pred)
test_f1_score = f1_score(y_test, y_pred, average='weighted')

print("Best feature set:", best_feature_set)
print("Best model hyperparameters:", best_model.get_params())
print("Test accuracy:", test_accuracy)
print("Test F1 score:", test_f1_score)


# In[38]:


import matplotlib.pyplot as plt

# Define lists to store results
feature_set_names = []  # Names of feature sets
accuracy_scores = []    # Accuracy scores for each try

# Iterate over feature sets
for features in feature_sets:
    X_train_subset = X_train[features]
    
    # Iterate over hyperparameters
    for params in hyperparameters:
        clf = DecisionTreeClassifier(**params)
        
        # Evaluate the model using cross-validation (you can also use train_test_split)
        scores = cross_val_score(clf, X_train_subset, y_train, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        
        # Store the results
        feature_set_names.append(', '.join(features))
        accuracy_scores.append(mean_score)

# Create a bar chart to visualize the results
plt.figure(figsize=(12, 4))
plt.barh(feature_set_names, accuracy_scores, color='skyblue')
plt.xlabel('Accuracy')
plt.title('Performance of Different Feature Sets and Models')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()

# Show the plot
plt.show()


# In[40]:


X = newdata[['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Area']]
y = newdata['Class']

# Split your data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=428)

# Create lists to store training and test errors for different max_depth values
max_depths = list(range(1, 21))
train_errors = []
test_errors = []

# Train the model for different max_depth values and record the errors
for depth in max_depths:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    
    # Training error
    y_train_pred = clf.predict(X_train)
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    train_errors.append(train_error)
    
    # Test error
    y_test_pred = clf.predict(X_test)
    test_error = 1 - accuracy_score(y_test, y_test_pred)
    test_errors.append(test_error)

# Find the best depth with minimum test error
best_depth = max_depths[np.argmin(test_errors)]

# Create a plot to visualize the errors and annotate the best depth
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_errors, label="Training Error", marker='o')
plt.plot(max_depths, test_errors, label="Test Error", marker='o')
plt.xlabel("Max Depth")
plt.ylabel("Error Rate")
plt.title("Training and Test Error vs. Max Depth")
plt.legend()
plt.grid()
plt.annotate(f'Best Depth: {best_depth}', (best_depth, min(test_errors)), textcoords="offset points", xytext=(0, 10), ha='center')
plt.scatter(best_depth, min(test_errors), color='red', marker='*')
plt.show()


# ### Creating Validation data

# In[41]:


X = newdata[['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Area']]
y = newdata['Class']


# Split the data into a training set, a test set, and a validation set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create a Decision Tree classifier
clf = DecisionTreeClassifier(max_depth=8)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = clf.predict(X_test)

# Make predictions on the validation set
y_val_pred = clf.predict(X_val)

# Evaluate the model on the test set and validation set
test_accuracy = accuracy_score(y_test, y_test_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Test Set Accuracy: {test_accuracy:.2f}")
print(f"Validation Set Accuracy: {val_accuracy:.2f}")


# In[44]:


# Define the range of tree depths to test
max_depths = np.arange(1, 21)  # Adjust the range as needed

# Lists to store training and validation scores
train_scores = []
val_scores = []

# Split the data into a training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Iterate over different tree depths
for depth in max_depths:
    # Create a Decision Tree classifier with the specified depth
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    
    # Fit the classifier to the training data
    clf.fit(X_train, y_train)
    
    # Make predictions on the training and validation sets
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    
    # Calculate training and validation scores
    train_score = accuracy_score(y_train, y_train_pred)
    val_score = accuracy_score(y_val, y_val_pred)
    
    # Append scores to the respective lists
    train_scores.append(train_score)
    val_scores.append(val_score)

# Plot the training and validation scores
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_scores, label='Training Score', marker='o')
plt.plot(max_depths, val_scores, label='Validation Score', marker='o')
plt.xlabel('Max Depth of Decision Tree')
plt.ylabel('Accuracy Score')
plt.title('Training and Validation Scores vs. Max Depth')
plt.legend()
plt.grid()
plt.show()


# In[68]:


def plot_learning_curves(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    # Annotate the plot with accuracy labels
    for i, acc in enumerate(test_scores_mean):
        plt.annotate(f'Accuracy={acc:.2f}', (train_sizes[i], test_scores_mean[i]), fontsize=8, ha='right', va='bottom')
    
    plt.legend(loc="best")
    return plt

# Specify your estimator (e.g., DecisionTreeClassifier)
estimator = DecisionTreeClassifier(max_depth=8, min_samples_split=3)

# Plot learning curves with accuracy labels
plot_learning_curves(estimator, "Learning Curves (Decision Tree)", X, y, ylim=(0.7, 1.01), cv=5, n_jobs=-1)

plt.show()


# In[70]:


def plot_learning_curves(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 15)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    # Annotate the plot with accuracy labels
    for i, acc in enumerate(test_scores_mean):
        plt.annotate(f'Accuracy={acc:.2f}', (train_sizes[i], test_scores_mean[i]), fontsize=8, ha='right', va='bottom')
    
    plt.legend(loc="best")
    return plt

# Specify your estimator (e.g., DecisionTreeClassifier)
estimator = DecisionTreeClassifier(max_depth=8, min_samples_split=3)

# Plot learning curves with accuracy labels
plot_learning_curves(estimator, "Learning Curves (Decision Tree)", X, y, ylim=(0.7, 1.01), cv=5, n_jobs=-1)

plt.show()


# In[73]:


def plot_learning_curves(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 20)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    # Annotate the plot with accuracy labels
    for i, acc in enumerate(test_scores_mean):
        plt.annotate(f'Accuracy={acc:.2f}', (train_sizes[i], test_scores_mean[i]), fontsize=8, ha='right', va='bottom')
    
    plt.legend(loc="best")
    return plt

# Specify your estimator (e.g., DecisionTreeClassifier)
estimator = DecisionTreeClassifier(max_depth=8, min_samples_split=3)

# Plot learning curves with accuracy labels
plot_learning_curves(estimator, "Learning Curves (Decision Tree)", X, y, ylim=(0.7, 1.01), cv=5, n_jobs=-1)

plt.show()


# ### Decision Boundaries

# In[81]:


X = newdata[['ShapeFactor4', 'ShapeFactor2']]
y = newdata['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=428)
model = DecisionTreeClassifier(max_depth=8,min_samples_split=3 )
model.fit(X_train, y_train)
# Create a meshgrid of SepalLengthCm and SepalWidthCm values
xx, yy = np.meshgrid(np.arange(X['ShapeFactor4'].min() - 1, X['ShapeFactor4'].max() + 1, 0.01),
                     np.arange(X['ShapeFactor2'].min() - 1, X['ShapeFactor2'].max() + 1, 0.01))

# Use the model to make predictions on the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# Plot the data points
plt.scatter(X['ShapeFactor4'], X['ShapeFactor2'], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('ShapeFactor4')
plt.ylabel('ShapeFactor2')
plt.title('DesicionTree Decision Boundary-depth8 & Sample Split3')

# Show the plot
plt.show()


# In[86]:


X = newdata[['ShapeFactor4', 'ShapeFactor2']]
y = newdata['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=428)
model = DecisionTreeClassifier(max_depth=3,min_samples_split=3 )
model.fit(X_train, y_train)
# Create a meshgrid of SepalLengthCm and SepalWidthCm values
xx, yy = np.meshgrid(np.arange(X['ShapeFactor4'].min() - 1, X['ShapeFactor4'].max() + 1, 0.01),
                     np.arange(X['ShapeFactor2'].min() - 1, X['ShapeFactor2'].max() + 1, 0.01))

# Use the model to make predictions on the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# Plot the data points
plt.scatter(X['ShapeFactor4'], X['ShapeFactor2'], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('ShapeFactor4')
plt.ylabel('ShapeFactor2')
plt.title('DesicionTree Decision Boundary-depth3 & Sample Split3')

# Show the plot
plt.show()


# In[82]:


X = newdata[['DFactor1', 'DFactor2']]
y = newdata['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=428)
model = DecisionTreeClassifier(max_depth=8,min_samples_split=3 )
model.fit(X_train, y_train)
# Create a meshgrid of SepalLengthCm and SepalWidthCm values
xx, yy = np.meshgrid(np.arange(X['DFactor1'].min() - 1, X['DFactor1'].max() + 1, 0.01),
                     np.arange(X['DFactor2'].min() - 1, X['DFactor2'].max() + 1, 0.01))

# Use the model to make predictions on the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# Plot the data points
plt.scatter(X['DFactor1'], X['DFactor2'], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('DFactor1')
plt.ylabel('DFactor2')
plt.title('DesicionTree Decision Boundary-depth8 & Sample Split3')

# Show the plot
plt.show()


# In[84]:


X = newdata[['DFactor1', 'DFactor2']]
y = newdata['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=428)
model = DecisionTreeClassifier(max_depth=3,min_samples_split=3 )
model.fit(X_train, y_train)
# Create a meshgrid of SepalLengthCm and SepalWidthCm values
xx, yy = np.meshgrid(np.arange(X['DFactor1'].min() - 1, X['DFactor1'].max() + 1, 0.01),
                     np.arange(X['DFactor2'].min() - 1, X['DFactor2'].max() + 1, 0.01))

# Use the model to make predictions on the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# Plot the data points
plt.scatter(X['DFactor1'], X['DFactor2'], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('DFactor1')
plt.ylabel('DFactor2')
plt.title('DesicionTree Decision Boundary-depth3 & Sample Split3')

# Show the plot
plt.show()


# ### RandomForest

# In[2]:


newdatarandom=pd.read_csv('FinalVer_imputeoutliersandcorrelation.csv')
newdatarandom.head(5)


# In[3]:


from sklearn.preprocessing import StandardScaler

# Create a StandardScaler instance
scaler = StandardScaler()

features = ['Area', 'DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']

# Fit and transform the selected features using the scaler
scaled_features = scaler.fit_transform(newdatarandom[features])

# Use .loc to assign the scaled values back to the DataFrame
newdatarandom.loc[:, features] = scaled_features


# In[4]:


newdatarandom.head()


# In[5]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X = newdatarandom[['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Area']]
y = newdatarandom['Class']

model = RandomForestClassifier(n_estimators=100, random_state=428)
model.fit(X, y)

# Get feature importances
feature_importance = model.feature_importances_

# Create a bar chart to visualize feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance, tick_label=X.columns)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.xticks(rotation=45)
plt.show()


# ### Hyperparameters Tuning

# In[7]:


X = newdatarandom[['DFactor1', 'DFactor4','DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'Area']]
y = newdatarandom['Class']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42, max_depth= 10,
                                       min_samples_split=2, min_samples_leaf=2)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)


# In[8]:


X = newdatarandom[['DFactor1', 'DFactor4','DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'Area']]
y = newdatarandom['Class']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=400, random_state=42, max_depth= 9,
                                       min_samples_split=3, min_samples_leaf=4)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)


# In[9]:


X = newdatarandom[['DFactor1', 'DFactor4','DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'Area']]
y = newdatarandom['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=400, random_state=42, max_depth= 9,
                                       min_samples_split=3, min_samples_leaf=4)  # You can adjust hyperparameters here
clf.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report
print(classification_report(y_test, y_pred))


# #### Decision Boundaries

# In[13]:


from sklearn.ensemble import RandomForestClassifier

X = newdatarandom[['DFactor4','DFactor9']]
y = newdatarandom['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=400, random_state=42, max_depth= 9,
                                       min_samples_split=3, min_samples_leaf=4)
model.fit(X_train, y_train)

xx, yy = np.meshgrid(np.arange(X['DFactor4'].min() - 1, X['DFactor4'].max() + 1, 0.01),
                     np.arange(X['DFactor9'].min() - 1, X['DFactor9'].max() + 1, 0.01))

# Use the model to make predictions on the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# Plot the data points
plt.scatter(X['DFactor4'], X['DFactor9'], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('DFactor4')
plt.ylabel('DFactor9')
plt.title('RandomForest Decision Boundary DF4 and DF9')

# Show the plot
plt.show()


# In[15]:


from sklearn.ensemble import RandomForestClassifier

X = newdatarandom[['ShapeFactor2', 'ShapeFactor3']]
y = newdatarandom['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=400, random_state=42, max_depth= 9,
                                       min_samples_split=3, min_samples_leaf=4)
model.fit(X_train, y_train)

xx, yy = np.meshgrid(np.arange(X['ShapeFactor2'].min() - 1, X['ShapeFactor2'].max() + 1, 0.01),
                     np.arange(X['ShapeFactor3'].min() - 1, X['ShapeFactor3'].max() + 1, 0.01))

# Use the model to make predictions on the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# Plot the data points
plt.scatter(X['ShapeFactor2'], X['ShapeFactor3'], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('ShapeFactor2')
plt.ylabel('ShapeFactor3')
plt.title('RandomForest Decision Boundary SHF2 and SHF3')

# Show the plot
plt.show()


# In[16]:


from sklearn.ensemble import RandomForestClassifier

X = newdatarandom[['DFactor1', 'Area']]
y = newdatarandom['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=400, random_state=42, max_depth= 9,
                                       min_samples_split=3, min_samples_leaf=4)
model.fit(X_train, y_train)

xx, yy = np.meshgrid(np.arange(X['DFactor1'].min() - 1, X['DFactor1'].max() + 1, 0.01),
                     np.arange(X['Area'].min() - 1, X['Area'].max() + 1, 0.01))

# Use the model to make predictions on the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# Plot the data points
plt.scatter(X['DFactor1'], X['Area'], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('DFactor1')
plt.ylabel('Area')
plt.title('RandomForest Decision Boundary DF1 and Area')

# Show the plot
plt.show()


# ### KNN

# In[2]:


newdataknn=pd.read_csv('FinalVer_imputeoutliersandcorrelation.csv')
newdataknn.head(5)


# In[3]:


from sklearn.preprocessing import StandardScaler

# Create a StandardScaler instance
scaler = StandardScaler()

features = ['Area', 'DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']

# Fit and transform the selected features using the scaler
scaled_features = scaler.fit_transform(newdataknn[features])

# Use .loc to assign the scaled values back to the DataFrame
newdataknn.loc[:, features] = scaled_features


# In[4]:


newdataknn.head()


# In[10]:


from sklearn.neighbors import KNeighborsClassifier
X = newdataknn[['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 
                'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Area']]
y = newdataknn['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid= train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model_choices=[]
valid_acc=[]
n_valid=y_valid.shape[0]
for k in range (1,11):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_valid_pred= knn.predict(X_valid)
    accuracy= np.sum(y_valid_pred==y_valid)/n_valid
    model_choices.append(k)
    valid_acc.append(accuracy)
    
best_valid_K= model_choices[valid_acc.index(max(valid_acc))]
knn= KNeighborsClassifier(n_neighbors=best_valid_K)
knn.fit(X_train, y_train)
y_test_pred= knn.predict(X_test)
accuracy= np.sum(y_test_pred==y_test)/y_test.shape[0]
print("Accuracy:",test_accuracy)


# In[8]:


plt.clf()
plt.plot(model_choices, valid_acc, marker='o', color='blue', label='validation')
plt.plot(best_valid_K, test_accuracy, marker='*', color='red', label='testing')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.savefig('training_validation_testing.png',bbox_inches='tight',dpi=300)


# In[16]:


from sklearn.metrics import accuracy_score, classification_report

X = newdataknn[['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 
                'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Area']]
y = newdataknn['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Set k to 10 (the number of neighbors)
k = 10

# Create and fit the KNeighborsClassifier with k=10
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = knn.predict(X_test)
y_pred = knn.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report
print(classification_report(y_test, y_pred))


# ### Deciosion Boundaries

# In[5]:


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Select the features
X = newdataknn[['DFactor1', 'DFactor2']]

# Target variable
y = newdataknn['Class']

# Train a KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Create a meshgrid of points to cover the feature space
x_min, x_max = X['DFactor1'].min() - 1, X['DFactor1'].max() + 1
y_min, y_max = X['DFactor2'].min() - 1, X['DFactor2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Use the model to make predictions on the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# Plot the data points
plt.scatter(X['DFactor1'], X['DFactor2'], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('DFactor1')
plt.ylabel('DFactor2')
plt.title('KNN Decision Boundary (k=3)')

# Show the plot
plt.show()


# In[7]:


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Select the features
X = newdataknn[['DFactor1', 'DFactor2']]

# Target variable
y = newdataknn['Class']

# Train a KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X, y)

# Create a meshgrid of points to cover the feature space
x_min, x_max = X['DFactor1'].min() - 1, X['DFactor1'].max() + 1
y_min, y_max = X['DFactor2'].min() - 1, X['DFactor2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Use the model to make predictions on the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# Plot the data points
plt.scatter(X['DFactor1'], X['DFactor2'], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('DFactor1')
plt.ylabel('DFactor2')
plt.title('KNN Decision Boundary (k=10)')

# Show the plot
plt.show()


# In[9]:


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Select the features
X = newdataknn[['ShapeFactor4', 'DFactor8']]

# Target variable
y = newdataknn['Class']

# Train a KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Create a meshgrid of points to cover the feature space
x_min, x_max = X['ShapeFactor4'].min() - 1, X['ShapeFactor4'].max() + 1
y_min, y_max = X['DFactor8'].min() - 1, X['DFactor8'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Use the model to make predictions on the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# Plot the data points
plt.scatter(X['ShapeFactor4'], X['DFactor8'], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('ShapeFactor4')
plt.ylabel('DFactor8')
plt.title('KNN Decision Boundary (k=3)')

# Show the plot
plt.show()


# In[11]:


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Select the features
X = newdataknn[['ShapeFactor4', 'DFactor8']]

# Target variable
y = newdataknn['Class']

# Train a KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X, y)

# Create a meshgrid of points to cover the feature space
x_min, x_max = X['ShapeFactor4'].min() - 1, X['ShapeFactor4'].max() + 1
y_min, y_max = X['DFactor8'].min() - 1, X['DFactor8'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Use the model to make predictions on the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# Plot the data points
plt.scatter(X['ShapeFactor4'], X['DFactor8'], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('ShapeFactor4')
plt.ylabel('DFactor8')
plt.title('KNN Decision Boundary (k=10)')

# Show the plot
plt.show()


# In[12]:


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Select the features
X = newdataknn[['Area', 'ShapeFactor3']]

# Target variable
y = newdataknn['Class']

# Train a KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Create a meshgrid of points to cover the feature space
x_min, x_max = X['Area'].min() - 1, X['Area'].max() + 1
y_min, y_max = X['ShapeFactor3'].min() - 1, X['ShapeFactor3'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Use the model to make predictions on the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# Plot the data points
plt.scatter(X['Area'], X['ShapeFactor3'], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('Area')
plt.ylabel('ShapeFactor3')
plt.title('KNN Decision Boundary (k=3)')

# Show the plot
plt.show()


# In[13]:


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Select the features
X = newdataknn[['Area', 'ShapeFactor3']]

# Target variable
y = newdataknn['Class']

# Train a KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X, y)

# Create a meshgrid of points to cover the feature space
x_min, x_max = X['Area'].min() - 1, X['Area'].max() + 1
y_min, y_max = X['ShapeFactor3'].min() - 1, X['ShapeFactor3'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Use the model to make predictions on the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# Plot the data points
plt.scatter(X['Area'], X['ShapeFactor3'], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('Area')
plt.ylabel('ShapeFactor3')
plt.title('KNN Decision Boundary (k=10)')

# Show the plot
plt.show()


# ### Logistic Regression

# In[2]:


newdatalogistic=pd.read_csv('FinalVer_imputeoutliersandcorrelation.csv')
newdatalogistic.head(5)


# In[4]:


from sklearn.preprocessing import StandardScaler

# Create a StandardScaler instance
scaler = StandardScaler()

features = ['Area', 'DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']

# Fit and transform the selected features using the scaler
scaled_features = scaler.fit_transform(newdatalogistic[features])

# Use .loc to assign the scaled values back to the DataFrame
newdatalogistic.loc[:, features] = scaled_features


# In[6]:


newdatalogistic.head()


# In[8]:


from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define your feature sets
X = newdatalogistic[['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 
                'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Area']]
y = newdatalogistic['Class']
# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.2, random_state=42)

# Define the logistic regression model
logistic_reg = LogisticRegression()

# Define a grid of hyperparameters to search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
    'penalty': ['l1', 'l2'],  # Regularization penalty type
}

# Use grid search to find the best combination of features and hyperparameters
grid_search = GridSearchCV(logistic_reg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and features
best_params = grid_search.best_params_
best_features = features

# Create the final model with the best features and hyperparameters
final_model = LogisticRegression(**best_params)
final_model.fit(X_train, y_train)

# Evaluate the final model using cross-validation or on the test set
cross_val_scores = cross_val_score(final_model, X_train, y_train, cv=5, scoring='accuracy')
test_accuracy = accuracy_score(y_test, final_model.predict(X_test))

# Print the results
print("Best Features:", best_features)
print("Best Hyperparameters:", best_params)
print("Cross-Validation Accuracy:", cross_val_scores.mean())
print("Test Set Accuracy:", test_accuracy)


# In[9]:


from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = newdatalogistic[['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 
                'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Area']]
y = newdatalogistic['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

C = 100
penalty = 'l2'  # 'l2' is Ridge regularization
logistic_reg = LogisticRegression(C=C, penalty=penalty, random_state=42)

# Train the model on the training data
logistic_reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logistic_reg.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display the classification report
print(classification_report(y_test, y_pred))


# In[10]:


X = newdatalogistic[['DFactor1', 'DFactor2']]

# Target variable
y = newdatalogistic['Class']

# Train a logistic regression model
logistic_reg = LogisticRegression(C=100, penalty='l2', random_state=42)
logistic_reg.fit(X, y)

# Create a meshgrid of points to cover the feature space
x_min, x_max = X['DFactor1'].min() - 1, X['DFactor1'].max() + 1
y_min, y_max = X['DFactor2'].min() - 1, X['DFactor2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Use the model to make predictions on the meshgrid
Z = logistic_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# Plot the data points
plt.scatter(X['DFactor1'], X['DFactor2'], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('DFactor1')
plt.ylabel('DFactor2')
plt.title('Logistic Regression Decision Boundary')

# Show the plot
plt.show()


# In[12]:


X = newdatalogistic[['DFactor8', 'DFactor7']]

# Target variable
y = newdatalogistic['Class']

# Train a logistic regression model
logistic_reg = LogisticRegression(C=100, penalty='l2', random_state=42)
logistic_reg.fit(X, y)

# Create a meshgrid of points to cover the feature space
x_min, x_max = X['DFactor8'].min() - 1, X['DFactor8'].max() + 1
y_min, y_max = X['DFactor7'].min() - 1, X['DFactor7'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Use the model to make predictions on the meshgrid
Z = logistic_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# Plot the data points
plt.scatter(X['DFactor8'], X['DFactor7'], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('DFactor8')
plt.ylabel('DFactor7')
plt.title('Logistic Regression Decision Boundary')

# Show the plot
plt.show()


# In[13]:


X = newdatalogistic[['Area', 'DFactor3']]

# Target variable
y = newdatalogistic['Class']

# Train a logistic regression model
logistic_reg = LogisticRegression(C=100, penalty='l2', random_state=42)
logistic_reg.fit(X, y)

# Create a meshgrid of points to cover the feature space
x_min, x_max = X['Area'].min() - 1, X['Area'].max() + 1
y_min, y_max = X['DFactor3'].min() - 1, X['DFactor3'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Use the model to make predictions on the meshgrid
Z = logistic_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# Plot the data points
plt.scatter(X['Area'], X['DFactor3'], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('Area')
plt.ylabel('DFactor3')
plt.title('Logistic Regression Decision Boundary')

# Show the plot
plt.show()


# ### Model Comparison

# In[8]:


newdataComparison=pd.read_csv('FinalVer_imputeoutliersandcorrelation.csv')
newdataComparison.head(5)


# In[3]:


from sklearn.preprocessing import StandardScaler

# Create a StandardScaler instance
scaler = StandardScaler()

features = ['Area', 'DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']

# Fit and transform the selected features using the scaler
scaled_features = scaler.fit_transform(newdataComparison[features])

# Use .loc to assign the scaled values back to the DataFrame
newdataComparison.loc[:, features] = scaled_features


# In[4]:


newdataComparison.head()


# In[12]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Load your data and define X and y here

X = newdataComparison[['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 
                'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Area']]
y = newdataComparison['Class']

# Binarize the output
y = label_binarize(y, classes=range(y.max() + 1))
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]
pref = {}

def cross_validate(model, X_input, Y_output):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    y = np.zeros((X_input.shape[0], n_classes))
    yh = np.zeros((X_input.shape[0], n_classes))
    
    for train_index, test_index in kf.split(X_input):
        model = OneVsRestClassifier(model)
        model.fit(X_input.iloc[train_index], Y_output[train_index])
        y[test_index] = Y_output[test_index]
        yh[test_index] = model.predict_proba(X_input.iloc[test_index])
    
    return y, yh

for model in models:
    model_name = type(model).__name__
    print(model_name)
    label, pred = cross_validate(model, X_train, y_train)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    pref[model_name] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}

fig = plt.figure()
i = 0

for model_name, model_pref in pref.items():
    for i in range(n_classes):
        plt.plot(model_pref['fpr'][i], model_pref['tpr'][i])
    
    i += 1

plt.axline((0, 0), (1, 1), linestyle="--", lw=1, color="gray")
plt.legend(loc='upper center', bbox_to_anchor=(0.75, 0.6))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc_multimethods.png', bbox_inches='tight', dpi=300)



# In[17]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Load your data and define X and y here

X = newdataComparison[['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 
                'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Area']]
y = newdataComparison['Class']

# Binarize the output
y = label_binarize(y, classes=range(y.max() + 1))
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]
pref = {}

def cross_validate(model, X_input, Y_output):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    y = np.zeros((X_input.shape[0], n_classes))
    yh = np.zeros((X_input.shape[0], n_classes))
    
    for train_index, test_index in kf.split(X_input):
        model = OneVsRestClassifier(model)
        model.fit(X_input.iloc[train_index], Y_output[train_index])
        y[test_index] = Y_output[test_index]
        yh[test_index] = model.predict_proba(X_input.iloc[test_index])
    
    return y, yh

for model in models:
    model_name = type(model).__name__
    print(model_name)
    label, pred = cross_validate(model, X_train, y_train)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    pref[model_name] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}

fig = plt.figure()
i = 0

for model_name, model_pref in pref.items():
    for i in range(n_classes):
        plt.plot(model_pref['fpr'][i], model_pref['tpr'][i],label=model_name)
    
    i += 1

plt.axline((0, 0), (1, 1), linestyle="--", lw=1, color="gray")
plt.legend(loc='upper center', bbox_to_anchor=(0.75, 0.6))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc_multimethods.png', bbox_inches='tight', dpi=300)



# In[18]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Load your data and define X and y here

X = newdataComparison[['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 
                'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Area']]
y = newdataComparison['Class']

# Binarize the output
y = label_binarize(y, classes=range(y.max() + 1))
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [KNeighborsClassifier(), RandomForestClassifier()]
pref = {}

def cross_validate(model, X_input, Y_output):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    y = np.zeros((X_input.shape[0], n_classes))
    yh = np.zeros((X_input.shape[0], n_classes))
    
    for train_index, test_index in kf.split(X_input):
        model = OneVsRestClassifier(model)
        model.fit(X_input.iloc[train_index], Y_output[train_index])
        y[test_index] = Y_output[test_index]
        yh[test_index] = model.predict_proba(X_input.iloc[test_index])
    
    return y, yh

for model in models:
    model_name = type(model).__name__
    print(model_name)
    label, pred = cross_validate(model, X_train, y_train)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    pref[model_name] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}

fig = plt.figure()
i = 0

for model_name, model_pref in pref.items():
    for i in range(n_classes):
        plt.plot(model_pref['fpr'][i], model_pref['tpr'][i],label=model_name)
    
    i += 1

plt.axline((0, 0), (1, 1), linestyle="--", lw=1, color="gray")
plt.legend(loc='upper center', bbox_to_anchor=(0.75, 0.6))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc_multimethods.png', bbox_inches='tight', dpi=300)



# In[20]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Load your data and define X and y here

X = newdataComparison[['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 
                'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Area']]
y = newdataComparison['Class']

# Binarize the output
y = label_binarize(y, classes=range(y.max() + 1))
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [LogisticRegression(), KNeighborsClassifier()]
pref = {}

def cross_validate(model, X_input, Y_output):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    y = np.zeros((X_input.shape[0], n_classes))
    yh = np.zeros((X_input.shape[0], n_classes))
    
    for train_index, test_index in kf.split(X_input):
        model = OneVsRestClassifier(model)
        model.fit(X_input.iloc[train_index], Y_output[train_index])
        y[test_index] = Y_output[test_index]
        yh[test_index] = model.predict_proba(X_input.iloc[test_index])
    
    return y, yh

for model in models:
    model_name = type(model).__name__
    print(model_name)
    label, pred = cross_validate(model, X_train, y_train)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    pref[model_name] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}

fig = plt.figure()
i = 0

for model_name, model_pref in pref.items():
    for i in range(n_classes):
        plt.plot(model_pref['fpr'][i], model_pref['tpr'][i],label=model_name)
    
    i += 1

plt.axline((0, 0), (1, 1), linestyle="--", lw=1, color="gray")
plt.legend(loc='upper center', bbox_to_anchor=(0.75, 0.6))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc_multimethods.png', bbox_inches='tight', dpi=300)



# In[22]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Load your data and define X and y here

X = newdataComparison[['DFactor1', 'DFactor2', 'DFactor3', 'DFactor4', 'DFactor6', 'DFactor7', 'DFactor8', 
                'DFactor9', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Area']]
y = newdataComparison['Class']

# Binarize the output
y = label_binarize(y, classes=range(y.max() + 1))
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [KNeighborsClassifier(), DecisionTreeClassifier()]
pref = {}

def cross_validate(model, X_input, Y_output):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    y = np.zeros((X_input.shape[0], n_classes))
    yh = np.zeros((X_input.shape[0], n_classes))
    
    for train_index, test_index in kf.split(X_input):
        model = OneVsRestClassifier(model)
        model.fit(X_input.iloc[train_index], Y_output[train_index])
        y[test_index] = Y_output[test_index]
        yh[test_index] = model.predict_proba(X_input.iloc[test_index])
    
    return y, yh

for model in models:
    model_name = type(model).__name__
    print(model_name)
    label, pred = cross_validate(model, X_train, y_train)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    pref[model_name] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}

fig = plt.figure()
i = 0

for model_name, model_pref in pref.items():
    for i in range(n_classes):
        plt.plot(model_pref['fpr'][i], model_pref['tpr'][i])
    
    i += 1

plt.axline((0, 0), (1, 1), linestyle="--", lw=1, color="gray")
plt.legend(loc='upper center', bbox_to_anchor=(0.75, 0.6))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc_multimethods.png', bbox_inches='tight', dpi=300)



# In[ ]:




