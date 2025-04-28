
# **Real Estate Price Prediction Project**

![real_estate.png](real_estate.png)

# Introduction

In this project, I acted as a Data Analyst for a Real Estate Investment Trust seeking to enter the Residential market. My main objective was to analyze a housing dataset and develop a model capable of predicting house prices based on a variety of features, such as square footage, number of bedrooms, and more.

The project involved several key steps: importing and cleaning the data, performing exploratory data analysis (EDA), developing predictive models using linear regression, and evaluating these models to improve accuracy.

## Background
This project addresses the Trust's need to make data-driven investment decisions in the residential real estate market. By accurately predicting property prices based on historical data, the Trust can better assess market opportunities and risks.

The dataset used covers house sales in King County, including Seattle, between May 2014 and May 2015. It includes important features such as number of bedrooms, living area size, waterfront presence, condition, and renovation status.

By leveraging Python and powerful libraries like Pandas, Seaborn, Scikit-learn, and Matplotlib, I built, evaluated, and refined multiple predictive models, culminating in a Ridge Regression model with a second-order polynomial transformation to maximize prediction accuracy.

### Key questions I addressed:

- What features are most correlated with house price?

- How does the presence of a waterfront affect house prices?

- How accurately can we predict house prices using basic linear models?

- How can model performance be improved using feature engineering and regularization?

## Tools I Used

To complete the Real Estate Price Prediction project, I utilized several essential tools:

- Python: The primary language used for data manipulation, visualization, and machine learning model development.

- Pandas and NumPy: Key libraries for efficient data handling, cleaning, and preprocessing.

- Seaborn and Matplotlib: Visualization libraries used to uncover data trends and relationships.

- Scikit-learn: For building and evaluating machine learning models, including linear regression, ridge regression, and polynomial transformations.

- Visual Studio Code: The code editor I used for development and script execution.

- Git & GitHub: Used for version control, tracking changes, and collaborating on project development.

# 1. Data Cleaning in Python

Here is my notebook: [House_Sales_in_King_Count_USA.ipynb](House_Sales_in_King_Count_USA.ipynb).

#### Actions Taken

- Drop Unnecessary Columns: Removed 'id' and 'Unnamed: 0' columns as they were not useful for the analysis.

- Handling Missing Values: Replaced missing values in 'bedrooms' and 'bathrooms' columns with their respective mean values to maintain data consistency.

- Data Types Inspection: Verified and understood the types of each feature to plan appropriate modeling strategies.

# 2. Exploratory Data Analysis (EDA)

EDA was crucial to understand feature distributions and relationships between variables.

**Key Findings**:
- Distribution of Floors: Analyzed the number of floors across properties using value counts.

- Waterfront vs Price: Visualized how waterfront properties influence prices through a boxplot, revealing higher median prices and more extreme outliers.

- Feature Correlation: Identified that sqft_living, grade, and sqft_above had strong positive correlations with price.

Example of visualization:

```python
sns.boxplot(x="waterfront", y="price", data=df)
```

# 3. Model Development

**Simple Linear Regression**

- Longitude (long) vs Price: Poor R² value (~0.00047), indicating longitude alone is a poor predictor.

- Square Footage (sqft_living) vs Price: Better R² value (~0.49), showing a moderate correlation.

```python
X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
lm.score(X, Y)
```
**Multiple Linear Regression**

Used multiple features:

```python
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]
```
Achieved an R² score of ~0.6577, showing better predictive power.

**Pipeline with Polynomial Features**

Created a pipeline integrating StandardScaler, PolynomialFeatures, and LinearRegression.

```python
pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('polynomial', PolynomialFeatures(degree=2)),
    ('model', LinearRegression())
])
```
Achieved a significantly improved R² value (~0.7512)

# 4. Model Evaluation and Refinement

**Train-Test Split**

- Training Set Size: 18,371 samples

- Testing Set Size: 3,242 samples

```python
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
```
**Ridge Regression**

- Applied Ridge regression with an alpha value of 0.1.

- Achieved an R² of ~0.6478 on the test set.

**Polynomial Ridge Regression**

- After applying a second-order polynomial transform, the Ridge model achieved an even higher R² value (~0.7755), demonstrating the power of feature expansion and regularization combined.

# What I Learned

Throughout this project, I significantly strengthened my data science and machine learning skills:

- 📈 Data Wrangling: Improved my ability to clean and preprocess real-world datasets efficiently.

- 🔍 Exploratory Data Analysis: Mastered using visualizations and statistics to uncover critical insights about feature relationships.

- 🧠 Predictive Modeling: Gained deeper experience building, training, and evaluating linear and regularized regression models.

- ⚡ Pipeline Optimization: Learned how to integrate preprocessing and modeling steps into streamlined pipelines for efficient experimentation.

- 🔨 Model Improvement: Understood how polynomial features and Ridge regression improve predictive performance by capturing non-linear relationships and avoiding overfitting.


# Conclusion

**Insights**

- Waterfront Impact: Houses with waterfront views are significantly more expensive.

- Key Predictors: Living area size, house grade, and location (latitude) are the most influential predictors of price.

- Modeling Techniques: Simple linear models provide a good starting point, but combining multiple features with polynomial transformations and Ridge regularization yields the best performance.

**Business Implications**

These insights enable the Trust to:

- Price Estimation: Predict house prices accurately to make informed investment decisions.

- Feature Prioritization: Focus on properties with waterfront views, larger living spaces, and high grades.

- Investment Strategy: Target renovations that increase square footage and upgrade house grades to maximize property value.