import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.pipeline import Pipeline

# GitHub文件的原始链接
file_url = 'https://raw.githubusercontent.com/Yorick0325/student-feedback-analysis/main/Student%20Feedback%20Data/Rating%20data/quantitative_data.xlsx'

# 使用 requests 下载文件
response = requests.get(file_url)

# 确保响应成功
if response.status_code == 200:
    # 将下载的内容转换为BytesIO对象，供pandas读取
    file_content = BytesIO(response.content)

    # 使用pandas读取Excel文件
    data = pd.read_excel(file_content)
else:
    print(f"Failed to download file: {response.status_code}")

# Data Cleaning: Handle missing values (impute with mean for simplicity)
data = data.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric, coerce errors to NaN
data = data.fillna(data.mean(numeric_only=True))

# Extract only numeric columns for analysis
numeric_data = data.select_dtypes(include=[float, int])

# Step 1: Determine optimal n_components for PCA using GridSearchCV
pca = PCA()
model = GradientBoostingRegressor()

pipe = Pipeline(steps=[('pca', pca), ('model', model)])

param_grid = {
    'pca__n_components': np.arange(1, min(numeric_data.shape[1], 20), 1)
}

search = GridSearchCV(pipe, param_grid, cv=5)
search.fit(numeric_data, numeric_data.mean(axis=1))
best_n_components = search.best_params_['pca__n_components']

# Output the best n_components
print(f"Optimal number of PCA components: {best_n_components}")

# Plot the GridSearchCV results to visualize the optimal number of components
cv_results = pd.DataFrame(search.cv_results_)
plt.figure(figsize=(10, 6))
plt.plot(cv_results['param_pca__n_components'], -cv_results['mean_test_score'], marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Negative MSE (Cross-Validation)')
plt.title('Grid Search for Optimal Number of PCA Components')
plt.axvline(x=best_n_components, color='r', linestyle='--', label=f'Optimal n_components = {best_n_components}')
plt.legend()
plt.show()

# Step 2: Apply PCA with optimal n_components
pca = PCA(n_components=best_n_components)
pca_result = pca.fit_transform(numeric_data)

# Output the shape of the PCA result to see the number of dimensions after PCA
print(f"Shape of PCA result: {pca_result.shape}")

# Visualize the composition of each principal component
components_df = pd.DataFrame(pca.components_, columns=numeric_data.columns,
                             index=[f'PC{i + 1}' for i in range(best_n_components)])
print("PCA Components:")
print(components_df)

# Plot the first few principal components' compositions
plt.figure(figsize=(12, 8))
for i in range(min(5, len(pca.components_))):  # Only plot the first 5 components for clarity
    plt.bar(components_df.columns, components_df.iloc[i], alpha=0.7, label=f'PC{i + 1}')
plt.xlabel('Original Features')
plt.ylabel('Component Weights')
plt.title('Composition of Principal Components')
plt.legend()
plt.xticks(rotation=90)
plt.show()

# Step 3: Determine optimal number of clusters for K-means using Elbow Method
sse = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(pca_result)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Step 4: Calculate silhouette scores for different k values (Optional)
silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(pca_result)
    score = silhouette_score(pca_result, kmeans.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.show()

# Apply K-means with the optimal number of clusters (based on the elbow method or silhouette score)
optimal_k = np.argmax(silhouette_scores) + 2  # Adding 2 because range starts from 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(pca_result)

# Add PCA components and cluster labels back to the dataframe
numeric_data['PCA1'] = pca_result[:, 0]
numeric_data['PCA2'] = pca_result[:, 1]
numeric_data['Cluster'] = clusters

# Step 5: Analyze and visualize cluster characteristics
# Calculate the mean of each feature for each cluster
cluster_means = numeric_data.groupby('Cluster').mean()

# Output the means to understand the characteristics of each cluster
print("\nCluster Means:\n", cluster_means)

# Example: Boxplot to show the distribution of a specific feature across clusters
plt.figure(figsize=(12, 8))
sns.boxplot(x='Cluster', y='4. I woke up feeling fresh and rested', data=numeric_data)
plt.title('Distribution of "Feeling Fresh and Rested" Across Clusters')
plt.show()

# Heatmap to visualize the mean values of each feature across clusters
plt.figure(figsize=(10, 8))
sns.heatmap(cluster_means, annot=True, cmap='coolwarm')
plt.title('Heatmap of Cluster Features')
plt.show()

# Step 6: Assume we need to predict 'Overall_Satisfaction' and we're adding the cluster labels and PCA components
numeric_data['Overall_Satisfaction'] = numeric_data.mean(axis=1)  # Replace with actual target if available

# Prepare data for Gradient Boosting Regressor
X = numeric_data.drop('Overall_Satisfaction', axis=1)
y = numeric_data['Overall_Satisfaction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting Regressor
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Output the mean squared error
mse = mean_squared_error(y_test, predictions)
mse_output = f'Mean Squared Error: {mse}'
print(mse_output)

# Output feature importances
importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Output the feature importances DataFrame
importance_df_output = importance_df.to_string(index=False)
print("\nFeature Importances:\n", importance_df_output)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances in Predicting Overall Satisfaction')
plt.show()

# Displaying predictions for the first few instances
predictions_output = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions
}).head()
print(predictions_output)
