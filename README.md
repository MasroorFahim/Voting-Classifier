# Voting-Classifier
So here, this is a simple demonstration of how a stacking classifiers work on a dataset. Here in my experiment , first of all, the name of the dataset is Drebin Dataset and with this dataset initially it had like 213 features and 15037 rows with a combination of malware and benign which is denoted by 1 and 0 respectively in the class column. The procedure of my experiment is as follows:

Dataset Collection
Dataset Analysis and fixing
Dataset Variance Threshold Detection
Pearson Correlation
Plotting with various methods  
Dataset Standardization (fit and transform)
Principal Component Analysis
Variance Inflation Factor
Feature Selection (Mutual Information)
For imbalance dataset using SMOTETomek with randomsampler
StratifiedShuffleSplit
Individual Evaluation of ML algorithms.
Voting it up for further evaluation.
The brief discussion of this will be:

So here, we first collect the dataset and verify if it sets up nicely with respect to the experiment.

Now comes the dataset analysis part. Here first of all the dataset is checked if theres any missing values in there. And if there is any missing valuem, it has to be fixed to analysis the dataset well.

Now the third part which is The VarianceThreshold is a feature selection method in machine learning, typically used for removing low-variance features from a dataset. Features with low variance generally contain little information, and removing them can be beneficial, especially in cases where there is redundancy or noise in the data.

Here is the full definition of VarianceThreshold:

Variance Threshold in scikit-learn: In scikit-learn, Variance Threshold is a feature selection method provided in the feature selection module. It operates on numerical features and removes those with variance below a certain threshold.

Key Components: Variance:

Variance is a measure of the spread or dispersion of a set of values. In the context of feature selection, it refers to the amount of variability or change in the values of a feature across the samples in the dataset. Threshold:

Variance Threshold takes a threshold parameter, and it removes features with variance below this threshold. Features with variance less than the specified threshold are considered low-variance and are removed.

The fourth part is Pearson correlation, often referred to as Pearson's correlation coefficient or simply Pearson's r, is a statistical measure that quantifies the strength and direction of a linear relationship between two continuous variables. It is widely used in statistics to assess how well the relationship between two variables can be described by a straight line.
Here, r=1 indicates a perfect positive linear relationship. r=−1 indicates a perfect negative linear relationship. r=0 indicates no linear relationship.

In our dataset, the columns or features which are 75 % correlated are removed by this special formulae.

After that, some methods of plotting were conducted between the features in order to know the insights and demonstrate the various way of visualizing it. The plotting that was used are:
i. Scatter plotting: 
A scatter plot is a type of data visualization that displays individual data points on a two-dimensional plane. Each data point is represented by a marker (such as a dot or a symbol) at the intersection of its x and y coordinates. Scatter plots are commonly used to examine the relationship between two continuous variables and identify patterns, trends, or clusters in the data.
ii. Line plotting:
A line plot, also known as a line chart or line graph, is a type of data visualization that displays data points connected by straight line segments. It is particularly useful for showing trends and patterns in data over a continuous interval or time period. Line plots are commonly used in various fields such as statistics, finance, economics, and science to illustrate the change in values of a variable.
iii. Bar plot:
A bar plot, also known as a bar chart or bar graph, is a data visualization technique that represents categorical data with rectangular bars. The length or height of each bar is proportional to the quantity it represents. Bar plots are effective for comparing the values of different categories or groups and are commonly used to visualize discrete data.

iv. Box plot:
A box plot, also known as a box-and-whisker plot, is a statistical data visualization method that provides a summary of the distribution of a dataset. It displays key statistical measures such as the median, quartiles, and potential outliers in a concise and easy-to-read format. Box plots are especially useful for comparing distributions across different groups or datasets.

v. Violin plot:
A violin plot is a data visualization method that combines aspects of a box plot and a kernel density plot to provide insights into the distribution of a dataset. It is particularly useful for comparing the distribution of a variable across different categories or groups. The violin plot displays the distribution of data by representing the probability density of different values.

vi. Count Plot:
A count plot is a type of data visualization that displays the number of occurrences of each unique category or value in a categorical variable. It is essentially a bar plot where the height of each bar represents the count of observations in each category. Count plots are particularly useful for visualizing the distribution of categorical data and identifying the frequency of each category.

vii. Histogram plot:
A histogram is a graphical representation of the distribution of a dataset. It provides a visual summary of the underlying frequency distribution of a continuous variable. In a histogram, the data is divided into intervals, and the frequency (or count) of observations falling into each interval is represented by the height of bars. The key purpose of a histogram is to depict the shape, center, and spread of the data distribution.
vii. Heatmap plot:

A heat map is a data visualization technique that represents the values of a matrix or a table of data as colours. It is particularly useful for visualizing the magnitude of relationships or patterns between two sets of variables. Heat maps are commonly employed in various fields, including statistics, biology, finance, and machine learning, to reveal underlying structures or trends in complex datasets.

After that, in the sixth step, the dataset is standardized and transformed.
The mechanism of this is to make the scale of each feature to have a mean of 0 and a standard deviation of 1. Other transformations or scalers in scikit-learn can be used depending on the specific needs of your data. The code used here is, 

By scaler = StandardScaler() scaler.fit(X) standardized_data = scaler.transform(X)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

The 7th step is Principal Component Analysis
Principal Component Analysis (PCA) is a dimensionality reduction technique widely used in machine learning and statistics. Its primary goal is to transform a high-dimensional dataset into a lower-dimensional representation while retaining as much of the original variance as possible. PCA achieves this by identifying the principal components, which are linear combinations of the original features, and ordering them by the amount of variance they capture. Generally, Each principal component represents a direction in feature space. The first principal component points in the direction of the maximum variance, the second principal component points in the direction of the second maximum variance, and so forth.PCA is often used for data exploration, noise reduction, and visualization. It can be applied to various domains, including image processing, genetics, and finance.

Key Concepts of PCA:

Variance Maximization:

PCA seeks to find the linear combinations (principal components) of the original features that maximize the variance in the data. The first principal component captures the most variance, the second principal component (orthogonal to the first) captures the second most variance, and so on. Orthogonality:

Principal components are orthogonal to each other, meaning they are uncorrelated. This ensures that each component contributes uniquely to the overall variance. Eigenvalue Decomposition:

PCA involves the eigenvalue decomposition of the covariance matrix of the original features. The eigenvectors represent the principal components, and the eigenvalues indicate the amount of variance captured by each component. Dimensionality Reduction:

By selecting a subset of the principal components that capture a significant amount of variance, one can create a lower-dimensional representation of the data. This reduction in dimensionality can lead to simplified models, faster training times, and improved interpretability. Principal Component Scores:

The transformed data, called the principal component scores, is obtained by projecting the original data onto the subspace spanned by the selected principal components.

After PCA, we did The Variance Inflation Factor (VIF) which is a statistical measure used to assess the severity of multicollinearity in a regression analysis. Multicollinearity occurs when two or more independent variables in a regression model are highly correlated, making it challenging to isolate the individual effect of each variable on the dependent variable.
In our experiment, we took every features which had a vif factor of less than 7.

The later step is Mutual Information feature selection method.
Mutual Information (MI) is a statistical metric used in feature selection to quantify the relationship between two variables by measuring the amount of information obtained about one variable through the observation of the other. In the context of feature selection, mutual information is often employed to evaluate the relevance of individual features with respect to the target variable.

Generally, Feature Selection Using Mutual Information: In the context of machine learning, mutual information can be employed for feature selection by ranking features based on their individual mutual information scores with the target variable. Features with higher scores are considered more informative and are selected for inclusion in the model.

In our experiment, we took the best 70 features using mutual informationn and with the help of SelectKBest, which is a feature selection technique in scikit-learn that is used to select the top k features based on a specified scoring function. This method is part of the feature selection module (sklearn.feature_selection) in scikit-learn and is commonly employed to improve the performance of machine learning models by focusing on the most relevant features.

Using SMOTETOMEK and randomsampler to address the dataset's imbalance is the tenth step.
Dealing with imbalanced datasets is a common challenge in machine learning. Two popular techniques for handling imbalanced datasets are using random sampling (RandomSampler) and SMOTE (Synthetic Minority Over-sampling Technique) combined with Tomek links (SMOTETomek).

RandomSampler: The RandomSampler is a simple approach to balance a dataset by randomly under-sampling the majority class. This technique involves removing some instances from the majority class to make the class distribution more balanced.

SMOTETomek: SMOTETomek is a combination of over-sampling the minority class using SMOTE and under-sampling the majority class using Tomek links. The goal is to generate synthetic samples for the minority class and remove potentially noisy instances from both classes.

Both RandomSampler and SMOTETomek are available in the imbalanced-learn library (imblearn). Before using these techniques, it's important to assess the specific characteristics of your dataset and choose the method that suits your problem. Keep in mind that no single technique is universally best for all imbalanced datasets, and experimentation is often necessary. Additionally, the performance of these methods may depend on the algorithm used for classification.

The dataset is then again StratifiedShuffle and spliited in to the test size of 20 percent and train size of 80 percent. And we used this expression, sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

Then the individual evaluation of ml algorithm starts. Here, we evaluated the algorithms are accuracy, precision, specificity ,f1 score, recall and plotted confusion matrix accordingly to access every matrix for this dataset.

In here, 13 different classifiers were taken and evaluated accordingly and they are hypertuned first to get an accurate view.

i. Decision Tree classifier
ii. Logistic regression classifier
iii. Gaussian Naive-Bayes Classifier
iv. Random Forest Classifier
v. KNN classifier
vi. SVM classifier
vii. SGD Classifier
viii. XG Boost Classifier
ix. Passive Aggressive Classifier
x. Extra Trees Classifier
xi. Perceptron Classifier
xii. LGBM Classifier
xiii. Ridge Classifier
A short Description of each of them are given below:

i. Decision Tree Classifier: A Decision Tree is a tree-shaped model of decisions where each node represents a decision based on a particular feature. It recursively splits the dataset into subsets based on the most significant feature, creating a tree-like structure to make predictions.

ii. Logistic Regression Classifier: Despite its name, logistic regression is a linear model for binary classification. It uses the logistic function to model the probability that a given instance belongs to a particular class. It's widely used and interpretable.

iii. Naive-Bayes Classifier: Naive-Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes that features are conditionally independent, given the class label. It's particularly efficient for text classification and simple problems

iv. Random Forest Classifier: Random Forest is an ensemble learning method that constructs a multitude of decision trees during training. It combines their predictions to improve accuracy and reduce overfitting.

v. K-Nearest Neighbours (KNN) Classifier: KNN is a non-parametric and instance-based algorithm for classification. It classifies new instances based on the majority class of their k nearest neighbors in the feature space. The choice of 'k' determines the number of neighbors to consider.

vi. Support Vector Machine (SVM) Classifier: SVM is a supervised machine learning algorithm used for classification and regression tasks. It finds the optimal hyperplane that separates classes in a high-dimensional space, maximizing the margin between classes.

vii. SGD Classifier (Stochastic Gradient Descent Classifier): SGD Classifier is a linear classifier that uses stochastic gradient descent as an optimization algorithm. It's particularly useful for large datasets and online learning scenarios.

viii. XG Boost Classifier: XG Boost is an efficient and scalable implementation of gradient boosting. It is an ensemble learning algorithm that builds a series of weak learners (usually decision trees) and combines them to create a strong learner.

ix. Passive Aggressive Classifier: The Passive-Aggressive algorithm is an online learning algorithm for classification. It is suitable for situations where the data is not static, and the model needs to adapt to changes over time.

x. Extra Trees Classifier: Extra Trees (Extremely Randomized Trees) is an ensemble learning method that builds multiple decision trees and selects the splits for nodes randomly. This randomness can often lead to improved performance.

xi. Perceptron Classifier: A perceptron is a simple neural network model that learns a binary linear classifier. It's a single-layer neural network with a threshold activation function.

xii. LGBM Classifier (Light GBM Classifier): Light GBM is a gradient boosting framework that uses tree-based learning algorithms. It's designed for distributed and efficient training with a focus on handling large datasets.

xiii. Ridge Classifier: Ridge Classifier is a linear classifier that uses Ridge Regression, a regularized linear regression model. It adds a penalty term to the least squares objective, promoting models with lower complexity. Each of these classifiers has its strengths and weaknesses, and the choice depends on the characteristics of the data and the specific requirements of the problem at hand.

The performance evaluation metrics that used in this experiment are:
1. Accuracy:
Definition: Accuracy is a measure of the overall correctness of the model and is calculated as the ratio of correctly predicted instances to the total instances.
Formula: Accuracy = (True Positives + True Negatives) / (True Positives + False Positives + True Negatives + False Negatives)
Interpretation: Accuracy provides a general assessment of how well the model is performing across all classes.
2. Precision:
Definition: Precision is a measure of the accuracy of positive predictions and is calculated as the ratio of true positive predictions to the total positive predictions (including both true positives and false positives).
Formula: Precision = True Positives / (True Positives + False Positives)
Interpretation: Precision focuses on the reliability of positive predictions, helping to assess the model's ability to avoid false positives.
3. Recall (Sensitivity or True Positive Rate):
Definition: Recall, also known as sensitivity or true positive rate, measures the ability of the model to capture all positive instances and is calculated as the ratio of true positive predictions to the total actual positive instances (including both true positives and false negatives).
Formula: Recall = True Positives / (True Positives + False Negatives)
Interpretation: Recall is particularly important in situations where the identification of positive instances is crucial, and false negatives should be minimized.
4. F1 Score:
Definition: The F1 score is a metric that balances precision and recall, providing a single score that considers both false positives and false negatives. It is the harmonic mean of precision and recall.
Formula: F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
Interpretation: The F1 score is useful when there is an uneven class distribution or when false positives and false negatives have different consequences.
5. Specificity:
Definition: Specificity, also known as the true negative rate, measures the ability of the model to correctly identify negative instances. It is calculated as the ratio of true negative predictions to the total actual negative instances (including both true negatives and false positives).
Formula: Specificity = True Negatives / (True Negatives + False Positives)
Interpretation: Specificity is crucial when the cost of false positives (Type I errors) needs to be minimized. It complements sensitivity, providing a comprehensive assessment of the model's performance across both positive and negative instances.
6. Receiver Operating Characteristic (ROC) Curve:
Definition: The ROC curve is a graphical representation of the trade-off between true positive rate (sensitivity) and false positive rate (1 - specificity) across different threshold values for a binary classification model. It illustrates the model's ability to distinguish between the positive and negative classes under varying conditions.
Components:
The x-axis represents the false positive rate (1 - specificity).
The y-axis represents the true positive rate (sensitivity or recall).
Each point on the ROC curve corresponds to a different threshold for classifying positive instances.
Interpretation: A curve that hugs the top-left corner indicates a better model performance.
The area under the ROC curve (AUC-ROC) provides a single metric summarizing the model's discriminatory power. An AUC-ROC value of 1 indicates perfect performance, while 0.5 suggests random guessing.

7. Precision-Recall (PR) Curve:
Definition: The Precision-Recall curve is a graphical representation of the trade-off between precision and recall across different threshold values for a binary classification model. It highlights the model's ability to provide accurate and relevant positive predictions while minimizing false positives.
Components:
The x-axis represents recall (sensitivity).
The y-axis represents precision.
Each point on the PR curve corresponds to a different threshold for classifying positive instances.
Interpretation: A curve that hugs the top-right corner indicates better model performance.Precision and recall are particularly relevant when dealing with imbalanced datasets, where one class is significantly more prevalent than the other. The area under the PR curve (AUC-PR) provides a summary metric, and a higher AUC-PR suggests better model performance.
So, These performance matrices were the prime way of evaluating the dataset and along with that for analysing the regression analysis, just to have some knowledge the above 3 are also included and demonstrated here. 

8. Mean Absolute Error (MAE):
Definition: Mean Absolute Error is a regression metric that measures the average absolute difference between the predicted and actual values. It gives an idea of the model's accuracy in predicting the absolute magnitude of errors.
Interpretation: MAE is easy to interpret, as it represents the average magnitude of errors. It is less sensitive to outliers compared to mean squared error.

9. Mean Squared Error (MSE):
Definition: Mean Squared Error is a regression metric that measures the average squared difference between the predicted and actual values. It penalizes larger errors more significantly than smaller errors.
Interpretation: MSE gives more weight to larger errors, making it sensitive to outliers. It provides insight into the overall variability of the errors.

10. Mean Squared Logarithmic Error (MSLE):
Definition: Mean Squared Logarithmic Error is a regression metric that measures the average squared logarithmic difference between the predicted and actual values. It is particularly useful when the predicted values cover a wide range of magnitudes.
Interpretation: MSLE is suitable for datasets with a wide range of target variable values, and it penalizes underestimation and overestimation symmetrically on a logarithmic scale.


The last step is to use voting classifier by taking or blending the used classifiers used earlier to see the results together how well they can work in soft and hard voting classifier.In the last step, 3 base models were formed with 3, 6 and 9 classifiers, respectively. And then 6 models were formed from by taking soft and hard voting classifier each time for each base models.

A general overview of voting classifier will be:
Voting Classifier:
A voting classifier is an ensemble learning method that combines the predictions of multiple base classifiers to make a final prediction. It operates by allowing each base classifier to vote on the predicted class, and the class that receives the majority of votes is selected as the final prediction.

Here is a brief definition and procedure for a voting classifier:

Definition:
A voting classifier is an ensemble model that combines the decisions of multiple individual classifiers to improve overall predictive performance and generalization.

Procedure:

Selection of Base Classifiers:

Choose a set of diverse base classifiers. These could be different algorithms or instances of the same algorithm trained on different subsets of the data.
Training Base Classifiers:
Train each base classifier on the training data independently.

Voting Mechanism:
Define a voting mechanism for combining the predictions of the base classifiers. Common voting mechanisms include "hard voting" and "soft voting":

Hard Voting 
A Hard Voting Classifier, a type of voting classifier, is an ensemble learning method that combines the predictions of multiple individual classifiers through a simple majority vote. In the context of classification tasks, each base classifier independently predicts a class for a given input, and the class that receives the most votes is chosen as the final prediction.
Here is a brief definition and explanation of the procedure for a Hard Voting Classifier:
Definition:
A Hard Voting Classifier is an ensemble model that combines the predictions of multiple base classifiers by taking a majority vote. The class with the most individual votes across all classifiers is selected as the final predicted class.
Procedure: 
Selection of Base Classifiers: Choose a set of diverse base classifiers. These can be different algorithms or instances of the same algorithm trained on different subsets of the data.
Training Base Classifiers: Train each base classifier on the training data independently.
Voting Mechanism (Hard Voting): Define a hard voting mechanism, where the final prediction is based on the class with the majority of votes.
Combining Predictions: Allow each base classifier to make predictions on new, unseen data.
Aggregating Votes: For each instance, count the votes for each class across all base classifiers.
Final Prediction: Determine the final prediction by selecting the class with the highest count of votes.
The hard voting strategy is effective when individual classifiers are diverse and may make different errors on different instances. By combining their decisions through a majority vote, a hard voting classifier can often achieve better performance than individual classifiers.

Soft Voting: 
A Soft Voting Classifier is an ensemble learning method that combines the predictions of multiple individual classifiers through a weighted average of their predicted probabilities. In soft voting, each base classifier provides a probability estimate for each class, and the final prediction is based on the class with the highest average probability.
Here is a brief definition and explanation of the procedure for a Soft Voting Classifier:
Definition: A Soft Voting Classifier is an ensemble model that combines the predicted probabilities of multiple base classifiers. The final prediction is based on the class with the highest average probability across all classifiers.
Procedure:
Selection of Base Classifiers: Choose a set of diverse base classifiers. These can be different algorithms or instances of the same algorithm trained on different subsets of the data.
Training Base Classifiers: Train each base classifier on the training data independently.
Voting Mechanism (Soft Voting): Define a soft voting mechanism, where the final prediction is based on the class with the highest average predicted probability.
Combining Predictions: Allow each base classifier to make probability predictions on new, unseen data.
Aggregating Probabilities: For each instance, calculate the average predicted probability for each class across all base classifiers.
Final Prediction: Determine the final prediction by selecting the class with the highest average probability.
The soft voting strategy is effective when individual classifiers provide probability estimates for each class. By combining these probabilities through a weighted average, a soft voting classifier can take into account the confidence or certainty of each classifier's prediction.

The key idea behind a voting classifier is that by combining the strengths of multiple classifiers, it can often achieve better performance and generalization than individual classifiers. It is especially useful when the base classifiers are diverse and make errors on different subsets of the data. Common algorithms used as base classifiers include decision trees, support vector machines, logistic regression, k-nearest neighbors, etc.
In Python, the VotingClassifier class in the scikit-learn library provides an implementation of a voting classifier. It allows you to combine multiple classifiers and specify the voting strategy.
So, this is a fun project and i hope alot of people will get benefit from this.

THANK YOU.
