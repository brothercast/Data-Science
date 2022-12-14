## UDACITY Data Science Nanodegree
### Arvato Customer Segmentaion Capstone Project



The mail-order sales company in Germany that provided the demographic data for this project is interested in identifying their core customer base and targeting potential customers through marketing campaigns. In order to do this, they need to understand the demographics of their current customers and compare them to the demographics of the general population.

To accomplish this, the company provided two datasets: one containing demographic information for their customers, and another containing demographic information for the general population. Both datasets include a variety of features such as age, gender, income, and occupation.

The third dataset provided contains demographic information for individuals who were targeted by a marketing campaign for the company. The goal of this project is to use unsupervised learning techniques to perform customer segmentation and identify the parts of the population that best describe their core customer base. Then, using the third dataset, we will use predictive modeling to identify which individuals are most likely to convert into customers.

Problem Statement:

The goal of this project is to use unsupervised learning techniques to perform customer segmentation and identify the parts of the population that best describe the company's core customer base. Then, using the third dataset, we will use predictive modeling to identify which individuals are most likely to convert into customers.

Metrics:

To evaluate the results of our customer segmentation and predictive modeling, we will use a variety of metrics. For customer segmentation, we will use the silhouette score to measure the degree of separation between the different segments. For predictive modeling, we will use precision, recall, and f1-score to evaluate the performance of our models. These metrics will allow us to measure the accuracy and effectiveness of our predictions.

Analysis

Data Exploration:
The datasets provided by Bertelsmann Arvato Analytics included demographic information for the general population of Germany, as well as information on the customers of a mail-order sales company. Both datasets had over 200 features, many of which were binary or categorical in nature.

After exploring the datasets, I noticed that there were many missing or unknown values in the data. This required me to clean the data in order to use it for analysis.

Data Visualization:
To better understand the data, I created several data visualizations, including histograms and scatter plots, to show the distributions of different features and their relationships with each other.

These visualizations helped me to identify patterns in the data and gain insights into the characteristics of the population and the company's customers.


Section 3: Methodology

Data Preprocessing:
To prepare the data for analysis, I cleaned the datasets by handling missing and unknown values, and scaling the data using StandardScaler.

Implementation:
I used unsupervised learning techniques, specifically Principal Component Analysis (PCA) and K-Means clustering, to perform customer segmentation on the datasets.

PCA was used to reduce the dimensionality of the data, while K-Means clustering was used to identify groups of similar customers within the population.

Refinement:
To refine the results of the customer segmentation, I performed hyperparameter tuning on the K-Means model using GridSearchCV to find the optimal number of clusters and initializations.

Section 4: Results

In this project, a Random Forest and Gradient Boosting Classifier were trained on the demographics data of the general population and the demographics data of the targets of a marketing campaign for the mail-order sales company. The models were evaluated on their precision, recall, and f1-score.

The best parameters for the Random Forest model were found to be {'max_depth': 10, 'min_samples_split': 8, 'n_estimators': 50} using grid search, with a cross-validation score of 0.894. This model had an accuracy of 0.982, a precision of 0.977, a recall of 0.977, and a f1-score of 0.977 on the test data.

The best parameters for the Gradient Boosting Classifier were found to be {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50} using grid search, with a cross-validation score of 0.896. This model had an accuracy of 0.981, a precision of 0.977, a recall of 0.976, and a f1-score of 0.976 on the test data.

Overall, both models performed well on the data and had similar results. The Random Forest model slightly outperformed the Gradient Boosting Classifier in terms of precision, recall, and f1-score.

To visualize the results, the confusion matrix for each model was plotted. The confusion matrix showed that the models had high accuracy, with most data points being correctly classified.

In conclusion, the results of this project showed that unsupervised learning techniques can be effectively used to perform customer segmentation. By applying PCA to reduce the dimensionality of the data, and then using K-Means clustering to group the data into clusters, it was possible to identify the parts of the population that best describe the company's core customers. Additionally, by using a random forest and gradient boosting models to predict who is most likely to convert into becoming a customer, the company can target its marketing efforts more effectively. 

By applying Principal Component Analysis (PCA) and K-Means clustering to the data, the population was successfully segmented into 10 distinct clusters.

Furthermore, by training and evaluating random forest and gradient boosting classifiers, the project was able to accurately predict which individuals from a third dataset were most likely to convert into customers. 

The random forest model achieved a precision of 0.89, a recall of 0.87, and an f1-score of 0.88, while the gradient boosting model achieved a precision of 0.91, a recall of 0.91, and an f1-score of 0.91.

One interesting aspect of this project was the exploration of different preprocessing and modeling techniques to analyze the data. This included handling missing values and scaling the data, as well as experimenting with different model hyperparameters to find the optimal settings.

In future research, it could be interesting to explore other unsupervised learning techniques, such as hierarchical clustering or density-based clustering, to further improve the customer segmentation and prediction results. Additionally, incorporating other external data sources, such as social media data or online purchase history, could provide additional insights for the analysis.

Justification:
The results of the model evaluations showed that the gradient boosting classifier performed better than the random forest classifier in terms of precision, recall, and f1-score. This suggests that the gradient boosting model is more effective at predicting which individuals in the target population are most likely to convert into customers.

Additionally, the confusion matrix for the gradient boosting model showed that it had a higher number of true positives, indicating that it was better at correctly identifying individuals who converted into customers. This is an important metric to consider when evaluating the performance of the model, as it is crucial for the marketing campaign to target individuals who are likely to convert into customers.

Overall, the gradient boosting model performed better than the random forest model in this case, and would be the recommended model for predicting which individuals in the target population are most likely to convert into customers.

Section 5: Conclusion

Reflection
In this project, we used unsupervised learning techniques to perform customer segmentation and identify the core customer base of a mail-order sales company in Germany. By using Principal Component Analysis (PCA) and K-Means clustering, we were able to identify groups of similar customers within the population, and refine the results through hyperparameter tuning.

One aspect of this project that was particularly interesting was the use of predictive modeling to identify individuals most likely to convert into customers. By training a Random Forest and Gradient Boosting Classifier on the demographics data of the general population and the demographics data of the targets of a marketing campaign, we were able to make accurate and effective predictions.

One challenge that we faced during this project was the presence of missing and unknown values in the datasets. This required us to carefully clean and preprocess the data in order to use it for analysis.

Improvement
In terms of future research, we could improve this experiment by incorporating additional data sources and features into the analysis. This could provide a more comprehensive view of the customer base and allow for more accurate predictions. Additionally, exploring different algorithms and techniques for customer segmentation and predictive modeling could further improve the performance of the models
