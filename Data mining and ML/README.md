# Data Science: A solution for predicting diabetes?
## Introduction
Diabetes is a well-known chronic disease that causes the impossibility for the body to self-regulate its blood glucose levels. It affects the overall well-being of the individual. However, if the risk of obtaining the disease is known in advance, it can be prevented. That’s why our goal here is to predict thanks to some features if the patient will develop diabetes over the next five years or not.

## Data analysis
For this aim, we are provided a dataset of 768 individuals, with information on pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, the diabetes pedigree function, and age. The outcome is also supplied for each individual, corresponding to whether they have diabetes over the next five years.

Looking at the distribution of the outcome, it can be noted that the dataset provided includes almost two-thirds of people who did not suffer from diabetes compared to one-third of people who had diabetes. It is therefore a quite unbalanced dataset, which should be considered in the following.

For the different variables as a function of the prediction variable, we can see a big difference between the distribution of people with and without diabetes for glucose. We can therefore assume that this variable will be of great importance for the prediction of the outcome. Moreover, there is also a slight difference in age and diabetes pedigree function.


On the other hand, we also notice many zeros for some variables: insulin, skin thickness, blood pressure. As this value is normally not a possible value, we can deduce that in this data set the missing values have been replaced by zeros.
Thanks to a heatmap, we find out that as supposed before, we have a great correlation between glucose and diabetes, but also BMI, age, and pregnancies. Moreover, it shows that some variables appear to be correlated together like age and pregnancies, but also insulin and skin thickness.
   
## Prediction of diabetes
### Preprocessing
As mentioned earlier, zero values are missing values. So, we need to find out how to deal with it. It can be noted that Skin Thickness and Insulin have many missing values whereas glucose, blood pressure, and BMI have few ones. Therefore, we are not going to manage these in the same way. Indeed, for glucose, blood pressure, and BMI, we are going to take the mean. However, when there is a too large proportion of missing values, using the mean is meaningless. That's why we are going to predict the value thanks to the others for skin thickness and insulin.

We can now split our dataset into the train and the test set. Since our database is unbalanced as seen before, we will perform an oversampling on the train set to obtain two outcome categories with similar proportions.

Once we have dealt with missing values and unbalancing outcomes, we need to scale our data. For that, several scalers are well-known as Standard Scaler, Robust Scaler, or Min-Max Scaler. We put them in competition to see which one works best with our data. Results show that the standard scaler is the one that seems to work best with our dataset, so we are going to use this one for the continuation of the project.

### Models training
Now that we have finished preprocessing the data, we can start training our models. For that, we are going to take the models we are used to use, that is to say: Logistic Regression, SVM, KNN, Decision Tree, and Random Forest. The first step is the tuning of the hyperparameters of the models. Many combinations are possible, and the goal is to find the one that achieves the best results. Once the best hyper-parameters have been found, we can train the models to obtain results for different metrics.

## Results
No model appears to be completely outscoring the others in terms of all the metrics represented. However, Random Forest seems to have the best and above all more stable results than all the other models.

With cross-validation, we obtain the following results for Random Forest, which are quite good:
Accuracy: 0.816; F1 weighted: 0.82; Precision: 0.801; AUC score: 0.905; Recall: 0.869.

A good way to understand the results of our algorithm is to look at the confusion matrix. This allows us to understand which category is best predicted and vice versa. The matrix is based on the test set, the part of the data with which the algorithm has not yet been trained. We can see that it is very good at predicting people who will not get diabetes. However, it is more complicated for him to know about those who will have it. This can surely be explained by the fact that the starting dataset was unbalanced and thus, even if we wanted to do oversampling, this still influences the learning of the model.

In some projects, it is not only the score of our model that is important but also the explicability of the model. Especially in medical projects like this one, where the doctor will wait to understand how the algorithm arrives at such an answer. To find the explication of our random forest model, we will use the shap library which allows us to obtain this graph. For instance, this means that the more glucose you have, the greater the chance you have of getting diabetes.

## Conclusion
To conclude, data science can be a great tool to predict diabetes. However, the results could be improved by increasing the dataset first, especially on the data of people with diabetes. Moreover, more features can be useful such as sex, physical training. More feature engineering could be implemented (dummies implemented but not conclusive), or we can review the way missing values are handled to have better results.
