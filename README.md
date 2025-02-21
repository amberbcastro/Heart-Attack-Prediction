![Heart Logo](https://i0.wp.com/post.healthline.com/wp-content/uploads/2020/06/485800-Heart-Disease-Facts-Statistics-and-You-1296x728-Header.png?h=1528)

# Heart-Attack-Prediction
What are the leading risk factors of a heart attack based on the analyzed dataset?

## Heart Attacks: An Overview of the Causes and Consequences
According to the Center for Disease Control and Prevention (CDC), heart disease is the leading cause of death for women and men in the United States. The conditions that fall under heart disease account for the causes of heart attacks -- primarily the clogging of arteries, which carry oxygen-rich blood from the heart to the rest of the body. The chances that someone will develop heart disease or have a heart attack are reliant on a number of factors such as physical inactivity, tobacco use and air pollution, among others. 
### Importance
Every 40 seconds, someone in the United States dies of a heart attack, which amounts to about 805,000 people a year. Because of its great contribution to mortality in the United States, the causes of heart disease -- and more specifically, heart attacks -- are worth studying. Discovering the underlying predictors that put people more at risk of developing heart disease or having a heart attack allows us to minimize their prevalence and thus eliminate their deadliness.

![Heart Arteries](https://www.cdc.gov/heart-disease/media/images/hd-facts-1.jpg)
__Figure 1:__ "As plaque builds up in the arteries of a person with heart disease, the inside of the arteries begins to narrow, which lessens or blocks the flow of blood" (CDC). This restriction of blood flow to the heart is the primary cause of a heart attack. 

*Source: [CDC](https://www.cdc.gov/heart-disease/data-research/facts-stats/index.html#:~:text=Heart%20disease%20is%20the%20leading,people%20died%20from%20heart%20disease.), [WHO](https://www.who.int/health-topics/cardiovascular-diseases#tab=tab_1)*
## Dataset: [Heart Attack Risk Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset)
The analyzed dataset was chosen from kaggle.com due to its extensive list of potential causation variables. Although synthetically-generated by ChatGPT, the dataset mirrors real-world data and was created for others to use in their exploration of various data modeling techniques. The data includes 8763 data points of patient data across 26 variables including demographic and health habit variables.
### Variables
For the purposes of this project, the following variables were analyzed:
| Variable Name   | Description                              | Type     | Default Value |
|-----------------|------------------------------------------|----------|---------------|
| `Diabetes`| Whether the patient has diabetes | `number` | `1: Yes, 0: No` |
| `Family.History`| Family history of heart-related problem | `number` | `1: Yes, 0: No` |
| `Smoking`| Smoking status of the patient | `number` | `1: Yes, 0: No` |
| `Obesity`| Obesity status of the patient | `number` | `1: Yes, 0: No` |
| `Previous.Heart.Problems`| Previous heart problems of the patient | `number` | `1: Yes, 0: No` |
| `Medication.Use`| Medication usage by the patient | `number` | `1: Yes, 0: No` |
| `Heart Attack Risk`| Presence of heart attack risk | `number` | `1: Yes, 0: No` |

## Method 1: Logistic Regression
Logistic regression is a good choice because our target variable, heart attack risk, is binary. It works well with categorical data and provides clear insights into how each factor affects heart attack risk. For example, if smoking has a high positive coefficient, it means smoking increases the risk.

In our dataset, we had to one-hot encode variables like 'Sex', 'Diabetes', 'Alcohol Consumption', and 'Diet' to transform them into a numerical format suitable for logistic regression. This is necessary because logistic regression requires numeric inputs and one-hot encoding allows us to represent categorical variables without imposing any ordinal relationships between them.

Additionally, we split the 'Blood Pressure' variable into two columns: 'Systolic BP' and 'Diastolic BP'. This is because 'Blood Pressure' was originally stored as a string (e.g., "120/80"), and logistic regression requires numerical inputs. By splitting it, we can represent each value as an integer, making it easier to include as a feature in the model.

Logistic regression doesn’t need much computing power, which makes it practical for a dataset of 8,763 patients. To ensure the model's effectiveness, we removed certain columns, such as 'Patient ID', 'Country', 'Continent', and 'Hemisphere'. These columns don’t contribute to predicting heart attack risk and may introduce noise into the model. 'Patient ID' is a unique identifier and irrelevant to risk factors, while 'Country', 'Continent', and 'Hemisphere' may not have a direct relationship with heart attack risk in this case, or their effects may be captured by other variables.

To address potential class imbalance in the dataset, we used SMOTETomek for resampling. This technique generates synthetic data for the minority class and removes overlapping instances, leading to a more balanced dataset. This helps improve the model’s predictive performance, particularly for the underrepresented class, and reduces bias that could result from imbalanced data.

Feature scaling was applied using StandardScaler. Logistic regression can be sensitive to features with different scales, and scaling the features ensures that all predictors contribute equally to the model. It also helps the model converge faster during training, making the process more efficient.

### Code
[Logistic Regression Model Script](code/Log_Regression.py)


### Results
The logistic regression model achieved 67% accuracy. The confusion matrix showed that the model correctly identified 333 true negatives and 312 true positives. It misclassified 155 false positives and 169 false negatives.

The model also helped identify key risk factors by looking at the coefficients. A positive coefficient means the factor increases heart attack risk, and a negative coefficient means it lowers the risk.

![Logistic Regression Coefficients](graphs/logistic_regression_coeffs.png)

Positive risk factors:

*Diet: Surprisingly, a healthy diet was the strongest predictor for heart attack risk in this model.
*Diabetes: Having diabetes increased the risk, which is consistent with medical research.
*Sex_Male: Being male also increased the risk, which aligns with known trends.
*Alcohol Consumption: Alcohol use was positively linked to heart attack risk.

Negative risk factors:

*Obesity: Surprisingly, obesity was linked to a lower risk of heart attack in this model.
*Family History: A family history of heart disease was also linked to a lower risk, which contradicts expectations.
*Previous Heart Problems: Having previous heart problems was associated with a lower risk of heart attack, which is unexpected.
*Medication Use: Medication use was also negatively correlated with heart attack risk, which is unusual.

The model’s performance was also evaluated using precision, recall, and F1-score. The precision was 0.66 for both classes, meaning the model was correct 66% of the time when predicting a heart attack risk. The recall was 0.67 for both classes, meaning the model identified 67% of the actual cases. The F1-score was also 0.66, showing a balanced trade-off between false positives and false negatives.

The several counterintuitive findings suggest potential issues with confounding variables or selection bias.

## Method 2: Feedforward Neural Network
Feedforward neural networks are used in the medical field for cardiovascular diseases, cancer detection, and image analysis. The primary goal of this project is to determine what factors are most likely to cause a heart attack or myocardial infarction. To start, any variables not in binary form were removed for this part of the analysis. The activation functions are sigmoid functions, so the data had to be binary. For the analysis to work correctly, many variables were not considered. 

```
#Load in dataset
Prediction<-read.csv("/Users/colint./Desktop/Working Directory/Prediction.csv")

#Remove columns
Prediction$Patient.ID<-NULL
Prediction$Age<-NULL
Prediction$Sex<-NULL
Prediction$Cholesterol<-NULL
Prediction$Blood.Pressure<-NULL
Prediction$Heart.Rate<-NULL
Prediction$Exercise.Hours.Per.Week<-NULL
Prediction$Diet<-NULL
Prediction$Stress.Level<-NULL
Prediction$Sedentary.Hours.Per.Day<-NULL
Prediction$Income<-NULL
Prediction$BMI<-NULL
Prediction$Triglycerides<-NULL
Prediction$Physical.Activity.Days.Per.Week<-NULL
Prediction$Sleep.Hours.Per.Day<-NULL
Prediction$Country<-NULL
Prediction$Continent<-NULL
Prediction$Hemisphere<-NULL
Prediction$Alcohol.Consumption<-NULL

#Install Packages
install.packages("keras")
install.packages("tensorflow")

#Load packages
library(keras)
library(tensorflow)
library(dplyr)
library(caret)

#Split dataset into train and test
trainIndex<-createDataPartition(Prediction$Heart.Attack.Risk,p=0.8,list=F)

train_data<-Prediction[trainIndex,]
test_data<-Prediction[-trainIndex,]

x_train<-as.matrix(train_data[, -which(names(train_data) == "Heart.Attack.Risk")])
y_train<-as.matrix(train_data$Heart.Attack.Risk)

x_test<-as.matrix(test_data[, -which(names(test_data) == "Heart.Attack.Risk")])
y_test<-as.matrix(test_data$Heart.Attack.Risk)

#Create and plot model
library(neuralnet)
model = neuralnet(
  Heart.Attack.Risk~.,
  data=train_data,
  hidden=c(2),
  linear.output=F
)
plot(model,rep="best")

#Check model accuracy
pred <- model %>% predict(x_test)
predicted_classes <- ifelse(pred > 0.50, 1, 0)
predicted_classes<-as.factor(predicted_classes)
confMatrix<-confusionMatrix(predicted_classes,y_test_factor,positive = "1")
confMatrix



```

Code, graphs, explanations here
### Results
## Cross Validation 
### Method 1: Logistic Regression
### Method 2: Feedforward Neural Network
### Results: Comparing Methods
## Conclusions
## Suggestions for Future Analysis
