import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

"""Data Collection & Pre-Processing"""

# Loading the data from csv file to a pandas DataFrame
raw_mail_data = pd.read_csv('spam_ham_dataset_.csv')
#print(raw_mail_data)

# Replace the null values with a null string
mail_data = raw_mail_data.fillna('')

# Printing the first 5 rows of the DataFrame
print(mail_data.head(10))

# Checking the number of rows and columns in the DataFrame
print(mail_data.shape)



"""Label Encoding"""
# Label spam mail as 0, ham mail as 1
mail_data['Category'] = mail_data['Category'].map({'spam': 0, 'ham': 1})

# Separating the data as texts and labels
X = mail_data['Message']
Y = mail_data['Category']

print(X)
print(Y)



"""Splitting the data into training data & test data"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print("Shape of Independet Data set",X.shape)
print("\nInput data Training Set Shape...:",X_train.shape)
print("\nInput data Testing Set Shape...:",X_test.shape)
print("\nShape of Dependet Data set",Y.shape)
print("\nOutput data Training Set Shape...:",Y_train.shape)
print("\nOutput data Testing Set Shape...:",Y_test.shape)

"""Feature Extraction"""

# Transform the text data to feature vectors that can be used as input to Logistic regression

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


# Convert Y_train and Y_test to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print(X_train)
#print(X_train_features)


"""Training the Model: Logistic Regression"""

model = LogisticRegression()
print("\nModel is Developed.....!")

# Training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)

print("\nModel is trained Successfully...!")
"""Evaluating the trained model"""

# Prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print('\nAccuracy on training data:', accuracy_on_training_data)

# Prediction on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print('Accuracy on test data:', accuracy_on_test_data)


#Testing the Model with Input Data

input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]
"""Application Update 139 Spam"""
input_mail =["Subject: password reset"] #Row Num 104

#input_mail=["Application Update"]
# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)
# making prediction

prediction = model.predict(input_data_features)
print(prediction)

print("Input Mail :- ",input_mail)

print("\nResult :- ")
if (prediction[0]==1):
  print('Your Mail is Ham mail')

else:
  print('Your Mail is Spam mail')