Spam/Ham Email Classifier

This project implements a machine learning-based Spam/Ham Email Classification system using Python. It leverages the scikit-learn library for data preprocessing, feature extraction, and model training. Below is a detailed overview of the project:

Features of the Project

	1.	Data Collection and Preprocessing
	•	Loads a dataset containing labeled email messages.
	•	Handles missing data by replacing null values with empty strings.
	•	Encodes the labels (spam as 0 and ham as 1) for compatibility with the machine learning model.
	2.	Data Splitting
	•	Splits the dataset into training (80%) and testing (20%) sets for model evaluation.
	3.	Feature Extraction
	•	Utilizes the TfidfVectorizer to transform text data into numerical feature vectors.
	•	Removes common English stopwords and normalizes the text to lowercase.
	4.	Model Training
	•	Trains a Logistic Regression model on the extracted features using the training data.
	5.	Model Evaluation
	•	Evaluates the model’s performance on training and testing data using accuracy metrics.
	6.	Spam/Ham Prediction
	•	Allows users to input new email messages to predict whether they are spam or ham.

How to Run

	1.	Install necessary Python libraries:

pip install numpy pandas scikit-learn


	2.	Place the dataset spam_ham_dataset_.csv in the project directory.
	3.	Run the script:

python spam_ham_classifier.py


	4.	Input an email when prompted to test the model’s prediction capability.

Key Metrics

	•	Accuracy on Training Data: Evaluates how well the model fits the training data.
	•	Accuracy on Testing Data: Validates the model’s performance on unseen data.

Example Output

Input email:

"Subject: password reset"

Output:

Result: Your Mail is Spam mail

Future Enhancements

	•	Integrating additional classifiers for comparison (e.g., Naive Bayes, SVM).
	•	Expanding the dataset for improved accuracy and generalizability.
	•	Deploying the model as a web application for real-time email classification.
