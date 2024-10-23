#Data_Science

Project 1 - Titanic Classification 

The Titanic Classification Project is a venture dedicated to employing machine learning and data analysis techniques to forecast passenger survival on the RMS Titanic. The sinking of the Titanic is one of history's most notorious maritime disasters, and this initiative aims to illuminate the determinants of passenger survival. By scrutinizing historical data, we can create a model for predicting whether a given passenger survived or perished. The project encompasses several core components, including data preprocessing, model selection, training, evaluation, and prediction. It makes use of a well-recognized dataset, the "Titanic: Machine Learning from Disaster" dataset available on Kaggle, to carry out these tasks. The dataset offers valuable insights into passengers, encompassing attributes like age, gender, ticket class, and more, alongside a binary indicator of their survival outcome.

1. Data Preprocessing:
   - Managing missing data: Identifying and addressing gaps in the dataset to ensure data cleanliness and completeness.
   - Categorizing features: Converting categorical attributes like gender and class into numerical formats suitable for machine learning models.
   - Selecting pertinent features: Choosing attributes that contribute to the prediction task.

2. Model Selection:
   - Opting for a suitable machine learning model for classification, with options including Decision Trees, Random Forest, Logistic Regression, and Support Vector Machines (SVM).

3. Model Training:
   - Training the selected model with the cleaned and preprocessed training data. This phase involves the model learning patterns and relationships within the data.

4. Model Evaluation:
   - Assessing the model's performance using diverse evaluation metrics, including accuracy, precision, recall, F1-score, and the receiver operating characteristic (ROC) area under the curve (AUC).
   - The evaluation step gauges the model's effectiveness in predicting passenger survival.

5. Hyperparameter Tuning:
   - Fine-tuning model hyperparameters to enhance its predictive accuracy, employing techniques like grid search and cross-validation.

6. Prediction:
   - Upon model training and evaluation, the model is applied to predict survival outcomes for the test dataset, aiming to determine whether passengers survived or not.

7. Output Generation:
   - Preparing the final predictions in a format suitable for submission, usually in a CSV file comprising passenger IDs and survival predictions.


The Titanic Classification Project is a testament to the power of data analysis and predictive modeling, illustrating how these tools can uncover insights and make predictions from historical data. By delving into the Titanic dataset, we aim to reveal the determinants of passenger survival and construct a predictive model. This endeavor showcases the capacity of data-driven analysis to extract meaningful insights from historical events.


Project 2 - Number Recognition with the MNIST Dataset

In this project, we showcase a Python script that utilizes a deep learning model to recognize handwritten numerals. It leverages the MNIST dataset for training and employs a pre-trained model to make predictions on digit images stored in a specific desktop folder. The code begins by loading and preparing the MNIST dataset, training a neural network model, and subsequently applying this model to predict digits in images stored in the designated directory.

1. Import Necessary Libraries:

- Import the 'os' library for file system operations.
- Utilize 'cv2' for image processing.
- Incorporate 'numpy' for numerical computations.
- Leverage 'matplotlib' for visualizing images.
- Rely on 'tensorflow' for machine learning and deep learning tasks.

2. Load and Preprocess the MNIST Dataset:

- Load the MNIST dataset, consisting of images of handwritten digits along with their corresponding labels.
- Normalize the pixel values of the images to ensure they fall within the range of 0 to 1, enhancing training efficiency.

3. Define a Neural Network Model:

- Create a Sequential model using TensorFlow/Keras.
- Flatten the 28x28 pixel images into a 1D array.
- Add two dense layers with ReLU activation functions for feature extraction.
- Append the output layer with 10 units and employ softmax activation for digit classification.
- Compile the model using the Adam optimizer and sparse categorical cross-entropy loss.

4. Train the Model:

- Train the model using the training data (x_train and y_train) for three epochs.
- As a result, the model becomes capable of classifying handwritten digits based on the training dataset.

5. Evaluate the Model:

- Assess the performance of the trained model on the test data (x_test and y_test).
- Provide reports on the model's loss and accuracy on the test dataset.

6. Load and Make Predictions on Desktop Images:

- Define the path to the folder containing digit images on the desktop.
- List all files in the folder with the ".png" extension.
- Iterate through the image files: Load each image using OpenCV and extract the first channel (assuming grayscale).
- Adjust the image colors if necessary.
- Utilize the trained model to predict the digit in the image.
- Display both the image and the predicted digit using matplotlib.
- Handle any exceptions or errors that may occur during the process.

This code serves as an illustration of the predictive capabilities of a neural network model trained on the MNIST dataset. It outlines the steps involved in loading, preprocessing, and predicting digits from a designated folder using the pre-trained model.
