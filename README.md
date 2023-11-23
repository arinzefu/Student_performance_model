# Student_performance_model

The dataset is from https://archive.ics.uci.edu/dataset/320/student+performance
 
This Python script conducts exploratory data analysis (EDA) on a dataset representing student performance. It generates and displays heatmaps using Seaborn to depict the relationship between specific columns and the final grade ('G3'). The next step involves encoding selected categorical columns using LabelEncoder. The script then proceeds to split the data into features (X) and the target variable (y) and applies 'StandardScaler' to standardize numerical values.

Following this, the script constructs a neural network model using TensorFlow and Keras. The model comprises multiple dense layers with ReLU activation, batch normalization, and dropout for regularization. Subsequently, the script compiles the model using the Adam optimizer, mean squared error as the loss function, and mean absolute error as a metric. The model undergoes training and is utilized to predict the final grades on the test set.

In summary, the regression model demonstrates high accuracy in predicting the final grades of the students.