ANN Classifier With Trainable Activation in TensorFlow/Keras, with LIME for Binary Classification - Base problem category as per Ready Tensor specifications.

- tensorflow
- keras
- neural network
- LIME
- XAI
- interpretable
- pandas
- numpy
- activation
- python
- adam optimizer
- binary classification
- scikit-optimize
- flask
- nginx
- uvicorn
- docker
- feature-engine

This Artificial Neural Network (ANN) is comprised of 3 layers, one of which includes a layer with trainable parameters to control the slope of the activation function, that uses the Adam optimizer to evaluate the performance of the model.

Model also has LIME implementation to get local explanations on impact of the inputs on the target.

The data preprocessing step includes missing data imputation, standardization, one-hot encoding for categorical variables, datatype casting, etc. The missing categorical values are imputed using the most frequent value if they are rare. Otherwise if the missing value is frequent, they are give a "missing" label instead. Missing numerical values are imputed using the mean and a binary column is added to show a 'missing' indicator for the missing values. Numerical values are also scaled using a Yeo-Johnson transformation in order to get the data close to a Gaussian distribution.

The Hyperparameter Tuning (HPT) involves finding the optimal values for the L1 and L2 regularization, learning rate for Adam optimizer, and the number of changepoints for the trainable activation layer.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as email spam detection, customer churn, credit card fraud detection, cancer diagnosis, and titanic passanger survivor prediction.

The main programming language is Python. The main algorithm is created using TensorFlow and Keras while Scikitlearn is used to calulate the model metrics and preprocess the data. Feature-engine, pandas, and numpy are used to preprocess the data while Scikit-Optimize is used for HPT. Flask, Nginx, gunicorn are used to allow for web services. The web service provides three endpoints- /ping for health check, /infer for predictions, /explain for local explanations in real time.
