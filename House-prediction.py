import pandas as pd
import numpy as np
import scipy as sp
import sklearn as sk
import math
from sklearn.metrics import mean_squared_error
import time
from sklearn.linear_model import LinearRegression
import random


df = pd.read_csv("houses_Madrid.csv")
print(f"The lenght {len(df.index)}")
print(f"The columns of the database {df.columns}")
df[["sq_mt_built", "buy_price"]].plot.scatter(x="sq_mt_built", y="buy_price")
df = df[["sq_mt_built", "buy_price"]].fillna(False)
## FIXME: add here the creation of the training data and test data

df_shuffled = df.sample(frac=1) # shuffle the rows
x = df_shuffled["sq_mt_built"].to_numpy(dtype=np.float64)
y = df_shuffled["buy_price"].to_numpy(dtype=np.float64)
training_data_x = x[:16000]
training_data_y = y[:16000]
test_data_x = x[16000:]
test_data_y = y[16000:]
training_data_y

# TODO: implement the loss function here using Python math ops and sklearn
import math
from sklearn.metrics import mean_squared_error
def loss_RMSE(y, yhat):# usingthe rmse formula to visualize the difference between the predicted values and the act vals Y
    actDiff = [actual - predicted for actual, predicted in zip(y, yhat)]
    squared_actDiff = [d ** 2 for d in actDiff]
    sum_squared_actDiff = sum(squared_actDiff)
    n = len(y)
    mean_squared_actDiff = sum_squared_actDiff / n
    rmse = math.sqrt(mean_squared_actDiff)
    return rmse

def loss_RMSE_sk(y, yhat):
    rmse = mean_squared_error(y, yhat, squared=False)
    return rmse
    # TODO: Now, run some experiments with your function, with the one taken with sklearn 
# Compare their outputs.
y = [5, 1, 2, 9]
yhat = [7.3,2.3, 1, 6]

rmse_homebrew = loss_RMSE(y, yhat)
rmse_sklearn = loss_RMSE_sk(y, yhat)
# 2 diff RMSE functions

print(f"my func for rmse: {rmse_homebrew}")
print(f"SK rmse: {rmse_sklearn}")

# TODO: implement the predictor function here
def predict(x, theta):
    #yhat predicted value for x ,x = input data, theta0 = the y intercept, theta1 = the slope
    yhat = theta[0] #starting point of data 
    #check if x is a list of features (basically an array) or a single value 
    if isinstance(x, list):
        for i in range(len(x)):
            yhat += theta [i+1] * x[i] #if x is a list of features it then proceeds to get multiplied by the corresponding weight
    else :
        yhat += theta[1] *x # in the case that its not a list we simply just multiply it by its weight
    return yhat
    # TODO: now, run some experiments with it

# helper code to visual the results better
def display_prediction(x, theta, yhat):
    print(f"Model Parameters (theta): {theta}")
    print(f"Input Features (x): {x}")
    print(f"Predicted Value (y_hat): {yhat}")
    print('-' * 50)
    
theta1 = [100000, 2500]
x1 = 200  
# Call the predict function
yhat_1 = predict(x1, theta1)
display_prediction(x1, theta1, yhat_1)

# TODO: implement the grid search function here 
def grid_search(training_data_x, training_data_y, grid0, grid1):
    #we do this to minimize the error on the training data 
    optimal_theta = [0,0]
    minimum_error = float(100000000000000000) # starting with rediculusly highnum so any actual error will be lower then it
    for t0 in grid0:
        for t1 in grid1:
            #nested loop implementation of a grid 
            theta = [t0, t1] 
            predicts = [predict(x, theta) for x in training_data_x] # where we predict the output using our theta 
            currMSE = sum((yhat - y) ** 2 for yhat, y in zip(predicts, training_data_y)) / len(training_data_y) # the MSE btw predict and actu
            
            if currMSE < minimum_error:
                minimum_error = currMSE
                optimal_theta = theta 
            
    return optimal_theta

# TODO: run some experiments with grid_search
# Define some grid values. Train it on the data set. Test it on the test set. 
# Print the loss on the data set and the test set. Measure and print how long the training takes.
import time
def run_experiment(training_data_x, training_data_y, test_data_x, test_data_y, grid0, grid1):
    # Measure training time
    start_time = time.time()
    best_theta = grid_search(training_data_x, training_data_y, grid0, grid1)
    duration = time.time() - start_time

    # Calculate training loss using sklearn's mean_squared_error
    train_predictions = [predict(x, best_theta) for x in training_data_x]
    train_loss = mean_squared_error(training_data_y, train_predictions)

    # Calculate test loss using sklearn's mean_squared_error
    test_predictions = [predict(x, best_theta) for x in test_data_x]
    test_loss = mean_squared_error(test_data_y, test_predictions)

    # Print results
    print(f"Training completed in {duration:.4f} seconds.")
    print(f"Training Loss (MSE): {train_loss}")
    print(f"Test Loss (MSE): {test_loss}")
    print(f"Best theta: {best_theta}")

# Define grid values for theta[0] (intercept) and theta[1] (weight)
grid0 = [0, 1, 2, 3, 4, 5]  # Grid for theta[0]
grid1 = [0.5, 1, 1.5, 2]    # Grid for theta[1]

# Run the experiment with your dataset
run_experiment(training_data_x, training_data_y, test_data_x, test_data_y, grid0, grid1)
# TODO: repeat the experimentation from above with different grids. 
# Finally, print the grid that provides the best value while still running faster 
# than 10 seconds.
def run_expw_time_constr(training_data_x, training_data_y, test_data_x, test_data_y, grid0, grid1):
    start_time = time.time()
    best_theta = grid_search(training_data_x, training_data_y, grid0, grid1)
    duration = time.time() - start_time

    if duration < 10:  # only using those < 10s
        train_predictions = [predict(x, best_theta) for x in training_data_x]
        train_loss = mean_squared_error(training_data_y, train_predictions)
        test_predictions = [predict(x, best_theta) for x in test_data_x]
        test_loss = mean_squared_error(test_data_y, test_predictions)
        return test_loss, duration, best_theta, grid0, grid1
        
grids = [
    (list(range(0, 101, 10)), list(range(10, 101, 10))),     
    (list(range(0, 201, 20)), list(range(10, 151, 20))),     
    (list(range(0, 501, 50)), list(range(50, 301, 50)))       
]
# Variables to keep best results
best_loss = float(10000000000000000000000000000000000000000000)
best_grid = None
best_theta = None
best_duration = 0

for grid0, grid1 in grids:
    test_loss, duration, theta, grid0_used, grid1_used = run_expw_time_constr(training_data_x, training_data_y, test_data_x, test_data_y, grid0, grid1)
    if test_loss < best_loss:
        best_loss = test_loss
        best_grid = (grid0_used, grid1_used)
        best_theta = theta
        best_duration = duration

if best_grid:
    print(f"Best Grid (Grid0: {best_grid[0]}, Grid1: {best_grid[1]}) completed in {best_duration:.4f} seconds.")
    # TODO: implement the random search function here

def random_search(training_data_x, training_data_y, trials):
    theta = [0, 0]
    best_mse = float(100000000000000000000000000000000000000)

    for _ in range(trials):
        # Randomly generate theta values
        theta0 = random.uniform(0, 1000) 
        theta1 = random.uniform(0, 1000) 
        
        theta = [theta0, theta1]
        
        # Get predictions and calculate MSE
        predictions = [predict(x, theta) for x in training_data_x]
        mse = mean_squared_error(training_data_y, predictions)
        
        # If current MSE is better, update theta
        if mse < best_mse:
            best_mse = mse

    return theta

    # TODO: run some experiments with random_search
# Choose some value for trial. Train it on the data set. Test it on the test set. 
# Print the loss on the data set and the test set. Measure and print how long the training takes.
def run_rando_exp(training_data_x, training_data_y, test_data_x, test_data_y, trials):
    start_time = time.time()
    best_theta = random_search(training_data_x, training_data_y, trials)
    
    # Measure training time
    duration = time.time() - start_time
    
    train_preds = [predict(x, best_theta) for x in training_data_x]
    test_preds = [predict(x, best_theta) for x in test_data_x]
    train_loss = mean_squared_error(training_data_y, train_preds)
    test_loss = mean_squared_error(test_data_y, test_preds)

    print(f"Training took {duration:.4f} seconds")
    print(f"Train Loss: {train_loss}")
    print(f"Test Loss: {test_loss}")
run_rando_exp(training_data_x, training_data_y, test_data_x, test_data_y, trials=1000)

# TODO: Implement here# Using sklearn's LinearRegression to solve the problem
from sklearn.linear_model import LinearRegression
def run_sklearn_experiment(train_x, train_y, test_x, test_y):
    start_time = time.time()

    # Train the model
    model = LinearRegression()
    model.fit(train_x.reshape(-1, 1), train_y)
    duration = time.time() - start_time
    train_preds = model.predict(train_x.reshape(-1, 1))
    test_preds = model.predict(test_x.reshape(-1, 1))

    # Calculate MSE
    train_loss = mean_squared_error(train_y, train_preds)
    test_loss = mean_squared_error(test_y, test_preds)

    theta_0_sklearn = model.intercept_
    theta_1_sklearn = model.coef_[0]

    # Output results
    print(f"Training with sklearn took {duration:.4f} seconds")
    print(f"Train Loss (MSE): {train_loss}")
    print(f"Test Loss (MSE): {test_loss}")
    return theta_0_sklearn, theta_1_sklearn, train_loss, test_loss, duration
theta_0_sklearn, theta_1_sklearn, train_loss_sklearn, test_loss_sklearn, duration_sklearn = run_sklearn_experiment(
    training_data_x, training_data_y, test_data_x, test_data_y
)
