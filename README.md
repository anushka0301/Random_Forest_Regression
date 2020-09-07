# Random_Forest_Regression

Random Forest Regression is based on "Ensemble Training".

Step 1: Pick at random K data points from the training set.
Step 2: Build the Decision Tree associated to these K points.
Step 3: Choose the number of Ntree of trees you want to build and repeat 1 and 2.
Step 4: For a new data point, make each one of your Ntree trees predict the value of Y for the data point in question and assign the new data point the average across all of the predicted Y values.
