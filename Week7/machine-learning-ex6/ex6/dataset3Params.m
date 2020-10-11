function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_Test_Values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
Sigma_Test_Values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
error_array = [];
C_Val_Test_Array = [];
Sigma_Val_Test_Array = [];


for c_Test = C_Test_Values
  for sigma_Test = Sigma_Test_Values
    model= svmTrain(X, y, c_Test, @(x1, x2) gaussianKernel(x1, x2, sigma_Test)); 
    predictions = svmPredict(model, Xval);
    prediction_error = mean(double(predictions ~= yval));
    error_array = [error_array, prediction_error];
    C_Val_Test_Array = [C_Val_Test_Array, c_Test];
    Sigma_Val_Test_Array = [Sigma_Val_Test_Array, sigma_Test];
  endfor
endfor


[x,ind]=min(error_array);

C = C_Val_Test_Array(ind);
sigma = Sigma_Val_Test_Array(ind);


% =========================================================================

end
