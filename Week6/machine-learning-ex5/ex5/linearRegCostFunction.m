function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% AUX

hypothesis = X*theta; 
squared_errors = (hypothesis - y).^2; 
normalLR = (1/(2*m))*sum(squared_errors);
regTerm = (lambda/(2*m))*sum(theta(2:end).^2);

% FINAL EXPRESSION 

J = normalLR + regTerm;


% =========================================================================

% Note: for j = 0, the regularization term is zero 

tempTheta = theta;
tempTheta(1) = 0;

grad = (1/m)*(sum((hypothesis .- y).*X));
gradRegTerm = (lambda/m)*tempTheta;

grad = grad(:) + gradRegTerm; % we need to do grad(:) to add two matrixes with the same dimensions

% alternative:
%grad = (1/m) * (sum((hypothesis .- y) .* X));
%grad = grad(:) + ((lambda / m) * tempTheta);
% alternative: grad = ((1/m) * X' * (h_theta - y)) + (lambda / m) * temp;

grad = grad(:);

end
