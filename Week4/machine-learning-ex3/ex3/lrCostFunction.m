function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% y contains the 1's and 0's 

z = X*theta; % z parameter of the sigmoid function
% multiply X (5x4 matrix) with theta (a 4x1 matrix) gets a 5x1 matrix
H = sigmoid(z); % gives  5x4 matrix

% ex 1.3.1 Vectorizing the cost function for logistic regression
J = (1/m) * sum( -y.*log(H) - ((1 .- y) .* log(1 - H)));

% ex 1.3.2 Vectorizing the gradient
grad = (1/m) * sum((H .- y) .* X);

% ex 1.3.3 Adding regularization to the cost function
% note: "you should not be regularizing theta 0 which is used for the bias term"
J = J + ((lambda/(2*m)) * sum(theta(2:end).^2));


% gradient

% for j = 0, the regularization term is zero
% for j >= 1

tempTheta = theta;
tempTheta(1) = 0;

grad = grad(:) + ((lambda / m) * tempTheta);
% alternative: grad = ((1/m) * X' * (h_theta - y)) + (lambda / m) * temp;
% grad = grad(:);


% =============================================================



end
