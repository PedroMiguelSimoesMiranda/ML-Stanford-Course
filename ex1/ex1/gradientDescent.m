function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % Un-Vectorized solution

    theta_zero = theta(1);
    theta_one = theta(2);
    
    k = 1:m; % creates a vector of m positions going from 1 to m (in this case 20)
    h = X*theta; % OR use this un-Vectorized solution: theta_zero + theta_one .* X(k,2);
    sqrdErrorTheta0 = sum(h(k) - y(k));
    sqrdErrorTheta1 = sum((h(k) - y(k)).* X(k,2));
    theta_zero = theta(1) - (alpha/m)  * sqrdErrorTheta0;
    theta_one = theta(2) - (alpha/m)  * sqrdErrorTheta1;
    theta(1) = theta_zero;
    theta(2) = theta_one;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
