function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%


% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

X = [ones(m,1), X]; % adds bias unit "+1", matrix 5000x401
a1 = X;

z2 = Theta1 * a1'; %mat(25*401)*mat(401*5000) = mat(25*5000)
a2 = sigmoid(z2);
a2 = [ones(m,1), a2']; % adds bias unit "+1", matrix 5000x26

z3 = Theta2 * a2'; %mat(10*26)*mat(26*5000) = mat(10*5000)
a3 = sigmoid(z3);

h_theta = a3;

% transform y (indicates the index for each of the labels in X) into a matrix of "1" or "0" labels
new_y = zeros (num_labels, m); % new matrix with zeros with size 10x5000 page 5 ex4
for i=1:m,
  new_y(y(i),i) = 1; 
end

% Cost function without regularization
J = (1/m) * sum( sum( -new_y.*log(h_theta) - ((1 .- new_y) .* log(1 - h_theta))));

% Part 3 - Adding regularization to the cost function
% note: "you should not be regularizing theta 0 which is used for the bias term"
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));
J = J + ((lambda/(2*m)) * ( sum(sum(t1.^2)) + sum(sum(t2'.^2)) ) );



% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

sizeTheta1 = size(Theta1);
sizeTheta2 = size(Theta2);
deltaSum1 = zeros(sizeTheta1(1), sizeTheta1(2));
deltaSum2 = zeros(sizeTheta2(1), sizeTheta2(2));
 
for t=1:m,

% step 1

  % Set the input's layer values a(1) to the t-th training example x(t).
  a1 = X(t, :); % t vai de 0 a 5000, vou buscar cada linha e vai ficar um vector de 401 (já tem o bias unit)
  
  % Perform a feedforward pass, computing the activations (z(2), a(2), z(3), a(3)) for layers 2 and 3
  z2 = Theta1 * a1'; %mat(25*401)*mat(401*1) = mat(25*1)
  a2 = sigmoid(z2); % "0"s and "1"s, 
  a2 = [1, a2']; % adds bias unit "+1", vector of 26 positions or matrix 1x26

  z3 = Theta2 * a2'; %mat(10*26)*mat(26*1) = mat(10*1)
  a3 = sigmoid(z3);
  
% step 2

  % For each output unit k in layer 3 (the output layer), set  delta(3)(k) = (a(3)(k) - y(k))
  delta3 = a3 - new_y(:, t); % temos de ir buscar o elemento t-th do new_y, matrix 10x1
  
% step 3

  z2=[1; z2]; % adiciona unidade bias (26*1)
  % For the hidden layer (l=2), set delta(2) = (Gradient(2))' delta(3) .* g'(z(2))
  delta2 = (Theta2' * delta3) .* sigmoidGradient(z2);
  
% step 4

  % Accumulate the gradient from this example using the following formula. Note that you should
  % skip or remove delta zero of layer 2. That corresponds to delta2=delta2(2:end);
  % formula: 
  %             deltaSum(l) = deltaSum(l) + delta(l+1) (a(l))'
  
  delta2=delta2(2:end);
  
  deltaSum2 = deltaSum2 + delta3 * a2; % (10*1)*(1*26)
	deltaSum1 = deltaSum1 + delta2 * a1; % (25*1)*(1*401)

end;

% step 5

% Obtain regularized gradient for the NN cost function by dividing the accumulated gradients by 1/m

Theta2_grad = (1/m) * deltaSum2; % (10*26)
Theta1_grad = (1/m) * deltaSum1; % (25*401)


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
  
  
% Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m; % for j = 0
% 
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 
% 
% Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m; % for j = 0
% 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)]; 
  


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
