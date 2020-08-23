function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%



% ============= Taken from figure 2 NN model =============
% a(layer1)=x (add a(layer1) subscript0
% add vector (matrix coluna) to X containing just 1s
% we get a 5000x401 vector
X = [ones(size(X,1),1), X]; 


% ============= Taken from the exercise's pdf =============
% Implementation Note: The matrix X contains the examples in rows. 
% When you complete the code in predict.m, you will need to add the 
% column of 1’s to the matrix. The matrices Theta1 and Theta2 contain 
% the parameters for each unit in rows. Specifically, the first row 
% of Theta1 corresponds to the first hidden unit in the second layer. 
% In Octave/MAT- LAB, when you compute z(layer2) = Θ(layer1)a(layer1), be sure that 
% you index (and if necessary, transpose) X correctly so that you 
% get a(l) as a column vector.

% X contains units in rows, Theta1 contains units in rows, so we must 
% make Theta1 a transpose so that unit is computed with corresponding unit
% Personal note: to know why we need to multiply X * Theta1' 
% and go to the notebook and check out AND OR examples, check the z parameter g(z)
ZLayer2 = X * Theta1'; 
HLayer2 = sigmoid(ZLayer2); % H layer 2 or hypothesis layer 2 matrix containing outputs of layer 2

% Theta2 is 10x26 (we have a bias unit weight here so we must add a bias unit to action unit matrix HLayer2)
HLayer2 = [ones(size(HLayer2,1),1), HLayer2]; 
% X contains units in rows, Theta2 contains units in rows, so we must 
% make Theta2 a transpose so that unit is computed with corresponding unit
ZLayer3 = HLayer2 * Theta2';
HLayer3 = sigmoid(ZLayer3); % output hypothesis, from output layer 3

% following the hint use the max function
[val, index] = max(HLayer3, [], 2);
p = index;


% =========================================================================


end
