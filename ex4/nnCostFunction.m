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
% Add ones do the matrix X.
% ====================== YOUR CODE HERE ======================
%Feed-forward
X = [ones(m,1) X];
z2 = X * Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
  
% Recode Y.
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end

% Cost Function.
J = (1/m) * sum(sum(-Y .* log(a3) - (1 - Y) .* log(1 - a3)));

% Add regularized error. Drop the bias terms in the 1st columns.
J = J + (lambda / (2*m)) * sum(sum(Theta1(:, 2:end) .^ 2));
J = J + (lambda / (2*m)) * sum(sum(Theta2(:, 2:end) .^ 2));

s3 = a3 - Y;
s2 = (s3 * Theta2 .* sigmoidGradient([ones(size(z2, 1), 1) z2])) (:, 2:end);
d1 = s2' * X;
d2 = s3' * a2;

Theta1_grad = d1./m + (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = d2./m + (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:,2:end)];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
