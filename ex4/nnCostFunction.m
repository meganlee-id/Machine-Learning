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
%   1) you need to manually fill in all the bias col for each layer
%.  2) for y is (m,1) matrix, you need to first convert it into a (m, K) matrix
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
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



% =========================== Part 1 Forward propagation ============================%
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% define commonly used variables
bias_col = ones(m, 1);
thetas = {Theta1, Theta2};
num_of_layers = length(thetas) + 1;
Z = {X};  % store Z = {X, z1, z2}. X here is a placeholder to make z_i index in sync with a_i
A = {X};  % store A = {X, a1, a2/output}
theta_square = 0;

% reshape y: m * 1 into y_vector: m * k matrix
% this might help you:
% eye(5)(1,:):  1   0   0   0   0
% eye(5)(2,:):  0   1   0   0   0
% eye(5)(3,:):  0   0   1   0   0
% eye(5)(4,:):  0   0   0   1   0
% eye(5)(5,:):  0   0   0   0   1
% eye(5)(1, :): the 1st row
% eye(5)(:, 2): the 2nd col
y_vector = eye(num_labels)(y',:); % [m, num_labels]

% calculate forward propagation and store in A list
for i = 1:length(thetas)
    % forward propagation
    A{i} = [bias_col, A{i}];
    Z{i + 1} = A{i} * (thetas{i})';
    % store current z and a
    A{i + 1} = sigmoid(Z{i + 1});
    % regularization, add theta up for the current layer
    theta_square += sum(thetas{i}(:,2:end)(:) .^ 2); % chop 1st col off, 1st col doesn't have to be 1
end

% logistic regression cost part
output = A{num_of_layers};
cost_total = log(output) .* y_vector + log(1 - output) .* (1 - y_vector);
cost_avg = (-1 / m) * sum(cost_total(:));

% theta regularization part
theta_square_avg = (lambda / (2 * m)) * theta_square;

% calculate the cost
J = cost_avg + theta_square_avg;


% =========================== Part 2 Back-prop ============================%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

% for each layer backwards
d = {} % {_, d2, d3} length(d) == length(thetas)
for layer = num_of_layers:-1:2
    if layer == num_of_layers
        d{layer} = output - y_vector;
    else
        d{layer} = d{layer + 1} * thetas{layer}(:,2:end) .* sigmoidGradient(Z{layer});
    end
end


D = {}; % same as theta; j -> j+1, accumulator
for layer=1:length(thetas)
    thetas{layer}(:,1) = 0; % clear first col
    D{layer} = lambda * thetas{layer} + (d{layer + 1})' * A{layer};
end


Theta1_grad = D{1} ./ m;
Theta2_grad = D{2} ./ m;


% =========================== Output ============================%

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
