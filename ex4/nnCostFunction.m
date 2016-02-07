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
Theta1_grad = zeros(size(Theta1,1));
Theta2_grad = zeros(size(Theta2,1));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Pull out number of data points
m;

% Convert y vector into vector matrix
yorig = y;
y = zeros(num_labels, size(yorig,1));
for i = 1:size(yorig, 1)
	y(yorig(i), i) = 1;
end

% Add column of 1's to design matrix
X = [ones(m, 1) X];

% Calculate cost function (unregularized)
sum = 0;
for i = 1:m
	% Calculate output layer for input data point (row i of design matrix X)
	output = sigmoid(Theta2 * [1; sigmoid(Theta1 * X(i,:)')]);
	
	for k = 1:num_labels
		% Calculate term of cost function for input data point i
		yvalue = y(k,i);
		term = ( (-1 * yvalue) * log(output(k)) - ((1 - yvalue) * log( 1 - output(k))) );
		
		% Add the term to running sum
		sum = sum + term;
	end
end
jUnreg = (1/m) * sum;


% Calculate regularization term
sum = 0;
% hidden_layer_size
% size(X,2)
% X(1:10,1)
% X(1:10,size(X,2))
% num_labels
for j = 1:(hidden_layer_size)
	for k = 2:size(X,2)
		sum = sum + (Theta1(j, k)) ^ 2;
	end
end
for j = 1:num_labels
	for k = 2:(hidden_layer_size+1)
		sum = sum + (Theta2(j, k)) ^ 2;
	end
end
% sum
% sum = 0;
% for j = 1:25
% 	for k = 2:401
% 		sum = sum + (Theta1(j, k)) ^ 2;
% 	end
% end
% for j = 1:10
% 	for k = 2:26
% 		sum = sum + (Theta2(j, k)) ^ 2;
% 	end
% end
% size(Theta1)
% size(Theta2)
% sum
Reg = (lambda/(2*m)) * sum;
% sum = 0;
% Reg = sum(Theta1(:).^2) + sum(Theta2(:).^2);



% Calculate cost function (regularized)
J = jUnreg + Reg;













% Calculate gradients using backpropagation
bigdelta2 = zeros(num_labels,hidden_layer_size+1);
bigdelta1 = zeros(hidden_layer_size, input_layer_size+1);

for t = 1:m
	% Step 1
	a1 = X(t,:)';
% 	a1 = [1; a1];   % Looks like I added a col of 1's to design matrix
% 	already
%     size(Theta1);
%     size(a1);
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1; a2];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	% Step 2
	delta3 = zeros(num_labels,1); % only 1 column, not m cols
	for k = 1:num_labels
		delta3(k) = a3(k) - y(k,t);
        % This is the error of node k in layer 3
	end
	
	% Step 3
	size(Theta2'); % endless output
% 	size(delta3(:,t))
	size(sigmoidGradient(z2)); % endless output
	delta2 = (Theta2)' * delta3 .* (sigmoidGradient([1; z2]));
        % This is the error of each of the 26 nodes in layer 2
	
	% Step 4
	bigdelta2 = bigdelta2 + delta3 * a2';
	bigdelta1 = bigdelta1 + delta2(2:end) * a1';
end
% Step 5
Theta1_grad = (1/m) * bigdelta1;
Theta2_grad = (1/m) * bigdelta2;

% 
% 	for k = 1:num_labels
% 		delta3(k,t) = a3(k) - y(k)
% 	end

% Regularized gradient
Theta1_grad(:,2:size(Theta1_grad,2)) = ...
    Theta1_grad(:,2:size(Theta1_grad,2)) + ...
    (lambda/m) * Theta1(:,2:size(Theta1_grad,2));
Theta2_grad(:,2:size(Theta2_grad,2)) = ...
    Theta2_grad(:,2:size(Theta2_grad,2)) + ...
    (lambda/m) * Theta2(:,2:size(Theta2_grad,2));
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
