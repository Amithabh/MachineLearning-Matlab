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
X = [ones(size(X,1), 1) X];
z_2 = X*Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(size(a_2,1), 1) a_2];
z_3 = a_2*Theta2';
a_3 = sigmoid(z_3);
[v,p] = max(a_3, [], 2);

% Test case:
% Theta1 = reshape(sin(0 : 0.5 : 5.9), 4, 3);
% Theta2 = reshape(sin(0 : 0.3 : 5.9), 4, 5);
% X = reshape(sin(1:16), 8, 2);
% Result:
% p = predict(Theta1, Theta2, X)
% % you should see this result
% p = 
  % 4
  % 1
  % 1
  % 4
  % 4
  % 4
  % 4
  % 2
 % The number of activation units in final layer is equal to the number of distinct classes in the classification 
 % problem which in the test case is 3 and so the 3 activation units are a(3)_1, a(3)_2 and a(3)_3.
 % The principle for solving any neural network problem is the following:
 % 1) First ensure that the training set X has a row for each training example and then add a column for the bias unit.
 % 2) Suppose there are m examples in the training set and has 3 inputs including x0. If Theta1 is given, one of its
 % dimension should match the training set. 
% =========================================================================


end
