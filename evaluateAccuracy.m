function accuracy = evaluateAccuracy(beta, X, y)
%EVALUATEACCURACY calculates the average prediction error of the learned 
%linear regression model using the testing data 

m = length(y); % number of testing examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the prediction error (mean squared error) using the
%               testing set.
%
predicted_probabilities = 1 ./ (1 + exp(-X * beta));
threshold = 0.5;
predicted_labels = predicted_probabilities >= threshold;
accuracy = sum(predicted_labels == y) / length(y);
accuracy = accuracy * 100; %to get percentage

% ============================================================

end