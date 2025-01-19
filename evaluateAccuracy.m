function accuracy = evaluateAccuracy(beta1, beta2, X, y)
%EVALUATEACCURACY calculates the prediction accuracy of the learned 
%neural network model using the testing data 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the percentage of accurately predicted examples 
%
%
m = size(X, 1); % number of examples
correct_predictions = 0;

for i = 1:m
    % Perform forward propagation to compute the outputs
    x1 = [1; X(i, :)']; % Add bias term
    
    sigmoid1 = beta1 * x1;
    x2 = [1; sigmoid(sigmoid1)]; % Add bias term to hidden layer
    
    sigmoid2 = beta2 * x2;
    x3 = sigmoid(sigmoid2);
    
    % Get the prediction
    prediction = round(x3);
    
    % Check if the prediction is correct
    if prediction == y(i)
        correct_predictions = correct_predictions + 1;
    end
accuracy = (correct_predictions / m) * 100;
end