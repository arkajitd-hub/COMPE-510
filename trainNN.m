function [beta1, beta2, J_history] = trainNN(X, y, beta1, beta2, alpha, num_epochs)
%TRAINNN train the neural network model using backpropagation algorithm. It
%updates the weights, beta1 and beta2 using the training examples. It also
%generates the cost computed after each epoch. 

% useful values
[n, ~] = size(X); % n is number of training examples


J_history = zeros(num_epochs,1); % stores value of the cost function J at each iteration

for epoch = 1:num_epochs
% for each training example, do the following
    Jd = 0;
    for d = 1:n
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the weights beta1 and
    %               beta2. The key steps are indicated as follows
    %
    %
    
        a1 = [1; X(d, :)'];
        %% Step 1: forward propagate to generate the network output

        
        
        z2 = beta1 * a1;
        a2 = [1; sigmoid(z2)]; % Add bias term to hidden layer
        
        z3 = beta2 * a2;
        a3 = sigmoid(z3);  
        
        
        %% Step 2: for each output unit, calculate its error term
        % Recall that the number of output units is num_output

        delta3 = a3 - y(d, :);
        
        
        
        %% Step 3: for each hidden unit, calculate its error term
        % Recall that number of hidden units is num_hidden+1
 
        
        delta2 = (beta2' * delta3) .* [1; sigmoid(z2)];
        delta2 = delta2(2:end); % remove bias term       
        

        %% Step 4: update the weights using the error terms

        beta1 = beta1 - alpha * delta2 * a1';
        beta2 = beta2 - alpha * delta3 * a2'; 
        
        
        %% calculate the cost (Jd = SSE) per epoch
        Jd = Jd + sum(-y(d,:) .* log(a3) - (1 - y(d,:)) .* log(1 - a3));
    end
    J_history(epoch) = Jd/(2*n);
end