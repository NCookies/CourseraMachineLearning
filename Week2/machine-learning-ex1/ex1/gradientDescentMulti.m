function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters, mu, sigma)
    %GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    %   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
        %       of the cost function (computeCostMulti) and gradient here.
        %

        h_theta = 0;

        %for nor = 2:size(X, 2)
        %    X(:, nor) - mu(nor - 1)
        %    (X(:, nor) - mu(nor - 1)) .* (1/sigma(nor - 1))
        %end

        X(:, 2) = (X(:, 2) - mu(1, :)) .* (1/sigma(1))
        X(:, 3) = (X(:, 3) - mu(2, :)) .* (1/sigma(2))

        for h = 1:size(X, 2)
            h_theta += theta(h) * X(:, h);
        end

        for tmp_iter = 1:size(X, 2)
            tmp(tmp_iter) = theta(tmp_iter) - alpha * (1/m) * sum((h_theta - y) .* X(:, tmp_iter));
        end

        % theta = tmp;

        % =============================================================================================

        % tmp1 = theta(1) - alpha * (1/m) * sum(h_theta - y);
        % tmp2 = theta(2) - alpha * (1/m) * sum((h_theta - y) .* X(:, 2));

        %h_theta = theta(1) * X(:, 1) + theta(2) * X(:, 2) + theta(3) * X(:, 3);

        %tmp1 = theta(1) - alpha * (1/m) * sum(h_theta - y);
        %tmp2 = theta(2) - alpha * (1/m) * sum((h_theta - y) .* X(:, 2));
        %tmp3 = theta(3) - alpha * (1/m) * sum((h_theta - y) .* X(:, 3));

        %theta = [tmp1; tmp2; tmp3];

        % =============================================================================================

        % fprintf('h_theta : %.3f \n', h_theta);
        % fprintf('theta : %.3f \t %.3f \t %.3f \n', theta(1), theta(2), theta(3));

        % plot(X, theta);
        % hold on;
        

        % ============================================================

        % Save the cost J in every iteration    
        J_history(iter) = computeCostMulti(X, y, theta);

    end 

end
