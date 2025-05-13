function u = rof_gradient_descent(f, lambda, epsilon)
% ROF_GRADIENT_DESCENT - Improved gradient descent solution of ROF model.
% Inputs:
%   f        - input image (HxW), double or single
%   lambda   - scalar or vector of smoothing parameters
%   epsilon  - scalar or vector of regularization parameters
% Output:
%   u        - smoothed image(s), size HxW x K x L

    %if ~isa(f, 'single') && ~isa(f, 'double')
    %    f = single(f); % Force consistent precision
    %end
    f = double(f);

    [H, W] = size(f);
    lambda = lambda(:)';
    epsilon = epsilon(:)';
    K = numel(lambda);
    L = numel(epsilon);

    % Expand input into 4D grid
    f = reshape(f, H, W, 1, 1);
    u = repmat(f, 1, 1, K, L);
    F = u;

    Lambda = reshape(lambda, 1, 1, K, 1);
    Eps2 = reshape(epsilon.^2, 1, 1, 1, L);
    Lambda = repmat(Lambda, H, W, 1, L);
    Eps2 = repmat(Eps2, H, W, K, L);

    % Parameters
    maxIter = 300;       % Allow more iterations for better convergence
    tau = 0.125;         % Slightly reduced step size for stability
    tol = 1e-7;          % Tighter convergence tolerance

    for iter = 1:maxIter
        u_old = u;

        % Neumann BC via symmetric padding
        u_pad = padarray(u, [1 1], 'symmetric');

        % Forward differences
        ux = u_pad(2:end-1, 3:end, :, :) - u_pad(2:end-1, 2:end-1, :, :);
        uy = u_pad(3:end,   2:end-1, :, :) - u_pad(2:end-1, 2:end-1, :, :);

        % Gradient magnitude
        mag = sqrt(ux.^2 + uy.^2 + Eps2);

        % Avoid division by zero
        px = ux ./ (mag + eps);
        py = uy ./ (mag + eps);

        % Divergence (backward difference)
        px_pad = padarray(px, [0 1], 0, 'pre');
        py_pad = padarray(py, [1 0], 0, 'pre');
        div_p = (px_pad(:, 2:end, :, :) - px_pad(:, 1:end-1, :, :)) + ...
                (py_pad(2:end, :, :, :) - py_pad(1:end-1, :, :, :));

        % Gradient descent update
        u = u + tau * (div_p - (1 ./ Lambda) .* (u - F));

        % Convergence check every few iterations to reduce overhead
        if mod(iter, 5) == 0 || iter == maxIter
            rel_change = norm(u(:) - u_old(:)) / (norm(u_old(:)) + eps);
            if rel_change < tol
                break;
            end
        end
    end

    % Ensure output is same type as input
    u = cast(u, 'like', f);
end
