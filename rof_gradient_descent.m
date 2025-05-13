function u = rof_gradient_descent(f, lambda, epsilon)
% ROF_GRADIENT_DESCENT - Gradient descent solution of ROF model.
% Inputs:
%   f        - input image (HxW), single precision
%   lambda   - scalar or vector of smoothing parameters
%   epsilon  - scalar or vector of regularization parameters
% Output:
%   u        - smoothed image(s), size HxW x K x L

    f = double(f);
    [H, W] = size(f);
    lambda = single(lambda(:)');
    epsilon = single(epsilon(:)');
    K = numel(lambda);
    L = numel(epsilon);

    % Initialize
    u = repmat(f, 1, 1, K, L);  % Initial guess
    F = u;

    Lambda = reshape(lambda, 1, 1, K, 1);
    Eps2 = reshape(epsilon.^2, 1, 1, 1, L);
    Lambda = repmat(Lambda, H, W, 1, L);
    Eps2 = repmat(Eps2, H, W, K, 1);

    % Parameters
    maxIter = 1000;
    tau = 0.2;
    tol = 1e-5;

    for iter = 1:maxIter
        u_old = u;

        % Neumann BC via symmetric padding
        u_pad = padarray(u, [1 1], 'symmetric');

        % Forward differences
        ux = u_pad(2:end-1, 3:end, :, :) - u_pad(2:end-1, 2:end-1, :, :);
        uy = u_pad(3:end,   2:end-1, :, :) - u_pad(2:end-1, 2:end-1, :, :);

        mag = sqrt(ux.^2 + uy.^2 + Eps2);
        px = ux ./ (mag + eps);
        py = uy ./ (mag + eps);

        % Divergence via backward difference with 0-padding
        px_pad = padarray(px, [0 1], 0, 'pre');
        py_pad = padarray(py, [1 0], 0, 'pre');

        div_p = (px_pad(:, 2:end, :, :) - px_pad(:, 1:end-1, :, :)) + ...
                (py_pad(2:end, :, :, :) - py_pad(1:end-1, :, :, :));

        % Gradient descent update
        u = u + tau * (div_p - (1 ./ Lambda) .* (u - F));

        % Convergence check
        rel_change = norm(u(:) - u_old(:), 2) / (norm(u_old(:), 2) + eps);
        if rel_change < tol
            break;
        end
    end

    u = double(u);  % Ensure output precision
end
