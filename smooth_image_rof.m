function u = smooth_image_rof(f, lambda, epsilon)
% SMOOTH_IMAGE_ROF - ROF image restoration with fixed-point scheme.
% Input:
%   f        - noisy input image (HxW), single precision
%   lambda   - scalar or vector of smoothing parameters
%   epsilon  - scalar or vector of regularization parameters
% Output:
%   u        - denoised image(s), size HxW x K x L

    f = double(f);
    [H, W] = size(f);
    lambda = lambda(:)';
    epsilon = epsilon(:)';
    K = numel(lambda);
    L = numel(epsilon);

    % Broadcast parameters
    F = repmat(f, 1, 1, K, L);
    u = F;  % initial guess

    Lambda = reshape(lambda, 1, 1, K, 1);
    Epsilon = reshape(epsilon, 1, 1, 1, L);
    Lambda = repmat(Lambda, H, W, 1, L);
    Eps2 = Epsilon .^ 2;

    % Settings
    maxIter = 200;
    tol = 1e-6;

    for iter = 1:maxIter
        u_old = u;

        % Symmetric padding (Neumann BC)
        u_pad = padarray(u, [1 1], 'symmetric');

        % Forward differences
        ux = u_pad(2:end-1, 3:end, :, :) - u_pad(2:end-1, 2:end-1, :, :);
        uy = u_pad(3:end,   2:end-1, :, :) - u_pad(2:end-1, 2:end-1, :, :);

        % Gradient magnitude + eps
        mag = sqrt(ux.^2 + uy.^2 + Eps2);
        px = ux ./ (mag + eps);
        py = uy ./ (mag + eps);

        % Zero-pre-padding for divergence
        px_padded = padarray(px, [0 1], 0, 'pre');
        py_padded = padarray(py, [1 0], 0, 'pre');

        % Backward divergence
        div_p = (px_padded(:, 2:end, :, :) - px_padded(:, 1:end-1, :, :)) + ...
                (py_padded(2:end, :, :, :) - py_padded(1:end-1, :, :, :));

        % Fixed-point update
        u = F - Lambda .* div_p;

        % Convergence check
        rel_change = norm(u(:) - u_old(:), 2) / (norm(u_old(:), 2) + eps);
        if rel_change < tol
            disp("converged within maxiter")
            break;
        end
    end

    u = double(u);  % Ensure output is single
end
