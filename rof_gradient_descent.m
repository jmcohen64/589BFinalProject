function u = rof_gradient_descent(f, lambda, epsilon)
% ROF_GRADIENT_DESCENT - Gradient descent solver for ROF model
% u = rof_gradient_descent(f, lambda, epsilon)
%
% Inputs:
%   f       - noisy grayscale image
%   lambda  - regularization parameter
%   epsilon - smoothing parameter for total variation
%
% Output:
%   u       - denoised image

    % Parameters
    [H, W] = size(f);
    maxIter = 500;
    tau = 0.125;         % Time step (should be < 1/4 for stability)
    tol = 1e-5;

    u = f;               % Initial guess

    for k = 1:maxIter
        u_old = u;

        % Compute forward differences
        ux = circshift(u, [0 -1]) - u;
        uy = circshift(u, [-1 0]) - u;

        grad_mag = sqrt(ux.^2 + uy.^2 + epsilon^2);

        % Compute divergence (backward differences)
        px = ux ./ grad_mag;
        py = uy ./ grad_mag;

        divx = px - circshift(px, [0 1]);
        divy = py - circshift(py, [1 0]);
        div_p = divx + divy;

        % Gradient descent update
        u = u + tau * (div_p - (1/lambda) * (u - f));

        % Check for convergence
        if norm(u - u_old, 'fro') / norm(u, 'fro') < tol
            break;
        end
    end
end
