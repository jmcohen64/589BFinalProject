function u = smooth_image_rof(f, lambda, epsilon)
% SMOOTH_IMAGE_ROF - ROF image smoothing with Neumann BC and fixed-point update
% Inputs:
%   f       - noisy grayscale image, HxW, single precision
%   lambda  - smoothing parameter(s), scalar or vector
%   epsilon - regularization parameter(s), scalar or vector
% Output:
%   u       - smoothed image(s), HxW x K x L (4D)

    f = single(f);
    [H, W] = size(f);
    lambda = single(lambda(:)');
    epsilon = single(epsilon(:)');
    K = numel(lambda);
    L = numel(epsilon);

    % Broadcast input to 4D
    F = repmat(f, 1, 1, K, L);
    u = F;

    Lambda = reshape(lambda, 1, 1, K, 1);
    Epsilon = reshape(epsilon, 1, 1, 1, L);
    Lambda = repmat(Lambda, H, W, 1, L);
    Epsilon = repmat(Epsilon, H, W, K, 1);

    % Parameters
    maxIter = 1000;      % More iterations for convergence
    tol = 1e-6;          % Tighter tolerance

    for iter = 1:maxIter
        u_old = u;

        % Neumann boundary condition via replicate padding
        u_pad = padarray(u, [1 1 0 0], 'replicate');

        % Forward differences
        ux = u_pad(2:end-1, 3:end, :, :) - u_pad(2:end-1, 2:end-1, :, :);
        uy = u_pad(3:end, 2:end-1, :, :) - u_pad(2:end-1, 2:end-1, :, :);

        % Gradient magnitude with numerical epsilon
        gradMag = sqrt(ux.^2 + uy.^2 + Epsilon.^2 + 1e-12);

        px = ux ./ gradMag;
        py = uy ./ gradMag;

        % Pad px and py for divergence computation
        px_pad = padarray(px, [0 1 0 0], 0, 'pre');
        py_pad = padarray(py, [1 0 0 0], 0, 'pre');

        % Backward differences (divergence)
        divx = px_pad(:, 2:end, :, :) - px_pad(:, 1:end-1, :, :);  % size H x (W+1)
        divy = py_pad(2:end, :, :, :) - py_pad(1:end-1, :, :, :);  % size (H+1) x W

        % Crop to H x W
        divx = divx(:, 1:W, :, :);
        divy = divy(1:H, :, :, :);
        div_p = divx + divy;

        % Assignment-specified fixed-point update
        u = F - Lambda .* div_p;

        % Convergence check
        if max(abs(u(:) - u_old(:)), [], 'all') < tol
            break;
        end
    end

    u = single(u);
end

