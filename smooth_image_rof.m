function u = smooth_image_rof(f, lambda, epsilon)
% SMOOTH_IMAGE_ROF - perform ROF image restoration
% U = SMOOTH_IMAGE_ROF(F, LAMBDA, EPSILON) performs ROF denoising.
% Supports scalar or vector lambda and epsilon.
% Returns 4D array if LAMBDA or EPSILON are vectors.

    % Parameters
    maxIter = 200;        % Max number of iterations
    tol = 1e-4;           % Convergence tolerance
    [H, W] = size(f);
    
    % Convert to GPU if available
    useGPU = parallel.gpu.GPUDevice.isAvailable;
    if useGPU
        f = gpuArray(f);
    end

    % Ensure lambda and epsilon are row vectors
    lambda = lambda(:)'; 
    epsilon = epsilon(:)';
    K = numel(lambda);
    L = numel(epsilon);

    % Initialize output
    u = repmat(f, 1, 1, K, L);

    % Expand f to 4D for broadcasting
    F = repmat(f, 1, 1, K, L);
    Lambda = reshape(lambda, 1, 1, K, 1);
    Epsilon = reshape(epsilon, 1, 1, 1, L);

    Lambda = repmat(Lambda, 1, 1, 1, L);
    Epsilon = repmat(Epsilon, 1, 1, K, 1);

    for iter = 1:maxIter
        u_old = u;

        % Forward differences
        ux = circshift(u, [0 -1 0 0]) - u;
        uy = circshift(u, [-1 0 0 0]) - u;

        gradMag = sqrt(Epsilon.^2 + ux.^2 + uy.^2);

        px = ux ./ gradMag;
        py = uy ./ gradMag;

        % Backward differences (divergence)
        divx = px - circshift(px, [0 1 0 0]);
        divy = py - circshift(py, [1 0 0 0]);
        div_p = divx + divy;

        % Update u
        u = F - Lambda .* div_p;

        % Check convergence
        if max(abs(u(:) - u_old(:)), [], 'all') < tol
            break;
        end
    end

    % Gather result if on GPU
    if useGPU
        u = gather(u);
    end
end
