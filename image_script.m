%----------------------------------------------------------------
% File: image_script.m
%----------------------------------------------------------------
%
% Author: Marek Rychlik (rychlik@arizona.edu)
% Date: Mon Apr 7 14:19:32 2025
% Copying: (C) Marek Rychlik , 2020. All rights reserved.
%
%----------------------------------------------------------------
% Basic operations on raw images
% Source image: https://www.reddit.com/r/EditMyRaw/comments/1jt4ecw the_official_weekly_raw_editing_challenge/

% Updated image path
raw_img_filename = fullfile('home', 'u32', 'jmc696', '589BFinalProject', 'DSC00099.ARW');

% Set up tiled layout
t = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
linked_axes = [];

% Read raw data and info
cfa = rawread(raw_img_filename);
ax = nexttile; imagesc(bitand(cfa, 7)); title('xor with 7');
linked_axes = [linked_axes, ax];

info = rawinfo(raw_img_filename);
disp(info);

% Extract color planes
Iplanar = raw2planar(cfa);
planes = 'rggb';
ang = info.ImageSizeInfo.ImageRotation;

for j = 1:size(Iplanar, 3)
    ax = nexttile;
    imagesc(imrotate(Iplanar(:,:,j), ang));
    title(['Plane ', num2str(j), ': ', planes(j)]);
    colorbar; colormap gray;
    linked_axes = [linked_axes, ax];
end

% Demosaicing and color
Irgb = raw2rgb(raw_img_filename);
ax = nexttile;
image(Irgb);  % No resizing or scaling
title('Demosaiced RGB image');
colorbar;
linked_axes = [linked_axes, ax];

% --- Apply ROF smoothing ---
gray_img = rgb2gray(Irgb);                  % Convert to grayscale
gray_img = double(gray_img);               % Force single precision, no scaling

lambda = 0.1;
epsilon = 1e-2;
smoothed = smooth_image_rof(gray_img, lambda, epsilon);

% --- Gradient descent ROF smoothing ---
u_grad = rof_gradient_descent(gray_img, lambda, epsilon);
u_grad = double(u_grad);  % Match class with smoothed

% Compare numerically
disp(class(smoothed))
disp(class(u_grad))
mse_val = immse(smoothed, u_grad);
psnr_val = psnr(smoothed, u_grad);

fprintf('MSE between smooth_image_rof and gradient descent: %.6f\n', mse_val);
fprintf('PSNR between smooth_image_rof and gradient descent: %.2f dB\n', psnr_val);

% Display ROF result
ax = nexttile;
imagesc(smoothed);
title('ROF Smoothed Grayscale');
colorbar; colormap gray;
linked_axes = [linked_axes, ax];

% Display Gradient Descent result
ax = nexttile;
imagesc(u_grad);
title('Gradient Descent ROF');
colorbar; colormap gray;
linked_axes = [linked_axes, ax];

% Link axes
linkaxes(linked_axes);

% Example parameter ranges
lambda_vals = linspace(0.01, 0.2, 25);
epsilon_vals = linspace(1e-3, 0.05, 25);

% --- Compute MSD surface plots for each Bayer color plane ---

% Reconstruct Bayer color planes individually
% Red: Top-left pixels
R = Iplanar(:,:,1);
% Green 1: Top-right
G1 = Iplanar(:,:,2);
% Green 2: Bottom-left
G2 = Iplanar(:,:,3);
% Blue: Bottom-right
B = Iplanar(:,:,4);

% Assemble into 3D array: HxWx4
color_planes = cat(3, R, G1, G2, B);


% Define parameter range
lambda_vals = linspace(0.01, 0.2, 25);
epsilon_vals = linspace(1e-3, 0.05, 25);
[Lambda, Epsilon] = meshgrid(lambda_vals, epsilon_vals);

% Create 3D surface plot for MSD(f, , )
figure;
hold on;

colors = {'r', [0 0.6 0], [0 0.9 0], 'b'};  % distinct red, green1, green2, blue
labels = {'Red', 'Green 1', 'Green 2', 'Blue'};

for i = 1:4
    f_plane = double(color_planes(:,:,i));  % ensure it's in double precision
    msd_plane = calculate_msd(f_plane, lambda_vals, epsilon_vals);  % MSD: (KxL)
    
    % Plot surface
    surf(Lambda, Epsilon, msd_plane', ...
        'EdgeColor', 'none', ...
        'FaceAlpha', 0.5, ...
        'FaceColor', colors{i}, ...
        'DisplayName', labels{i});
end

xlabel('\lambda');
ylabel('\epsilon');
zlabel('MSD');
legend show;
title('MSD vs (\lambda, \epsilon) for each Bayer color plane');
grid on;
view(45, 30);


