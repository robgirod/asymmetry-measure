function aligned_reconstruction = align_z_pca(reconstruction, plot)
% Align a particle with clear orientation along the z (slice) axis.
%
% Parameters:
%   reconstruction (3D Array):
%     Object containing input images and corresponding tilt angles.
%   plot (Bool):
%     Default is false. True to plot a scatter3 of the edges with principal
%     components.
%
% Returns:
%   aligned_tilt_series (TiltSeries)

if nargin < 2, plot = false; end
aligned_reconstruction = reconstruction;

%% Find and correct for center of mass
im = rescale(reconstruction);
otsu = graythresh(im);
im_bin = imbinarize(im, otsu);
weighted_center_of_mass = regionprops3(im_bin, im, "WeightedCentroid");
center_of_volume = size(im)/2;

transl = center_of_volume - weighted_center_of_mass.WeightedCentroid(1,:);

%% Find edges
im = rescale(reconstruction);
im_bin = imresize3(im, 0.25);
otsu = graythresh(im_bin);
im_bin = imbinarize(im_bin, otsu);
im_edges = edge3(im_bin, 'sobel', otsu);

%% Reshape into an xy plot

%return the rows and the linear idx in trailing dim, from the top left frontmost corner
[row, idx] = find(im_edges);
sz = size(im_edges);
[x, z] = ind2sub(sz(2:end), idx);
y = row; %should mabye invert, but then things become weird

% center for pca
x = x-mean(x);
y = y-mean(y);
z = z-mean(z);
data = [x y z];

%% PCA
fprintf('Finding the axes with PCA ... ')
[coeff, score, latent] = pca(data, 'NumComponents', 3);
fprintf('Done\n')


if plot
    scatter3(x, y, z);
    axis([-150 150 -150 150]);
    daspect([1 1 1])
    hold on;
    quiver3(0, 0, 0, coeff(1,1) * latent(1) / 10, coeff(2,1) * latent(1) / 10, coeff(3,1) * latent(1) / 10);
    quiver3(0, 0, 0, coeff(1,2) * latent(1) / 10, coeff(2,2) * latent(1) / 10, coeff(3,2) * latent(1) / 10);
    quiver3(0, 0, 0, coeff(1,3) * latent(1) / 10, coeff(2,3) * latent(1) / 10, coeff(3,3) * latent(1) / 10);
end



%% Transformation

% %% Angle 2D
% theta = rad2deg(atan(coeff(2,1)/coeff(1,1)));
% im_rot = imrotate(im, -theta, "crop");
% imshow(im_rot);

fprintf('Rotating ... ')

R = circshift(coeff', 2, 1); %roll axis 1 (row) to position PC1 on row 3 and align PC1 to z

if single(det(R)) == -1.0, R = [R(2,:); R(1,:); R(3,:)]; end % if det(R) is -1 instead of 1, need to invert rows of PC2 and PC3 

tform = rigidtform3d(R,transl);
CenterOutput = affineOutputView(size(im),tform,"BoundsStyle","CenterOutput"); % we assume there is enough black around the particle to allow for croping after the rotation

aligned_reconstruction = imwarp(reconstruction, tform, "cubic", "OutputView", CenterOutput);

fprintf('Done\n')

