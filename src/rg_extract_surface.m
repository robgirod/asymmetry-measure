function mesh = rg_extract_surface(fname, n_faces, n_smooth, show_surface, write_surface)
% Reads an .stl file and return the faces and vertices or reads an .rec and
% extract the surface, smoothes it and simplified. The surface will be
% extracted at Otsu's thershold
%
% Parameters:
%   - fname (str):              file name
%   - n_faces (int):            number of faces to keep during the simplification. Not super
%   accurate
%   - n_smooth (int):           parameter forwarded to 'smoothSurfaceMesh'
%   - show_surface (Bool):      will show the surface in a new Figure
%   - write_surface (Bool):     if True, will write a new '_Matlab.stl' file in
%   the folder of the input file

if nargin < 5, write_surface = 0; end
if nargin < 4, show_surface = 0; end
if nargin < 3, n_smooth = 1; end
if nargin < 2, n_faces = 400000; end


if endsWith(char(fname), '.stl')
    fprintf('Loading %s...\n', fname);
    data = stlread(fname);
    verts = data.Points; 
    faces = data.ConnectivityList;
    mesh = surfaceMesh(verts, faces);
    %mesh = triangulation(faces, verts); % Uncomment if the Lidar toolbox is not
    %available

else
    if endsWith(char(fname), '.rec')
        fprintf('Loading and computing surface from %s...\n', fname);
    
        % Read
        [vol, pix_size] = read_rec(fname);
        vol = rescale(single(vol));

    elseif endsWith(char(fname), '.tif')
        fprintf('Loading and computing surface from %s...\n', fname);
    
        % Read
        vol = tiffreadVolume(fname);
        pix_size = 1;
        vol = rescale(single(vol));
    else
        vol = fname;
    end
    
    % Get iso value and keep largest component TO-DO: make/use GUI
    otsu = graythresh(vol);
    vol_bin = imbinarize(vol, otsu);
    
    s = regionprops3(vol_bin, "BoundingBox", "Volume");
    volumes = cat(1, s.Volume);
    boxes = cat(1, s.BoundingBox);
    
    [~, i_position] = max(volumes);
    cuboid = boxes(i_position, :);

    try
        cuboid(4:6) = cuboid(4:6) + 6;
        cuboid(1:3) = cuboid(1:3) - 3;
        vol_clean = imcrop3(vol, cuboid);
    catch
        warning('cuboid might be larger than image, keeping the original dims');
        cuboid = boxes(i_position, :);
        disp(cuboid);
        vol_clean = imcrop3(vol, cuboid);
    end
    
   
    
    % otsu = graythresh(vol);
    % 
    % cc = bwconncomp(vol_bin);
    % [~, idx_biggest_cc] = max(cellfun(@numel, cc.PixelIdxList));
    % 
    % vol_clean = zeros(size(vol));
    % vol_clean(cc.PixelIdxList{idx_biggest_cc}) = 1;
    
    % Get surface
    [faces, verts] = isosurface(vol_clean, graythresh(vol_clean));
    mesh = surfaceMesh(verts * pix_size, faces);
    fprintf('%u faces obtained from isosurface mapping...\n', size(faces,1));

    % Simplify and clean
    disp("Smoothing...");
    mesh = smoothSurfaceMesh(mesh, n_smooth, "Method","Average");
    %mesh = smoothSurfaceMesh(mesh, n_smooth, "Method","Laplacian", "ScaleFactor", 0.9);

    if size(mesh.Faces, 1) > n_faces
        disp("Simplifying...");
        simplify(mesh, "TargetNumFaces", n_faces, "MaxError", 1, "BoundaryWeight", 10);

        disp("Cleaning the mesh...")    
        removeDefects(mesh,"nonmanifold-edges");
        removeDefects(mesh,"unreferenced-vertices");

        disp("Smoothing...");
        %mesh = smoothSurfaceMesh(mesh, n_smooth, "Method","Average");
        %mesh = smoothSurfaceMesh(mesh, n_smooth, "Method","Laplacian", "ScaleFactor", 0.9);
        %mesh = smoothSurfaceMesh(mesh, n_smooth, "Method","Laplacian", "ScaleFactor", 0.9);
        mesh = smoothSurfaceMesh(mesh, n_smooth, "Method","Average");

        fprintf('%u faces after simplification...\n', size(mesh.Faces,1));
    end

    if write_surface
        writeSurfaceMesh(mesh, replace(fname, '.rec', '_matlab.stl'));
        fprintf('saved %s\n', replace(fname, '.rec', '_matlab.stl'));
    end

end

if show_surface, surfaceMeshShow(mesh); end



