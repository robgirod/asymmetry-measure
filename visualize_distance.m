%%
vol = rg_extract_surface("test\img.tif", 50000);
vol_m = rg_extract_surface("test\img_mirror.tif", 50000);

%%
tree = KDTreeSearcher(vol_m.Vertices);
[idxKDT, D] = knnsearch(tree, vol.Vertices);

%%
faces = vol.Faces;
verts = vol.Vertices;

% Plot surface
figure;
set(gcf, 'Color', 'w');
trisurf(faces,verts(:,1),verts(:,2),verts(:,3), D, 'EdgeColor', 'none');
daspect([1 1 1]);
view(0,0);
axis off;

% Color bar
colormap('sky');
cb = colorbar;
fontsize(12, 'points');
cb.Position(3:4) = [0.03 0.6];  % [x, y, width, height]
