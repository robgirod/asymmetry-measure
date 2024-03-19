%% Export aligned volume

file = "C:\Users\rgirod\OneDrive - Universiteit Antwerpen\Data\Results\2023_Chiral NPs\2023_12_SCvsPT\FromMisha_LCys\PT_NP_LCys3 (pixelsize 0.425 nm).rec";
pixel_size = 0.425;

rec = read_rec(file); %Will open the GUI if no file

bin_factor = 4; %1 for None

rec = imresize3(rec, bin_factor); %Optional: reduce size
pixel_size = pixel_size / bin_factor;

rec_aligned = align_z_pca(rec); %rotates the particle to have major axis along Z

name = replace(file, '.rec', '_aligned.rec');
write_rec(rec_aligned, name, pixel_size);

%% Export rotational slices

% Optional: load a prealigned volume
file = "C:\Users\rgirod\OneDrive - Universiteit Antwerpen\Data\Results\2023_Chiral NPs\2023_12_SCvsPT\From_Misha_Raw\Wrinkled PT NPs\P1_tiltSeries_alignedFull_conSIRT_05_aligned.rec";
pixel_size = 0.52;

rec_aligned = read_rec(file);

[nx, ny, nz] = size(rec_aligned);

slices = zeros([nx, nz, 10]);
i = 1;
nSlice = floor(nx/2);
%nSlice = 77;
nEvery = 1;

h = waitbar(0, 'Processing...');
for theta = 0:nEvery:360-nEvery    
    rec_rot = imrotate3(rec_aligned, theta, [0 0 1], "linear", "crop");
    slices(:,:,i) = squeeze(rec_rot(nSlice,:,:));
    i = i + 1;

    waitbar(i / size(0:nEvery:180, 2), h)
end
close(h);

slices = imrotate3(slices, -90, [0 0 1], "linear", "loose");

name = replace(file, '.rec', sprintf('_rotSlicesMore_atSlice%2.0f.rec', nSlice));
write_rec(slices, name, pixel_size);
