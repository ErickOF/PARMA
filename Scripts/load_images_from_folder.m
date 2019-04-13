imgs = [];

# Image folder
PATH = 'C:\Users\'
selpath = uigetdir(PATH);

FULLPATH = fullfile(selpath, '*.tif');
dirs = dir(FULLPATH);

for i = 1:numel(dirs)
    path_filename = fullfile(selpath, dirs(i).name);
    imgs(i) = double(imread([path_filename]));
end