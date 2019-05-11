pkg load image;

% Dataset folder
in_dir = '.\..\save\gt\';
% Saving folder
out_dir = '.\..\save\dt\';
dataset = dir(strcat(in_dir,'*.png'));

parfor k = 1:length(dataset)
    img_name = dataset(k).name;
    % Read the image
    img = imread(strcat(in_dir, img_name));
    % New image to save
    new_img = zeros(size(img));
    labels = unique(img);
    % Iterate over each cell
    for idx = 2:length(labels)
        cell = (img == labels(idx));
        dist_trans = bwdist(~cell);
        cell_weighted = dist_trans / max(max(dist_trans));
        cell_indexes = find(cell_weighted ~= 0);
        for idx2 = 1:length(cell_indexes)
           cell_weighted(cell_indexes(idx2)) = 1 - cell_weighted(cell_indexes(idx2));
        end
        new_img = new_img + cell_weighted;
    end
    imwrite(new_img,strcat(out_dir, img_name));
    disp(strcat('Ready: ', img_name));
    printf("Listo\n");
end