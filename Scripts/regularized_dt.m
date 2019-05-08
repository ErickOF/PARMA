
in_dir = '.\GT_BBBC\';             % Dataset folder
out_dir = '.\DT\';      % Saving folder
dataset = dir(strcat(in_dir,'*.png'));

parfor k = 1:length(dataset)
    img_name = dataset(k).name;
    img = imread(strcat(in_dir, img_name));    % Read the image
    new_img = zeros(size(img));                     % New image to save
    labels = unique(img);
    for idx = 2:length(labels)                      % Iterate over each cell
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
end