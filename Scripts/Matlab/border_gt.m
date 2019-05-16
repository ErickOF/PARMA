function Border_of_GT()
    in_dir = 'C:\Users\ErickOF\Documents\Git\PARMA\Scripts\save\gt\';
    out_dir = 'C:\Users\ErickOF\Documents\Git\PARMA\Scripts\save\border\';
    dataset = dir(strcat(in_dir,'*.png'));

    for k = 1:length(dataset)
        % open image
        img_name = dataset(k).name;
        img = imread(strcat(in_dir, img_name));

        % each layer of the new image
        background = zeros(size(img));
        interior = zeros(size(img));
        touching = zeros(size(img));
        
        % every background pixel 
        background(img == 0) = 1;
        
        % border and foreground
        kernel_size = 5;
        [r_cells, c_cells] = find(img ~=0);  % every pixel that is not background
        for idx = 1:length(r_cells)
            n = neighbors(img, r_cells(idx), c_cells(idx), kernel_size); % window
            if (find(n ~= img(r_cells(idx), c_cells(idx)))) % when the window has different pixels
                touching(r_cells(idx), c_cells(idx)) = 1;
            else
                interior(r_cells(idx), c_cells(idx)) = 1;   %when every pixel is the same
            end
        end
        new_img = cat(3,background,interior, touching); %build and save new image
        imwrite(new_img,strcat(out_dir, img_name));
        disp(strcat('Ready: ', img_name));
    end
end

% Creates a window to see pixels around
function n = neighbors(img, r, c, kernel_size)
    n = zeros(kernel_size);
    offset = floor(kernel_size/2);
    offset_x = -offset;
    offset_y = -offset;
    
    height = size(img,1);
    width = size(img,2);
    
    for i = 1:kernel_size
        for j = 1:kernel_size
            new_r = r+offset_x;
            new_c = c+offset_y;
            if (new_r >= 1 && new_r <=height && new_c >= 1 && new_c <=width)
                n(i, j) = img(new_r, new_c);
            else
                n(i, j) = 0;
            end
            offset_y = offset_y + 1;
        end
        offset_x = offset_x + 1;
        offset_y = -offset;
    end    
end
    