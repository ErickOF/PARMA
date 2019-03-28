clc;
clear all;
close all;
dbstop if error;
pkg load image;


% Image dimensions
img.h = 360;
img.w = 480;
img.L = 255;
img.L_ = linspace(0, img.L, img.L+1);

job = 0;
fig = [];


function hsi = rgb2hsi(rgb)
  %RGB2HSI Converts an RGB image to HSI.
  % Extract the individual component images.
  rgb = im2double(rgb);
  r = rgb(:, :, 1);
  g = rgb(:, :, 2);
  b = rgb(:, :, 3);
  % Implement the conversion equations
  num = 0.5*((r - g) + (r - b));
  den = sqrt((r - g).^2 + (r - b).*(g - b));
  theta = acos(num./(den + eps));
  H = theta;
  H(b > g) = 2*pi - H(b > g);
  H /= 2*pi;
  num = min(min(r, g), b);
  den = r + g + b;
  den(den == 0) = eps;
  S = 1 - 3.*num./den;
  H(S == 0) = 0;
  I = (r + g + b)/3;
  % Combine all three results into an hsi image.
  hsi = cat(3, H, S, I);
end


function rgb = hsi2rgb(hsi)
  % HSI2RGB Converts an HSI image to RGB.
  % Extract the individual HSI component images.
  H = hsi(:, :, 1) * 2 * pi;
  S = hsi(:, :, 2);
  I = hsi(:, :, 3);
  % Implement the conversion equations.
  R = zeros(size(hsi, 1), size(hsi, 2));
  G = zeros(size(hsi, 1), size(hsi, 2));
  B = zeros(size(hsi, 1), size(hsi, 2));
  % RG sector (0 <= H < 2*pi/3).
  idx = find((0 <= H) & (H < 2*pi/3));
  B(idx) = I(idx) .* (1 - S(idx));
  R(idx) = I(idx) .* (1 + S(idx) .* cos(H(idx)) ./ cos(pi/3 - H(idx)));
  G(idx) = 3*I(idx) - (R(idx) + B(idx));
  % BG sector (2*pi/3 <= H < 4*pi/3).
  idx = find((2*pi/3 <= H) & (H < 4*pi/3));
  R(idx) = I(idx) .* (1 - S(idx));
  G(idx) = I(idx) .* (1 + S(idx) .* cos(H(idx) - 2*pi/3) ./ cos(pi - H(idx)));
  B(idx) = 3*I(idx) - (R(idx) + G(idx));
  % BR sector.
  idx = find((4*pi/3 <= H) & (H <= 2*pi));
  G(idx) = I(idx) .* (1 - S(idx));
  B(idx) = I(idx) .* (1 + S(idx) .* cos(H(idx) - 4*pi/3) ./ cos(5*pi/3 - H(idx)));
  R(idx) = 3*I(idx) - (G(idx) + B(idx));
  % Combine all three results into an RGB image.  Clip to [0, 1] to
  % compensate for floating-point arithmetic rounding effects.
  rgb = cat(3, R, G, B);
  rgb = max(min(rgb, 1), 0);
end


function [RGB] = stretch(img, RGB)
  % Normalize image
  RGB = double(RGB)/img.L;
  for k=1:size(RGB, 3),
    min_value = min(min(RGB(:, :, k)));
    max_value = max(max(RGB(:, :, k)));
    RGB(:, :, k) -= min_value;
    RGB(:, :, k) /= max_value;
  end;
  RGB = uint8(RGB*img.L);
end


function [huv, ovr] = restore(huv, guv, img)
  z0 = find(huv < 0);
  z1 = find(huv > 1);
  
  huv(z0) = guv(z0);
  huv(z1) = guv(z1);
  
  ovr = (length(z0) + length(z1))/img.N;
end


function [huv, huv_entropy, over_range_pixeles] = golden(k, guv, duv, img)
  lambda_guv = 0.5*(1 + tanh(3 - 12*abs(guv - 0.5)));
  lambda_duv = 0.5*(1 + tanh(3 - (6*abs(duv) - 0.5)));
  lambda_uv = lambda_guv.*lambda_duv;
  huv = guv + k*lambda_uv.*duv;
  [huv, over_range_pixeles] = restore(huv, guv, img);
  huv_entropy = entropy(huv(2:end-1, 2:end-1))*(1 - over_range_pixeles);
end

#{
img - struct with the image information (h:height, w:width, RGB:resized rgb img, JPG:read jgp image, N:hxw)
jmg - rgb image
K - center value of the filter
kMin - Minimun gain
kMax - Maximun gain
tol - Solution tolerance
#}
function [kmg, ovr, k] = AUSM_GRAY(img, jmg, K=8, kMin=0, kMax=2, tol=0.01)
  H = 0.125*[-1 -1 -1; -1 K -1; -1 -1 -1];
  jmg = stretch(img, jmg);
  
  HSI = rgb2hsi(jmg);
  guv = HSI(:, :, 3);
  
  filteredImage = imfilter(guv, H, 'same');
  duv = zeros(size(filteredImage));
  duv(2:end-1, 2:end-1) = filteredImage(2:end-1, 2:end-1);
  
  rng = kMax - kMin;
  gsr = 0.5*(sqrt(5) - 1);
  
  k(1) = kMin + (1 - gsr)*rng;
  k(2) = kMin + gsr*rng;

  [ENH(:, :, 1), ent(1), ovr(1)] = golden(k(1), guv, duv, img);
  [ENH(:, :, 2), ent(2), ovr(2)] = golden(k(2), guv, duv, img);
  
  while rng > tol,
    k_ = k;
    ovr_ = ovr;
    if ent(1) > ent(2),
      kMax = k(2);
      rng = kMax - kMin;
      k(1) = kMin + (1-gsr)*rng;
      k(2) = kMin + gsr*rng;
      ent(2) = ent(1);
      [ENH, ent(1), ovr(1)] = golden(k(1), guv, duv, img);
    else
      kMin = k(1);
      rng = kMax - kMin;
      k(1) = kMin + (1-gsr)*rng;
      k(2) = kMin + gsr*rng;
      ent(1) = ent(2);
      [ENH, ent(2), ovr(2)] = golden(k(2), guv, duv, img);
    end;
  end;
  ovr = mean(ovr_);
  k = mean(k_);
  HSI(:, :, 3) = ENH;
  kmg = hsi2rgb(HSI);
  kmg = uint8(kmg*img.L);
end


function [fig] = getFig()
  mon = get(0, 'MonitorPositions');
  x = (rand*0.1 + 0.1)*mon(3);
  y = (rand*0.1 + 0.2)*mon(4);
  fig = figure('units', 'pixel', 'position', [x y 600 400]);
end


function [] = plotImg(fig, jmg, filename, job)
  % remove boundary
  jmg = jmg(2:end, 2:end, :);
  figure(fig);
  imshow(uint8(jmg));
  set(gca, 'position', [0 0 1 1]);
  set(gcf, 'name', filename);
  drawnow;
end


function [img, jmg] = resizeImg(img, jmg)
  [v, u, w] = size(jmg);
  if u > v,
    k = [img.h img.w];
  else
    k = [img.w img.h];
  end;
  % resize
  jmg = imresize(jmg, k, 'bilinear');
  % image size
  [img.V, img.U, img.N] = size(jmg); 
  img.N = img.V*img.U;
end


% Main
[filename, path, index] = uigetfile('C:\\Users\\ErickOF\\Google Drive\\PARMA\\Datasets\\B2\\*.tif');

if index > 0,
  loadedImg = double(imread([path filename]));
  loadedImg *= img.L/max(loadedImg(:)(:)(:));
  img.JPG = cat(3, loadedImg, loadedImg, loadedImg);
  
  [img, img.RGB] = resizeImg(img, img.JPG);
  fig(end + 1) = getFig();
  plotImg(fig(end), img.RGB, filename, job);
  
  for K=3:2:17
    for kMax=0:9
      for kMin=0:kMax-1
        for tol=0.001:0.005:0.1
          job += 1;
          [ausmImg, ovr, k] = AUSM_GRAY(img, img.RGB, K, kMin, kMax, tol);
          f = strsplit(filename, '.')(1);
          name = strcat(f, '_K=', num2str(K), '_kMin=', num2str(kMin), '_kMax=',
                        num2str(kMax), '_tol=', num2str(int8(tol)), '.png')
          imwrite(ausmImg, strcat(num2str(job), '.png'));
          #plotImg(fig(end), ausmImg, filename, job);
        end
      end
    end
  end
end;
