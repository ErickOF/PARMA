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

function [fig]=MyFigure()
  mon=get(0,'MonitorPositions');
  x=(rand*0.1+0.1)*mon(3);
  y=(rand*0.1+0.2)*mon(4);
  fig=figure('units','pixel','position',[x y 600 400]);
end

function []=MyImshow(fig,jmg,fn,job)
  jmg=jmg(2:end,2:end,:);% remove boundary
  figure(fig); imshow(uint8(jmg));
  set(gca,'position',[0 0 1 1]);
  set(gcf,'name',fn); drawnow;
end

function [img, jmg]=myImResize(img, jmg)
  [v, u] = size(jmg);
  if u > v,% landscape
    k = [img.h img.w];
  else% portrait
    k = [img.w img.h];
  end;
  jmg = imresize(jmg, k, 'bilinear');% resize
  [img.V, img.U] = size(jmg);% image size 
  img.N = img.V*img.U;
end

function [ENH,ovr]=MyRestore(ENH,GRY,img)
  z0=find(ENH<0);
  ENH(z0)=GRY(z0);
  
  z1=find(ENH>1);
  ENH(z1)=GRY(z1);
  
  ovr=length(z0)+length(z1);
  ovr=ovr/img.N;
end

function [ENH,ent,ovr]=MyGolden(k,GRY,MKF,img)
  TAH1=0.5*(1+tanh(3-12*abs(GRY-0.5)));
  TAH2=0.5*(1+tanh(3-(6*abs(MKF)-0.5)));
  TAH=TAH1.*TAH2;
  ENH=GRY+k*TAH.*MKF;
  [ENH,ovr]=MyRestore(ENH,GRY,img);
  ent=entropy(ENH(2:end-1,2:end-1))*(1-ovr);
end

function [kmg, ovr, k]=AUSM(img, GRY)
  H = [-1 -1 -1; -1 8 -1; -1 -1 -1];
  
  MKF_ = imfilter(GRY, H, 'same');
  MKF = zeros(size(MKF_));
  MKF(2:end-1, 2:end-1) = MKF_(2:end-1, 2:end-1);

  kL = 0;
  kH = 2;
  rng = kH - kL;
  rho = 0.5*(sqrt(5) - 1);
  tol = 0.01;
  
  k(1) = kL + (1-rho)*rng;
  k(2) = kL + rho*rng;
  [ENH(:,:,1), ent(1), ovr(1)] = MyGolden(k(1), GRY, MKF, img);
  [ENH(:,:,2), ent(2), ovr(2)] = MyGolden(k(2), GRY, MKF, img);
  while rng > tol,
    k_ = k;
    ovr_ = ovr;
    if ent(1) > ent(2),
      kH = k(2);
      rng = kH - kL;
      k(1) = kL + (1-rho)*rng;
      k(2) = kL + rho*rng;
      ent(2) = ent(1);
      [ENH, ent(1), ovr(1)] = MyGolden(k(1), GRY, MKF, img);
    else
      kL = k(1);
      rng = kH - kL;
      k(1) = kL + (1-rho)*rng;
      k(2) = kL + rho*rng;
      ent(1) = ent(2);
      [ENH, ent(2), ovr(2)] = MyGolden(k(2), GRY, MKF, img);
    end;
  end;
  kmg=ENH*img.L;
end;


% Main
[fn, pn, bn] = uigetfile('C:\\Users\\ErickOF\\Google Drive\\PARMA\\Datasets\\B2\\*.tif');

final = [];
MAX_PIXEL_VALUE_TIFF = 0;

if bn > 0,
  img.TIFF = double(imread([pn fn]));
  MAX_PIXEL_VALUE_TIFF = max(img.TIFF(:)(:))
  [img, img.GRAY] = myImResize(img, img.L*img.TIFF/MAX_PIXEL_VALUE_TIFF);
  fig(end + 1) = MyFigure();
  MyImshow(fig(end), img.GRAY, fn, job);
  
  job = job + 1;
  
  [filteringImg, ovr, k] = AUSM(img, img.GRAY);
  fig(end+1)=MyFigure();
  MyImshow(fig(end),img.L*filteringImg/MAX_PIXEL_VALUE_TIFF, fn, job);
end;