function hist = get_imagehist_FPGA(dictionary, imsize, frames,descrs, options)

% descrs = normc(double(descrs));

width=imsize(2);
height=imsize(1);

numWords = options.num_bins ;
binsa = zeros(1,1); 
% quantize appearance
% binsa = double(vl_kdtreequery(dictionary.kdtree, double(dictionary.vocab), ...
%                               descrs, ...
%                               'MaxComparisons', 0)) ;

parfor i = 1: size(descrs,2) %use 'for' if you don't have the toolbox
   [binsa(i),~] = kd_closestpointfast(dictionary.kdtree,descrs(:,i));
end

hists= cell(1,length(options.numSpatialX));
for i = 1:length(options.numSpatialX)
  binsx = vl_binsearch(linspace(1,width+1,options.numSpatialX(i)+1), frames(1,:)) ;
  binsy = vl_binsearch(linspace(1,height+1,options.numSpatialY(i)+1), frames(2,:)) ;

  % combined quantization
  bins = sub2ind([options.numSpatialY(i), options.numSpatialX(i), numWords], ...
                 binsy,binsx,binsa) ;
  hist = zeros(options.numSpatialY(i) * options.numSpatialX(i) * numWords, 1) ;
  hist = vl_binsum(hist, ones(size(bins)), bins) ;
  hists{i} = hist;
  %single(hist / sum(hist)) ;
end
hist = cat(1,hists{:}) ;
% hist = hist / sum(hist) ;
% hist=normc(hist);