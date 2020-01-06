function hist = get_imagehist(dictionary, imsize, frames,descrs, options, normoption)

% descrs = normc(double(descrs));

width=imsize(2);
height=imsize(1);

numWords = options.num_bins ;

% quantize appearance
binsa = double(vl_kdtreequery(dictionary.kdtree, double(dictionary.vocab), ...
                              descrs, ...
                              'MaxComparisons', 15)) ;
hists= cell(1,length(options.numSpatialX));
for i = 1:length(options.numSpatialX)
  binsx = vl_binsearch(linspace(1,width+1,options.numSpatialX(i)+1), frames(1,:)) ;
  binsy = vl_binsearch(linspace(1,height+1,options.numSpatialY(i)+1), frames(2,:)) ;

  % combined quantization
  bins = sub2ind([options.numSpatialY(i), options.numSpatialX(i), numWords], ...
                 binsy,binsx,binsa) ;
  hist = zeros(options.numSpatialY(i) * options.numSpatialX(i) * numWords, 1) ;
  hist = vl_binsum(hist, ones(size(bins)), bins) ;
  if strcmp(normoption, 'nonorm')
      hists{i} = single(hist) ;
  else
      hists{i} = single(hist / sum(hist)) ;
  end
  %
end
hist = cat(1,hists{:}) ;
  if strcmp(normoption, 'nonorm') == 0 
      hist = hist / sum(hist) ;
  end
% 
% hist=normc(hist);