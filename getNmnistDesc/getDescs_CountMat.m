function [frames, descs] = getDescs_CountMat(Count_Mat, cur_loc, param)

    rmin = param.rmin;
    rmax = param.rmax;
    nr = param.nr;
    nw = param.nw;
    
    r_bin_edges = logspace(log10(rmin),log10(rmax),nr);
    
    [frames, descs] = getDescriptors_CountMat(Count_Mat, r_bin_edges, nr, nw, cur_loc);
    
    
end

function [frame, desc] = getDescriptors_CountMat(Count_Mat, r_bin_edges, nr, nw, cur_loc)
    logHistEmpty = zeros(nw, nr);
    [currSpikeArr.y,currSpikeArr.x] = find(Count_Mat);
    
    [currSpikeArr.rad, currSpikeArr.theta] = getLPCoord(currSpikeArr.x, currSpikeArr.y, cur_loc);
    frame = [cur_loc.x cur_loc.y]';
    currSpikeArr.wedgeNum = ceil(nw* currSpikeArr.theta / (2*pi)) + nw/2;
    ringNum = bsxfun(@lt, r_bin_edges(1:end),currSpikeArr.rad);
    currSpikeArr.ringNum = sum(ringNum, 2) + 1;
    discardedIndices = find(currSpikeArr.ringNum > nr);
    currSpikeArr.ringNum(discardedIndices) = [];
    currSpikeArr.wedgeNum(discardedIndices) = [];
    currSpikeArr.x(discardedIndices) = [];
    currSpikeArr.y(discardedIndices) = [];
    
    not_discarded_locs = sub2ind(size(Count_Mat), currSpikeArr.y, currSpikeArr.x);
%     weights = abs(1-exp(1./abs(TD.ts(first_saccade_pt:i) - TD.ts(i))));
%     weights(discardedIndices) = [];
%     weights(end) = 0;
%     logHist = accumarray([currSpikeArr.wedgeNum(:), currSpikeArr.ringNum(:)], weights);
    logHist = accumarray([currSpikeArr.wedgeNum(:), currSpikeArr.ringNum(:)], Count_Mat(not_discarded_locs));
    
    logHist = logHist/sum(logHist(:));
    
    logHistEmpty(1:size(logHist,1), 1:size(logHist, 2)) = logHistEmpty(1:size(logHist,1), 1:size(logHist, 2)) + logHist;
    desc = logHistEmpty(:);
end

function [rad, theta] =  getLPCoord(xArr, yArr, cur_loc)
    xc = cur_loc.x;
    yc = cur_loc.y;
    rad = sqrt( (xArr - xc).^2 + (yArr - yc).^2 );
    theta = atan2(yArr - yc, xArr - xc);
end
