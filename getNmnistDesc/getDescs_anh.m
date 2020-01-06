function [frames, descs] = getDescs(TD, param)

    rmin = param.rmin;
    rmax = param.rmax;
    nr = param.nr;
    nw = param.nw;
    
    r_bin_edges = logspace(log10(rmin),log10(rmax),nr);
    count = 0;

    %writeFile('TD.txt', TD);
    descs = zeros(nr*nw, 1);
    
    first_saccade_events = TD.ts <= 105e3;
    cumsum_first_saccade_events = cumsum(first_saccade_events);
    
    tic;
    first_sampling_pt = 200;
    first_saccade_pt = 1;
    last_sampling_pt = max(cumsum_first_saccade_events);
    for i=first_sampling_pt:last_sampling_pt
        logHistEmpty = zeros(nw, nr);
        currSpikeArr.x = TD.x(first_saccade_pt:i);
        currSpikeArr.y = TD.y(first_saccade_pt:i);
        
        [currSpikeArr.rad, currSpikeArr.theta, frame] = getLPCoord(currSpikeArr.x, currSpikeArr.y);
        currSpikeArr.wedgeNum = ceil(nw* currSpikeArr.theta / (2*pi)) + nw/2;
        ringNum = bsxfun(@lt, r_bin_edges(1:end-1),currSpikeArr.rad);
        currSpikeArr.ringNum = sum(ringNum, 2) + 1;
        %ringFreq = unique(currSpikeArr.ringNum);
        %out = [ringFreq,histc(currSpikeArr.ringNum(:),ringFreq)];
        logHist = accumarray([currSpikeArr.wedgeNum(:), currSpikeArr.ringNum(:)], 1);
        logHist = logHist/sum(logHist(:));
        %logHist = logHist(:);
        count = count + 1;
        frames(:,count) = frame;
        
        %descs(:,count) = 0;
        logHistEmpty(1:size(logHist,1), 1:size(logHist, 2)) = logHistEmpty(1:size(logHist,1), 1:size(logHist, 2)) + logHist;
        descs(:,count) = logHistEmpty(:);
        %descs(1:numel(logHist),count) = descs(1:numel(logHist),count) +logHist;
    end
    
    second_saccade_events = TD.ts > 105e3 & TD.ts <= 210e3;
    cumsum_second_saccade_events = cumsum(second_saccade_events) + i;
    first_sampling_pt = i+200;
    first_saccade_pt = i+1;
    last_sampling_pt = max(cumsum_second_saccade_events);
    for i=first_sampling_pt:last_sampling_pt
        logHistEmpty = zeros(nw, nr);
        currSpikeArr.x = TD.x(first_saccade_pt:i);
        currSpikeArr.y = TD.y(first_saccade_pt:i);
        
        [currSpikeArr.rad, currSpikeArr.theta, frame] = getLPCoord(currSpikeArr.x, currSpikeArr.y);
        currSpikeArr.wedgeNum = ceil(nw* currSpikeArr.theta / (2*pi)) + nw/2;
        ringNum = bsxfun(@lt, r_bin_edges(1:end-1),currSpikeArr.rad);
        currSpikeArr.ringNum = sum(ringNum, 2) + 1;
        %ringFreq = unique(currSpikeArr.ringNum);
        %out = [ringFreq,histc(currSpikeArr.ringNum(:),ringFreq)];
        logHist = accumarray([currSpikeArr.wedgeNum(:), currSpikeArr.ringNum(:)], 1);
        logHist = logHist/sum(logHist(:));
        %logHist = logHist(:);
        count = count + 1;
        frames(:,count) = frame;
        
        logHistEmpty(1:size(logHist,1), 1:size(logHist, 2)) = logHistEmpty(1:size(logHist,1), 1:size(logHist, 2)) + logHist;
        descs(:,count) = logHistEmpty(:);
%         descs(:,count) = 0;
%         descs(1:numel(logHist),count) = descs(1:numel(logHist),count) +logHist;
    end
    
    third_saccade_events = TD.ts > 210e3;
    cumsum_third_saccade_events = cumsum(third_saccade_events) + i;
    first_sampling_pt = i+200;
    first_saccade_pt = i+1;
    last_sampling_pt = max(cumsum_third_saccade_events);
    for i=first_sampling_pt:last_sampling_pt
        logHistEmpty = zeros(nw, nr);
        currSpikeArr.x = TD.x(first_saccade_pt:i);
        currSpikeArr.y = TD.y(first_saccade_pt:i);
        
        [currSpikeArr.rad, currSpikeArr.theta, frame] = getLPCoord(currSpikeArr.x, currSpikeArr.y);
        currSpikeArr.wedgeNum = ceil(nw* currSpikeArr.theta / (2*pi)) + nw/2;
        ringNum = bsxfun(@lt, r_bin_edges(1:end-1),currSpikeArr.rad);
        currSpikeArr.ringNum = sum(ringNum, 2) + 1;
        %ringFreq = unique(currSpikeArr.ringNum);
        %out = [ringFreq,histc(currSpikeArr.ringNum(:),ringFreq)];
        logHist = accumarray([currSpikeArr.wedgeNum(:), currSpikeArr.ringNum(:)], 1);
        logHist = logHist/sum(logHist(:));
        %logHist = logHist(:);
        count = count + 1;
        frames(:,count) = frame;
        logHistEmpty(1:size(logHist,1), 1:size(logHist, 2)) = logHistEmpty(1:size(logHist,1), 1:size(logHist, 2)) + logHist;
        descs(:,count) = logHistEmpty(:);
%         descs(:,count) = 0;
%         descs(1:numel(logHist),count) = descs(1:numel(logHist),count) +logHist;
    end
    
    toc;
end


function [rad, theta, frame] =  getLPCoord(xArr, yArr)
    xc = xArr(end);
    yc = yArr(end);
    rad = sqrt( (xArr - xc).^2 + (yArr - yc).^2 );
    theta = atan2(yArr - yc, xArr - xc);
    frame = [xc yc]';
end

function writeFile(fileName, var)
    writetable(struct2table(var), fileName);
end