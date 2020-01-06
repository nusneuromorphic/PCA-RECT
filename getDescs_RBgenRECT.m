function [frames, descs] = getDescs_RBgenRECT(TD, param)

    TD.x = TD.x(:);
    TD.y = TD.y(:);
    TD.ts = TD.ts(:);
    TD.p = TD.p(:); 
    
        rmax = param.rmax;
    binres = param.binres;
    minNumEvents = param.minNumEvents;
    
    [X, Y] = meshgrid(1:2*rmax+1,1:2*rmax+1);
    template.X = ceil(X/binres);
    template.Y = ceil(Y/binres);
    
    nlen = max(template.X(:));
    nwidth = max(template.Y(:));

    maxNumEvents = param.maxNumEvents; 
    if maxNumEvents > numel(TD.x)
        maxNumEvents = ceil(numel(TD.x));
    end
    last_event = numel(TD.x); 
    
        %writeFile('TD.txt', TD);
    descs = uint16(zeros(nlen*nwidth, 1));
    frames = uint8(zeros(2, 1));
    
    tic
    %writeFile('TD.txt', TD);

    count = 1; 
    first_saccade_pt = count;
    last_saccade_pt = maxNumEvents-minNumEvents;
    flash_count = 1;
    while first_saccade_pt < last_event
        parfor i=first_saccade_pt-(flash_count-1)*minNumEvents:last_saccade_pt
            [frame, desc] = getDescriptors(TD, template, first_saccade_pt, i + flash_count*minNumEvents);
            frames(:,i) = (frame);     
            descs(:,i) = (desc);
        end
        flash_count = flash_count + 1;

        first_saccade_pt = first_saccade_pt+maxNumEvents;
        last_saccade_pt = first_saccade_pt +maxNumEvents - flash_count * minNumEvents - 1;
        if last_saccade_pt > last_event
            last_saccade_pt = last_event;
        end
        if last_saccade_pt + flash_count*minNumEvents > last_event
                break
        end
    end
    toc
end

function [frame, desc] = getDescriptors(TD, template, first_saccade_pt, i)
    logHistEmpty = zeros(max(template.Y(:)), max(template.X(:)));
    
    currSpikeArr.x = TD.x(first_saccade_pt:i);
    currSpikeArr.y = TD.y(first_saccade_pt:i);
    
    template_center = ceil(size(template.X)./2);
    max_displacement = max(size(template.X)) - template_center;
    
    [currSpikeArr.diffX, currSpikeArr.diffY, frame] = getRECTCoord(currSpikeArr.x, currSpikeArr.y);
    discardedIndices = (abs(currSpikeArr.diffX) > max_displacement(1) | abs(currSpikeArr.diffY) > max_displacement(2));
   
    currSpikeArr.diffX(discardedIndices) = [];
    currSpikeArr.diffY(discardedIndices) = [];
    currSpikeArr.y(discardedIndices) = [];
    currSpikeArr.x(discardedIndices) = [];
    
    currSpikeArr.templatediffX = currSpikeArr.diffX + template_center(1);
    currSpikeArr.templatediffY = currSpikeArr.diffY + template_center(1);
    
    currSpikeArr.indices = sub2ind(size(template.X), currSpikeArr.templatediffY, currSpikeArr.templatediffX);
    
    currSpikeArr.binX = template.X(currSpikeArr.indices);
    currSpikeArr.binY = template.Y(currSpikeArr.indices);
    
    
    logHist = accumarray([currSpikeArr.binY(:), currSpikeArr.binX(:)], 1);
    %logHist = logHist/sum(logHist(:));
    logHistEmpty(1:size(logHist,1), 1:size(logHist, 2)) = logHistEmpty(1:size(logHist,1), 1:size(logHist, 2)) + logHist;
    desc = logHistEmpty(:);
end

function [diffX, diffY, frame] =  getRECTCoord(xArr, yArr)
    xc = xArr(end);
    yc = yArr(end);
    diffX = (xArr - xc);
    diffY = (yArr - yc);
    frame = [xc yc]';
end