function [frames, descs] = getDescs_RECT_FPGA(TD, param)
    
    subsamp_count_matrix = param.countmatsubsamp; 
    subsamp_desc_regionsize = floor(param.descsize/2); %radius of descsize by descsize region
    
    % subsample the TD x y locations and account for zero padding in the
    % count matrix for easy desc generation
    TD.x = ceil(TD.x(:)./subsamp_count_matrix) + subsamp_desc_regionsize;
    TD.y = ceil(TD.y(:)./subsamp_count_matrix) + subsamp_desc_regionsize;
    TD.ts = TD.ts(:);
    TD.p = TD.p(:); 
    
    % x and y locations have already accounted for left and top padding 
    count_matrix = zeros(max(TD.y) + subsamp_desc_regionsize, max(TD.x) + subsamp_desc_regionsize);

    global_eventcount = 0;
    desc_count = 0;
    
    queue = CQueue();
    queue.capacity = param.queuesize;
    minNumEvents = param.minNumEvents;
    
    descs = uint16(zeros(param.descsize*param.descsize, 1));
    frames = uint8(zeros(2, 1));

for i = 1:size(TD.x,1)
    
    global_eventcount = global_eventcount + 1;
    
    event_x = TD.x(i);
    event_y = TD.y(i);
    
    if global_eventcount <= minNumEvents
        queue.push([event_x event_y]);
        count_matrix(event_y, event_x) = count_matrix(event_y, event_x) + 1; 
    else
	    if global_eventcount > param.queuesize
             old_xy = queue.pop();
             count_matrix(old_xy(2), old_xy(1)) = count_matrix(old_xy(2), old_xy(1)) - 1; 
	     end
        queue.push([event_x event_y]);
        count_matrix(event_y, event_x) = count_matrix(event_y, event_x) + 1; 
        
        frame = [event_x; event_y];
        desc = count_matrix(event_y-subsamp_desc_regionsize:event_y + subsamp_desc_regionsize, ...
            event_x - subsamp_desc_regionsize:event_x + subsamp_desc_regionsize);
        desc_count = desc_count + 1;
        
       descs(:,desc_count) = desc(:);
       frames(1:2,desc_count) = frame;
    end

end

end
