function [frames, descrs] = getLogHistArr(spikeArr)

    count = 1;
    [row, col] = size(spikeArr);
    for i=1:row
        for j=1:col
            %disp(spikeArr{i,j}); disp(length(spikeArr{i,j})); pause;
            currSpikeArr = spikeArr{i,j};
            if isempty(currSpikeArr)
                continue;
            end
            xc = currSpikeArr(1, 1);
            yc = currSpikeArr(1, 2);
            rmin = 3;
            rmax = 34/sqrt(2);
            arr = logHist(currSpikeArr, rmin, rmax, xc, yc, 12, 10); % no  of rings 12, no of wedges is 10 
%             desc = arr(:)/sum(arr(:)); % L1-norm
            desc = uint16(arr(:)); 
            descrs(:, count) = desc;
            frames(:,count) = [xc yc]';
            count = count + 1;
            
            %disp(sum(arr(:))); disp(logHistArr{i,j}); pause;
        end
    end


end