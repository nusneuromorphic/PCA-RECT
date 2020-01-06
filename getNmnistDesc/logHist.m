%Generate a log polar histogram
%Input an array of spike coordinates and the center point
function output = logHist(arr, rmin, rmax, xc, yc, nr, nw)
    
output = zeros(nw, nr);
input = [];
for i=1:length(arr)
    rad = sqrt((arr(i,1) - xc)^2 + (arr(i,2) - yc)^2);
    theta = atan2(arr(i,2) - yc, arr(i,1) - xc);
    input = [input [rad theta]];
end

input = reshape(input, 2, [])';
%disp(input); disp(arr); pause;

r_bin_edges=logspace(log10(rmin),log10(rmax),nr);

for i=1:length(input)
   ringNum = 1;
   index = 1;
   while input(i,1)> (r_bin_edges(index))
       %fprintf('%f > %f\n', input(i,1), r_bin_edges(index));
       ringNum = ringNum+1;
       index = index + 1;
       if index==nr
           break;
       end
   end;
   wedgeNum = ceil(nw*input(i,2)/(2*pi)) + nw/2;%to remove negative values
   %fprintf('x = %d, y = %d\n', arr(i, 1), arr(i, 2));
   %fprintf('rad = %f, theta = %f, ringNum = %d, wedgeNum = %d\n', input(i,1), input(i,2), ringNum, wedgeNum);
   %pause;
   output(wedgeNum, ringNum) = output(wedgeNum, ringNum) + 1;

end
end
