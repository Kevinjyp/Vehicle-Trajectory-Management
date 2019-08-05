%计算相对于旋转轴的转动惯量
function inertia
string='E:\track\track\trajectory\_save_1885.jpg';
tra=imread(string);
figure(1);
imshow(tra);

width=size(tra,2);
height=size(tra,1);
num=1;
for i=1:height
    for j=1:width
        if(tra(i,j)==255) 
            points(num).x=j;
            points(num).y=i;
            num=num+1;
        end
    end
end

meanx=0;meany=0;
%centralize
for i=1:size(points,2)
    meanx=meanx+points(i).x;
    meany=meany+points(i).y;
end
meanx=meanx/size(points,2);
meany=meany/size(points,2);

%cal inertia
N=100;
interval=zeros(1,N);
for i=1:N
    theta=pi*i/N;
    for j=1:size(points,2)
        inertia(i)=inertia(i)+((points(j).x-meanx)*sin(theta)-(points(i).y-meany)*cos(theta))^2;
    end
end
figure(2);
plot(inertia);