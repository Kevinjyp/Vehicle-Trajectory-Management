%计算相对于旋转轴的投影
function pro
clear all;
clc;

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

N=10;
pro=zeros(N,round((width^2+height^2)^0.5));
for i=1:N
    theta=pi*i/(2*N);
    for j=1:size(points,2)
        prox=round(points(j).x*cos(theta)+points(j).y*sin(theta));
        pro(i,prox)=pro(i,prox)+1;
    end
end

for k=1:N 
  figure(k+1);
  plot(pro(k,:));
  ylim([0,25]);
end
