clc
clear all
close all
load B0005
j=1;
for i=1:616
    if strcmpi(B0005.cycle(i).type,'impedance')
        a(j)=B0005.cycle(i).data.Rct;
        j=j+1;
    end
end
plot(a)
