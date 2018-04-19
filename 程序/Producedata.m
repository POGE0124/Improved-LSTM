clc
close all
clear all
load data2
b = a(1:100);
c = a(101:end);
[m,refl] = ar(b,2)
yp = predict(m,c,1)
plot(yp,'g')
hold on 
plot(c,'r')
legend('pre','true')

