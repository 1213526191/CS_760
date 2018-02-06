x1 = seq(-1,0,0.001)
x2 = seq(0,1,0.001)
y1 = x1 + 1
y2 = -1 - x1
y3 = 1 - x2
y4 = -1 + x2
plot(x1,y1,xlim = c(-1,1), ylim = c(-1,1))
points(x1,y2)
points(x2,y3)
points(x2,y4)


x3 = seq(-1,1,0.001)
y5 = (1-x3^2)^0.5
y6 = -(1-x3^2)^0.5
plot(x3,y5,xlim = c(-1,1), ylim = c(-1,1))
points(x3,y6)
xx1 = c(rep(-1,201))
xx2 = c(rep(1,201))
yy = seq(-0.1,0.1,0.001)
points(xx1,yy)
points(xx2,yy)


x11 = seq(-1,1,0.001)
x12 = rep(1,2001)
x13 = rep(-1,2001)
y11 = seq(-1,1,0.001)
y12 = rep(1,2001)
y13 = rep(-1,2001)
plot(x11,y12,xlim = c(-1.5,1.5), ylim = c(-1.5,1.5))
points(x11,y13)
points(x12,y11)
points(x13,y11)
