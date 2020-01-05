import numpy as np


irange = ["00", "old"]

stf = ""
for i in irange:
    address = i+"/4l.txt"
    file = open(address, "r")
    st =  file.read();
    stf += st;
    file.close();


f = open("4l_conv.txt", "w");
f.write(stf)
f.close();