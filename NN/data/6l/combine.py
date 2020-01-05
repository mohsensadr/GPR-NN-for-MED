import numpy as np


irange = ["00", "01", "02", "03", "04", "05", "06", "07", "08"]

stf = ""
for i in irange:
    address = i+"/6l.txt"
    file = open(address, "r")
    st =  file.read();
    stf += st;
    file.close();


f = open("6l.txt", "w");
f.write(stf)
f.close();