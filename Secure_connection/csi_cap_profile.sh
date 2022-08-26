#! /bin/bash

echo > /dev/null | tee /var/log/csi.dat
sleep 120
# scp nss@192.168.1.196:/var/log/csi.dat /Users/liangxintai/Desktop/
# sshpass -p nss ssh root@192.168.1.196 "echo > /dev/null | sudo tee /var/log/csi.dat"

cp /var/log/csi.dat /home/nss/profile.dat
echo > /dev/null | tee /var/log/csi.dat
# echo hello
