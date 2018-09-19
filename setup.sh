#!/bin/bash
cd /home/ubuntu
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
bash Anaconda3-5.2.0-Linux-x86_64.sh -b -p /home/ubuntu/anaconda
export PATH=/home/ubuntu/anaconda/bin:$PATH
echo 'export PATH=/home/ubuntu/anaconda/bin:$PATH' >>/home/ubuntu/.bashrc
source /home/ubuntu/.bashrc
export PATH=/home/ubuntu/anaconda/bin:$PATH
echo 'export PATH=/home/ubuntu/anaconda/bin:$PATH' >>~/.bashrc
source ~/.bashrc
conda install -y tensorflow=1.8
pip install pydot
pip install keras
pip install cleverhans
git clone https://github.com/chenhx1992/ECG_adv.git
cd ECG_adv/
wget https://www.dropbox.com/s/2kwgtab3ksb7nd0/training_raw.zip
unzip training_raw.zip
python setup_wd.py build_ext --inplace