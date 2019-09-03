# INSTALL
```sh
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
#(CPU)pip install https://github.com/mind/wheels/releases/download/tf1.8-cpu/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl
pip install https://github.com/mind/wheels/releases/download/tf1.8-gpu-nomkl/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl
pip install pydot
pip install keras
pip install cleverhans==2.1.0
git clone https://github.com/chenhx1992/ECG_adv.git
cd ECG_adv/
wget https://www.dropbox.com/s/ywwh1hxdbcbm0nb/training_raw.zip
unzip training_raw.zip
python setup_wd.py build_ext --inplace
mkdir output
rm -rf training_raw.zip
cd ..
rm -rf Anaconda3-5.2.0-Linux-x86_64.sh
```

# RUN
```sh
#!/bin/bash
export PATH=/home/ubuntu/anaconda/bin:$PATH
cd /home/ubuntu/ECG_adv
git pull https://github.com/chenhx1992/ECG_adv.git
python EOT-test.py
```

GPU
```sh
#!/bin/bash
export PATH=/home/ubuntu/anaconda/bin:$PATH
export PATH=${PATH}:/usr/local/cuda-9.0/bin
export CUDA_HOME=${CUDA_HOME}:/usr/local/cuda:/usr/local/cuda-9.0
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
cd /home/ubuntu/ECG_adv
git pull https://github.com/chenhx1992/ECG_adv.git
python EOT_tile-largetest.py 0 1

#!/bin/bash
export PATH=/home/ubuntu/anaconda/bin:$PATH
export PATH=${PATH}:/usr/local/cuda-9.0/bin
export CUDA_HOME=${CUDA_HOME}:/usr/local/cuda:/usr/local/cuda-9.0
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
cd /home/ubuntu/ECG_adv
git pull https://github.com/chenhx1992/ECG_adv.git
cd EOT_adv
python UniversalPerturbEval-largetest.py 0 1 3000
python UniversalPerturbEval-largetest.py 0 2 3000
python UniversalPerturbEval-largetest.py 0 3 3000
python UniversalPerturbEval-largetest.py 1 0 3000
python UniversalPerturbEval-largetest.py 1 2 3000
python UniversalPerturbEval-largetest.py 1 3 3000
python UniversalPerturbEval-largetest.py 2 0 3000
python UniversalPerturbEval-largetest.py 2 1 3000
python UniversalPerturbEval-largetest.py 2 3 3000
python UniversalPerturbEval-largetest.py 3 0 3000
python UniversalPerturbEval-largetest.py 3 1 3000
python UniversalPerturbEval-largetest.py 3 2 3000
```
