#!/bin/bash
# Install spinningup environment.

echo -e "\033[1;36mStep 1: Create conda environment.\n\033[0m"
# conda env remove --name spinningup
conda create -y --name spinningup python=3.7
source activate spinningup
conda install -y tensorflow=1.15 cloudpickle ipython joblib matplotlib mpi4py numpy pandas pytest psutil scipy seaborn tqdm zipp importlib-metadata lockfile imageio cython pyopengl pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

echo -e "\033[1;36mStep 2: Install mujoco and pip dependencies.\n\033[0m"
echo -e "\033[1;36mFor mujoco, these packages are needed on the host system. They should already been installed on UCLA machines.\033[0m"
echo -e "sudo apt install -y libosmesa6-dev patchelf libglvnd-dev libgl-dev"
echo
wget http://d.huan-zhang.com/storage/software/mjpro150.tar.bz2
mkdir ${HOME}/.mujoco
tar -C ${HOME}/.mujoco -jxvf mjpro150.tar.bz2; rm mjpro150.tar.bz2
cp ${HOME}/.mujoco/mjpro150/bin/mjkey.txt ${HOME}/.mujoco
echo -e "\033[1;36mRemember to change your LD_LIBRARY_PATH in .bashrc by adding the following line to .bashrc:\033[0m"
echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/.mujoco/mjpro150/bin'
echo
# echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mjpro150/bin" >> ${HOME}/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mjpro150/bin
pip install --no-cache-dir gym[mujoco,atari,box2d,classic_control] swig glfw mujoco
pip install --no-cache-dir mujoco-py==1.50.1.68
echo -e "\033[1;36mIf you encounter /lib/libstdc++.so.6: version GLIBCXX_3.4.30 not found error, run the following:\033[0m"
echo 'source activate spinningup; ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 $(dirname $(which python))/../lib/libstdc++.so.6'
echo

echo -e "\033[1;36mStep 3: Install spinningup.\n\033[0m"
git clone https://github.com/openai/spinningup.git
cd spinningup; python setup.py develop --no-deps

echo -e "\033[1;36m\nStep 4: Run your tests. Run the following commands yourself.\n\033[0m"
echo "cd spinningup; source activate spinningup"
echo 'python -m spinup.run ppo --hid "[32,32]" --env Walker2d-v2 --exp_name mujocotest'

