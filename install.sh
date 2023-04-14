# install indri-5.6
cd scripts
tar -zxf indri-5.6.tar.gz
cd indri-5.6
./configure
make

# install packages from requirements.txt
cd ../../
pip install -r requirements.txt




