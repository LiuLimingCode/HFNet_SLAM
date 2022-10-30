echo "Building ROS nodes"

cd Examples/ROS/HFNet_SLAM
curpath=`pwd`
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:"$curpath"/../ 
mkdir build
cd build
cmake .. -DROS_BUILD_TYPE=Release
make -j
