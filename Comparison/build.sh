echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../../

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

mkdir build
cd build
cmake ..
make -j4
