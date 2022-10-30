cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

mkdir build
cd build
cmake ..
make -j4
