mkdir -p build
cd build

cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -GNinja ..
ninja

echo "Compilation Finish"
cd ..
for file in $(find "./build" -maxdepth 1 -name "*.so"); do
    abs_file=$(realpath $file)
    if [ -e $abs_file ]; then
        ln -s $abs_file ../
        echo "Copied $abs_file..."
    fi
done