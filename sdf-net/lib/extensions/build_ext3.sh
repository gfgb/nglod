# Install C++/CUDA extensions
for ext in mesh2sdf_cuda sol_nglod; do
# for ext in mesh2sdf_cuda; do
    cd $ext && python3 setup.py clean --all install --user && cd -
done
