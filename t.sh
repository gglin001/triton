# dev in docker image:
# https://github.com/gglin001/Dockerfiles/blob/master/clangdev/Dockerfile

git clone https://github.com/llvm/llvm-project.git --single-branch -b main --depth=1000

pushd llvm-project && git checkout $(cat ../build_tools/llvm_version.txt) && popd

bash build_tools/build_mlir.sh ${PWD}/llvm-project/ ${PWD}/llvm-build

apt install libz-dev

# build & install targets with cmake in vscode

cp build/libtriton.so python/triton/_C/

pushd python
pip install -e .
popd

python -c "import triton; print(triton.__version__)"
