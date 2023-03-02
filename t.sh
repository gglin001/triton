# dev in docker image:
# https://github.com/gglin001/Dockerfiles/blob/master/clangdev/Dockerfile

git clone https://github.com/llvm/llvm-project.git --single-branch -b main --depth=1

pushd llvm-project && git fetch --depth 1 origin $(cat ../build_tools/llvm_version.txt) && popd

bash build_tools/build_mlir.sh ${PWD}/llvm-project/ ${PWD}/llvm-build

# apt install libz-dev
pip install pybind11-global
mamba install pytorch

# build & install targets with cmake in vscode

cp build/libtriton.so python/triton/_C/

pushd python
pip install -e .
popd

python -c "import triton; print(triton.__version__)"

pip install mypy
# stubgen -m triton._C.libtriton -o python/
# stubgen -m triton._C.libtriton.triton -o python/
# stubgen -m triton._C.libtriton.triton.runtime -m triton._C.libtriton.triton.ir -o python/
stubgen -m triton._C.libtriton -m triton._C.libtriton.triton -m triton._C.libtriton.triton.runtime -m triton._C.libtriton.triton.ir -o python/

echo "from . import ir" >>python/triton/_C/libtriton/triton/__init__.pyi
echo "from . import runtime" >>python/triton/_C/libtriton/triton/__init__.pyi
