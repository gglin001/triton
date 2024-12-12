###############################################################################

# micromamba create -n torch_cpu python=3.12 pytorch

# pip install cmake ninja pybind11 pybind11-global

###############################################################################

# copy headers
pushd python
python setup.py
popd

###############################################################################

# "TRITON_PLUGIN_DIRS": "${sourceDir}/triton-shared",
# git clone git@github.com:microsoft/triton-shared.git
# ln -s $PWD/../triton-shared $PWD/triton-shared

###############################################################################

# cmake --preset osx
cmake --preset osx_allen

cmake --build $PWD/build --target all

cmake --build $PWD/build --target help
cmake --build $PWD/build --target triton

###############################################################################

# "TRITON_BUILD_PYTHON_MODULE": true,
rm $PWD/python/triton/_C/libtriton.so
ln -sf $PWD/build/libtriton.so $PWD/python/triton/_C/
# ln -s $PWD/triton-shared/backend $PWD/python/triton/backends/cpu

# # "TRITON_BUILD_PROTON": true
# rm $PWD/python/triton/_C/libproton.so
# ln -sf $PWD/build/third_party/proton/libproton.so $PWD/python/triton/_C/

pushd python
# pip install -e . -vvv
pip install --no-build-isolation -e . -vvv
popd

python -c "import triton; print(triton.__version__)"
python -c "import triton; print(triton.__file__)"

ln -s ~/.cache $PWD/_demos/.cache

###############################################################################

# micromamba install mypy

mkdir -p python/triton/_C/libtriton
args=(
  -m triton._C.libtriton
  -m triton._C.libtriton.ir
  -m triton._C.libtriton.passes
  -m triton._C.libtriton.interpreter
  -m triton._C.libtriton.llvm
  -o python
)
stubgen "${args[@]}"

######

args=(
  -m triton._C.libtriton.passes.analysis
  -m triton._C.libtriton.passes.common
  -m triton._C.libtriton.passes.convert
  -m triton._C.libtriton.passes.ttir
  -m triton._C.libtriton.passes.ttgpuir
  -m triton._C.libtriton.passes.llvmir
  -o python
)
stubgen "${args[@]}"

args=(
  -m triton._C.libtriton.nvidia
  -m triton._C.libtriton.nvidia.passes
  -m triton._C.libtriton.nvidia.passes.ttgpuir
  -m triton._C.libtriton.nvidia.passes.ttnvgpuir
  -o python
)
stubgen "${args[@]}"

###############################################################################
