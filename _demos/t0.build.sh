###############################################################################

# "TRITON_PLUGIN_DIRS": "${sourceDir}/triton-shared",
git clone git@github.com:microsoft/triton-shared.git

# build & install targets with cmake(cmake presets) in vscode
# cmake --preset iree_llvm
cmake --preset iree_llvm_allen
cmake --build $PWD/build --target all

###############################################################################

# "TRITON_BUILD_PYTHON_MODULE": true,
rm $PWD/python/triton/_C/libtriton.so
ln -sf $PWD/build/libtriton.so $PWD/python/triton/_C/

pushd python
pip install -e .
popd

python -c "import triton; print(triton.__version__)"

###############################################################################

micromamba install mypy

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

###############################################################################
