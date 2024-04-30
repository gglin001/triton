# build & install targets with cmake(cmake presets) in vscode

###############################################################################

# "TRITON_BUILD_PYTHON_MODULE": true,
# cp build/libtriton.so python/triton/_C/
ln -sf $PWD/build/libtriton.so $PWD/python/triton/_C/

pushd python
pip install -e .
popd

python -c "import triton; print(triton.__version__)"

###############################################################################

micromamba install mypy
# stubgen -m triton._C.libtriton -o python/
# stubgen -m triton._C.libtriton.triton -o python/
# stubgen -m triton._C.libtriton.triton.runtime -m triton._C.libtriton.triton.ir -o python/
stubgen -m triton._C.libtriton -m triton._C.libtriton.triton -m triton._C.libtriton.triton.runtime -m triton._C.libtriton.triton.ir -o python/

echo "from . import ir" >>python/triton/_C/libtriton/triton/__init__.pyi
echo "from . import runtime" >>python/triton/_C/libtriton/triton/__init__.pyi

###############################################################################
