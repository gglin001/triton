python -c "import triton._C.libtriton"

# > python -c "import triton._C.libtriton"
# [1]    19509 segmentation fault  python -c "import triton._C.libtriton"

lldb -f $(which python) -- -c "import triton._C.libtriton"

###############################################################################

pushd build
lldb -f $(which python) -- -c "import libtriton"
popd

###############################################################################
