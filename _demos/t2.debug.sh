python -c "import triton._C.libtriton"

# TODO: fix segmentation fault on macos
# > python -c "import triton._C.libtriton"
# [1]    19509 segmentation fault  python -c "import triton._C.libtriton"

lldb -f $(which python) -- -c "import triton._C.libtriton"
