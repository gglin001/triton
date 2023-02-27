git clone https://github.com/llvm/llvm-project.git --single-branch -b main --depth=1000

pushd llvm-project && git checkout $(cat ../build_tools/llvm_version.txt) && popd

bash build_tools/build_mlir.sh ${PWD}/llvm-project/ ${PWD}/llvm-build
