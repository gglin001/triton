#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 32, 32]}>
#shared = #triton_gpu.shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  tt.func @dot_high_precision_acc(%a: !tt.memdesc<32x32xf8E5M2, #shared>, %b: !tt.memdesc<32x32xf8E5M2, #shared1>, %c: tensor<32x32xf32, #mma>) {
    %m = triton_nvidia_gpu.warp_group_dot %a, %b, %c
      {maxNumImpreciseAcc = 32 : i32, inputPrecision = 0 : i32} :
      !tt.memdesc<32x32xf8E5M2, #shared> * !tt.memdesc<32x32xf8E5M2, #shared1> -> tensor<32x32xf32, #mma>
    tt.return
  }
}
