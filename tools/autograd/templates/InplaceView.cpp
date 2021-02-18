#include "torch/csrc/autograd/VariableTypeUtils.h"

#include <torch/library.h>

#include "torch/csrc/autograd/function.h"
#include <ATen/core/grad_mode.h>

#include <ATen/RedispatchFunctions.h>
#include "ATen/quantized/Quantizer.h"

// ${generated_comment}


using namespace at;

namespace torch {

namespace InplaceView {

namespace {
${inplace_view_method_definitions}
}  // namespace
}  // namespace InplaceView

namespace {

TORCH_LIBRARY_IMPL(aten, Inplace, m) {
  ${inplace_view_wrapper_registrations};
}

}  // namespace

} // namespace torch
