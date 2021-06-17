#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <torch/library.h>

/*
 * This file implements a variable fallback kernel for custom operators.
 * Since tensors always have the Autograd set, but custom operators
 * usually don't have a kernel registered for Autograd, the dispatcher
 * will call into this fallback kernel instead.
 * Note that this is not a correct autograd implementation. It will just
 * fallthrough to the custom operator implementation.
 * If you want a custom operator to work with autograd, you need to use
 * autograd::Function so that the custom operator implementation knows how to
 * do autograd.
 * Note also that ops from native_functions.yaml register their own variable
 * kernels, so this is never called for them.
 */

// TODO This whole file should be deleted and replaced with the mechanism
//      described in https://github.com/pytorch/pytorch/issues/29548

using c10::OperatorHandle;
using c10::Stack;
using c10::DispatchKey;
using c10::DispatchKeySet;
using c10::Dispatcher;
using c10::KernelFunction;

namespace {

// Register fallthrough for Autograd backends dispatch keys
// NB: But not the private use ones; maybe the extension wants
// to override it themselves!

TORCH_LIBRARY_IMPL(_, AutogradOther, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, AutogradCPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, AutogradXPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, AutogradCUDA, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, AutogradXLA, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, AutogradMLC, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

namespace {
  void func2Fallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatchKeySet, torch::jit::Stack* stack) {
    const auto& schema = op.schema();
    const auto num_arguments = schema.arguments().size();
    const auto arguments_begin = stack->size() - num_arguments;
    auto arguments = torch::jit::last(stack, num_arguments);
    for (int64_t idx = 0; idx < num_arguments; ++idx) {
      const auto& ivalue = arguments[idx];
      if (ivalue.isTensor()) {
        at::Tensor t = ivalue.toTensor();
        if (t.has_view_meta() && !t.is_up_to_date()) {
          t.sync_();
        }
        auto materialized_ivalue = c10::IValue(t);
        (*stack)[arguments_begin + idx] = std::move(materialized_ivalue);
      } else if (ivalue.isTensorList()) {
        std::vector<at::Tensor> tensors = ivalue.toTensorList().vec();
        for (auto& t: tensors) {
          if (t.has_view_meta()) {
            t.sync_();
          }
        }
        auto materialized_ivalue= c10::IValue(c10::List<at::Tensor>(tensors));
        (*stack)[arguments_begin + idx] = std::move(materialized_ivalue);
      }
    }
    {
      at::AutoDispatchBelowFunc2 guard;
      // redispatchBoxed with specified dispatchKeySet cannot prevent composite kernels
      // called inside from going back up dispatcher. We still need the RAII guard here.
      op.redispatchBoxed(dispatchKeySet & c10::after_ADInplaceOrView_keyset, stack);
    }
  }
}

TORCH_LIBRARY_IMPL(_, Func2, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&func2Fallback>());
}

// see Note [ADInplaceOrView key]
TORCH_LIBRARY_IMPL(_, ADInplaceOrView, m) {
      m.fallback(torch::CppFunction::makeFallthrough());
}

}
