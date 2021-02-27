#pragma once

#include <c10/macros/Macros.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

namespace c10 {

// public API 


struct TORCH_API InferenceMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};

// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct TORCH_API InferenceOnlyMode {
  InferenceOnlyMode(bool enabled) : prev_mode(InferenceMode::is_enabled()),
  autograd_guard_(enabled? c10::autograd_dispatch_keyset : DispatchKeySet()) {
    InferenceMode::set_enabled(enabled);
  }
  ~InferenceOnlyMode() {
    InferenceMode::set_enabled(prev_mode);
  }
  bool prev_mode;
  // MUST HAVE
  /*
  
    torch::Tensor a = torch::ones({1, 2, 3}).set_requires_grad(true);
    torch::Tensor k = a + 2;
    {
      c10::InferenceOnlyMode guard(true);
      // Good! add_ still go through Inplace kernel so that it's prepared for future autograd
      k.add_(2);
    }
  */
  c10::impl::ExcludeDispatchKeyGuard autograd_guard_;
};

} // namespace c10