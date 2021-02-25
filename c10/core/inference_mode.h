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
  // FIXME: remember prev_exclude_set
  InferenceOnlyMode(bool enabled) : prev_mode(InferenceMode::is_enabled()) {
    InferenceMode::set_enabled(enabled);
  }
  ~InferenceOnlyMode() {
    InferenceMode::set_enabled(prev_mode);
  }
  bool prev_mode;
};

} // namespace c10