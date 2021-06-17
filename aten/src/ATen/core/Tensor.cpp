#include <ATen/core/Tensor.h>
#include <ATen/core/Formatting.h>
#include <ATen/core/VariableHooksInterface.h>

#include <iostream>

namespace at {

void Tensor::enforce_invariants() {
  if (impl_.get() == nullptr) {
    throw std::runtime_error("TensorImpl with nullptr is not supported");
  }
  // Following line throws if the method is not a POD data type or is not
  // supported by ATen
  scalar_type();
  if (defined()) {
    TORCH_INTERNAL_ASSERT(
        impl_->dtype_initialized(),
        "Partially-initialized tensor not supported by Tensor");
    TORCH_INTERNAL_ASSERT(
        !impl_->is_sparse(),
        "Sparse Tensors are supported by Tensor, but invariant checking isn't implemented.  Please file a bug.");
    TORCH_INTERNAL_ASSERT(
        impl_->storage_initialized(),
        "Partially-initialized tensor not supported by Tensor");
  }
}

void Tensor::print() const {
  if (defined()) {
    std::cerr << "[" << toString() << " " << sizes() << "]" << std::endl;
  } else {
    std::cerr << "[UndefinedTensor]" << std::endl;
  }
}

std::string Tensor::toString() const {
  std::string base_str;
  if (scalar_type() == ScalarType::Undefined) {
    base_str = "UndefinedType";
  } else {
    base_str = std::string(at::toString(options().computeDispatchKey())) + at::toString(scalar_type()) + "Type";
  }
  return base_str;
}

Tensor Tensor::variable_data() const {
  return impl::GetVariableHooks()->variable_data(*this);
}

Tensor Tensor::tensor_data() const {
  return impl::GetVariableHooks()->tensor_data(*this);
}

bool Tensor::is_leaf() const {
  return impl::GetVariableHooks()->is_leaf(*this);
}

int64_t Tensor::output_nr() const {
  return impl::GetVariableHooks()->output_nr(*this);
}

void Tensor::set_data(const Tensor & new_data) const {
  impl::GetVariableHooks()->set_data(*this, new_data);
}

Tensor Tensor::data() const {
  return impl::GetVariableHooks()->data(*this);
}

int64_t Tensor::_version() const {
  return impl::GetVariableHooks()->_version(*this);
}

void Tensor::retain_grad() const {
  impl::GetVariableHooks()->retain_grad(*this);
}

bool Tensor::retains_grad() const {
  return impl::GetVariableHooks()->retains_grad(*this);
}

void Tensor::_backward(TensorList inputs,
        const c10::optional<Tensor>& gradient,
        c10::optional<bool> keep_graph,
        bool create_graph) const {
  return impl::GetVariableHooks()->_backward(*this, inputs, gradient, keep_graph, create_graph);
}

const Tensor& Tensor::requires_grad_(bool _requires_grad) const {
  impl::GetVariableHooks()->requires_grad_(*this, _requires_grad);
  return *this;
}

// View Variables
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bool Tensor::is_view() const {
  return impl::GetVariableHooks()->is_view(*this);
}

const Tensor& Tensor::_base() const {
  return impl::GetVariableHooks()->base(*this);
}

bool Tensor::is_up_to_date() const {
  if (alias_) {
    return generation_ == alias_->generation();
  }
  return true;
}

void Tensor::add_update(const at::Tensor& updated_val, std::vector<at::ViewMeta> metas) {
  alias_->add_update(updated_val, metas);
}

void Tensor::sync_() {
  if (is_up_to_date()) {
    return;
  }
  // Apply all updates on alias_
  alias_->SyncUpdateOperations();
  // Reapply views to Get the viewed tensor from updated base in alias_
  auto t = alias_->base();
  for (auto& view_meta: view_metas_) {
    switch (view_meta.view_type) {
      case ViewMeta::Type::kReshape:
          t = t.view_copy(view_meta.size);
          break;
      case ViewMeta::Type::kNoOp:
          break;
      default:
          TORCH_CHECK(false, "Other types are not supported yet.");
    }
  }
  // Note this goes back to dispatcher but set_ is simply redispatch
  // at Func2. (fallback kernel materializes tensors before redispatch)
  this->set_(t);
  generation_ = alias_->generation();
}

const std::string& Tensor::name() const {
  return impl::GetVariableHooks()->name(*this);
}

const std::shared_ptr<torch::autograd::Node>& Tensor::grad_fn() const {
  return impl::GetVariableHooks()->grad_fn(*this);
}

void Tensor::remove_hook(unsigned pos) const {
  impl::GetVariableHooks()->remove_hook(*this, pos);
}

bool Tensor::is_alias_of(const at::Tensor& other) const {
  // If self and other are the same
  if (unsafeGetTensorImpl() == other.unsafeGetTensorImpl()) return true;
  // For tensors without storage, check alias_ information
  if (has_view_meta()) {
    return alias_->base().unsafeGetTensorImpl() == other.unsafeGetTensorImpl();
  }
  return impl_->storage().is_alias_of(other.storage());
}

unsigned Tensor::_register_hook(std::function<Tensor(const Tensor&)> hook) const {
  return impl::GetVariableHooks()->_register_hook(*this, std::move(hook));
}

const at::Tensor Alias::base() const {
  return base_;
}

void Alias::add_update(const at::Tensor& updated_val, std::vector<at::ViewMeta> metas) {
  updates_.push_back({updated_val, metas});
  generation_++;
}

void Alias::apply_update(const Update& update) {
  // TODO: Should handle more kinds of view ops. Only do kReshape now.
  at::Tensor t = update.new_val;
  for(int i = update.view_metas.size()-1; i >= 0; --i) {
    switch (update.view_metas[i].view_type) {
      case ViewMeta::Type::kReshape:
          t = t.view_copy(update.view_metas[i].source_size);
          break;
      case ViewMeta::Type::kNoOp:
          break;
      default:
          TORCH_CHECK(false, "Other types are not supported yet.");
    }
  }
  base_.set_(t);
}

void Alias::SyncUpdateOperations() {
  for (auto& update_data: updates_) {
    apply_update(update_data);
  }
  updates_.clear();
}

} // namespace at
