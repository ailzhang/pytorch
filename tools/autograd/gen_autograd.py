"""
To run this file by hand from the root of the PyTorch
repository, run:

python -m tools.autograd.gen_autograd \
       build/aten/src/ATen/Declarations.yaml \
       $OUTPUT_DIR \
       tools/autograd

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch/csrc/autograd/generated/
"""

# gen_autograd.py generates C++ autograd functions and Python bindings.
#
# It delegates to the following scripts:
#
#  gen_autograd_functions.py: generates subclasses of torch::autograd::Node
#  gen_variable_type.py: generates VariableType.h which contains all tensor methods
#  gen_python_functions.py: generates Python bindings to THPVariable
#

import argparse
import os
from tools.codegen.selective_build.selector import SelectiveBuilder
from tools.codegen.api.autograd import *
from tools.codegen.gen import parse_native_yaml
from tools.codegen.api.types import *
from typing import List

# See NOTE [ Autograd View Variables ] in variable.h for details.
# If you update list VIEW_FUNCTIONS or RETURNS_VIEWS_OF_INPUT,
# you **MUST** also update the public list of view ops accordingly in
# docs/source/tensor_view.rst. Note not all ATen functions are exposed to public,
# e.g alias & sparse_coo_tensor_with_dims_and_tensors.
#
# A map: function name => name of the argument that all outputs are view of

VIEW_FUNCTIONS_WITH_METADATA_CHANGE = ['view_as_real', 'view_as_complex']

VIEW_FUNCTIONS = {
    'numpy_T': 'self',
    'alias': 'self',
    'as_strided': 'self',
    'diagonal': 'self',
    'expand': 'self',
    'permute': 'self',
    'select': 'self',
    'slice': 'self',
    'split': 'self',
    'split_with_sizes': 'self',
    'squeeze': 'self',
    't': 'self',
    'transpose': 'self',
    'unfold': 'self',
    'unsqueeze': 'self',
    'flatten': 'self',
    'view': 'self',
    'unbind': 'self',
    '_indices': 'self',
    '_values': 'self',
    'indices': 'self',
    'values': 'self',
    # sparse_coo ctor output should really be views of both indices and values,
    # but we only supports making as view of a single variable, and indices is
    # discrete anyways.
    # FIXME: clone indices on construction.
    'sparse_coo_tensor_with_dims_and_tensors': 'values',
}

for key in VIEW_FUNCTIONS_WITH_METADATA_CHANGE:
    VIEW_FUNCTIONS[key] = 'self'

# Functions for which we use CreationMeta::MULTI_OUTPUT_SAFE. I.e., the ones for
# which inplace modification of outputs is being gradually deprecated.
MULTI_OUTPUT_SAFE_FUNCTIONS = {
    'split',
    'split_with_sizes',
}

# note: some VIEW_FUNCTIONS are just compositions of the view functions above
# this list contains both the root view functions and any that are purely composed
# of viewing functions, and is used by the JIT to determine when an operator
# may return a view of its inputs; however they may sometimes return a copy.
# (e.g. `contiguous`)
RETURNS_VIEWS_OF_INPUT = set(VIEW_FUNCTIONS.keys()).union({
    'chunk', 'detach', 'contiguous', 'reshape', 'reshape_as',
    'expand_as', 'view_as', 'real', 'imag', 'narrow', 'movedim',
    'tensor_split', 'swapdims', 'swapaxes'
})

def modifies_arguments(f: NativeFunction) -> bool:
    return f.func.kind() in [SchemaKind.inplace, SchemaKind.out]


def is_inplace_or_view(fn: NativeFunctionWithDifferentiabilityInfo) -> bool:
    f = fn.func
    if modifies_arguments(f):
        return True
    base_name = f.func.name.name.base  # TODO: should be str(f.func.name.name)?
    view_info = VIEW_FUNCTIONS.get(base_name, None)
    if view_info is None and base_name in RETURNS_VIEWS_OF_INPUT:
        view_info = "self"
    return view_info is not None

def inplace_view_method_definition(fn: NativeFunctionWithDifferentiabilityInfo) -> Optional[str]:
    if not is_inplace_or_view(fn):
        return None
    f = fn.func
    return METHOD_DEFINITION.substitute(
        return_type=cpp.returns_type(f.func.returns),
        type_wrapper_name=type_wrapper_name(f),
        formals=gen_formals(f),
        type_definition_body=emit_inplace_view_body(fn),
    )

def match_differentiability_info(
    native_functions: List[NativeFunction],
    differentiability_infos: Sequence[DifferentiabilityInfo],
) -> List[NativeFunctionWithDifferentiabilityInfo]:
    """Sets the "derivative" key on declarations to matching autograd function

    In-place functions will use the out-of-place derivative definition if there
    is no in-place specific derivative.
    """

    info_by_schema = {info.func.func: info for info in differentiability_infos}
    functional_info_by_signature = {
        info.func.func.signature(strip_default=True): info
        for info in differentiability_infos
        if info.func.func.kind() == SchemaKind.functional}

    def find_info(f: NativeFunction) -> Tuple[Optional[DifferentiabilityInfo], bool]:
        if f.func in info_by_schema:
            return info_by_schema[f.func], True

        # if there is no exact match look for the out-of-place signature.
        # i.e mul() for mul_() or mul_out()
        return functional_info_by_signature.get(f.func.signature(strip_default=True)), False

    result: List[NativeFunctionWithDifferentiabilityInfo] = []
    for f in native_functions:
        info, is_exact_match = find_info(f)

        # Currently, the '.strides()' to 'strides_or_error' replacement does not support
        # 'self' derivatives of an inplace function, so we must check for this case.
        if f.func.kind() == SchemaKind.inplace and (info is not None):
            for derivative in info.derivatives:
                if 'self' in derivative.var_names:
                    for saved_input in derivative.saved_inputs:
                        assert 'strides_or_error' not in saved_input.expr, (
                            "Calling '.strides()' in the 'self' derivative formula of an "
                            f"in-place function is not supported: {f.func}")

        result.append(NativeFunctionWithDifferentiabilityInfo(
            func=f,
            info=info,
        ))

    return result

def dispatch_strategy(fn: NativeFunctionWithDifferentiabilityInfo) -> str:
    """How are we going to call the underlying implementation of a
    declaration?  There are two strategies:

        - use_derived: we want to call the implementation on CPUDoubleType
          (or a similar, derived Type instance).  Because these derived
          instances deal in Tensors, not Variables (it's a completely different
          object, so it doesn't dispatch back to VariableType), code on
          this dispatch path needs to wrap/unwrap tensors.  If the
          derived implementation takes and returns tensors, the
          implementation is usually differentiable (although we also use
          the derived dispatch path for non-differentiable functions
          that we still want to dispatch on the derived Type instance;
          e.g., size())

        - use_type: we want to call the implementation on Type, because
          it is implemented concretely, and the functions it invokes will
          get dispatched back to VariableType (which will ensure that they
          are differentiable.)
    """
    if fn.func.is_abstract or (fn.info is not None and fn.info.has_derivatives):
        # If the function is abstract (not implemented on at::Type), we must
        # call the implementation on the derived type with unpacked tensors.

        # If the function has a derivative specified and is concrete, we could
        # call either implementation. We prefer the calling the derived
        # type's implementation with unpacked tensors because it is more
        # performant in some cases: any internal calls to other ATen functions
        # won't have the history tracked.

        # If the function has a type dispatched argument (i.e. is a factory),
        # we prefer calling the derived type's implementation both because it is
        # more performant and to ensure factory functions return tensors with _version
        # of 0 (probably not strictly necessary, but nice to have to keeps versions simple
        # to understand.

        return 'use_derived'
    else:
        # If the function is concrete (we don't have to override it) and we
        # didn't declare it in derivatives.yaml, we'll assume that it is
        # actually implemented out of differentiable functions. (This
        # assumption might not hold, but then you'll see gradcheck fail.)
        return 'use_type'

def gen_autograd(
    aten_path: str,
    native_functions_path: str,
    out: str,
    autograd_dir: str,
    operator_selector: SelectiveBuilder,
    disable_autograd: bool = False,
) -> None:
    # Parse and load derivatives.yaml
    from .load_derivatives import load_derivatives
    differentiability_infos = load_derivatives(
        os.path.join(autograd_dir, 'derivatives.yaml'), native_functions_path)

    template_path = os.path.join(autograd_dir, 'templates')

    fns = list(sorted(filter(
        operator_selector.is_native_function_selected_for_training,
        parse_native_yaml(native_functions_path)), key=lambda f: cpp.name(f.func)))
    fns_with_infos: List[NativeFunctionWithDifferentiabilityInfo] = match_differentiability_info(fns, differentiability_infos)

    # Generate VariableType.h/cpp
    from .gen_trace_type import gen_trace_type
    from .gen_variable_type import gen_variable_type
    from .gen_inplace_view import gen_inplace_view
    if not disable_autograd:
        gen_variable_type(out, native_functions_path, fns_with_infos, template_path)

        # operator filter not applied as tracing sources are excluded in selective build
        gen_trace_type(out, native_functions_path, template_path)

        # generate inplace and view kernel
        gen_inplace_view(out, native_functions_path, fns_with_infos, template_path)

    # Generate Functions.h/cpp
    from .gen_autograd_functions import gen_autograd_functions_lib
    gen_autograd_functions_lib(
        out, differentiability_infos, template_path)

    # Generate variable_factories.h
    from .gen_variable_factories import gen_variable_factories
    gen_variable_factories(out, native_functions_path, template_path)


def gen_autograd_python(
    aten_path: str,
    native_functions_path: str,
    out: str,
    autograd_dir: str,
) -> None:
    from .load_derivatives import load_derivatives
    differentiability_infos = load_derivatives(
        os.path.join(autograd_dir, 'derivatives.yaml'), native_functions_path)

    template_path = os.path.join(autograd_dir, 'templates')

    # Generate Functions.h/cpp
    from .gen_autograd_functions import gen_autograd_functions_python
    gen_autograd_functions_python(
        out, differentiability_infos, template_path)

    # Generate Python bindings
    from . import gen_python_functions
    deprecated_path = os.path.join(autograd_dir, 'deprecated.yaml')
    gen_python_functions.gen(
        out, native_functions_path, deprecated_path, template_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate autograd C++ files script')
    parser.add_argument('declarations', metavar='DECL',
                        help='path to Declarations.yaml')
    parser.add_argument('native_functions', metavar='NATIVE',
                        help='path to native_functions.yaml')
    parser.add_argument('out', metavar='OUT',
                        help='path to output directory')
    parser.add_argument('autograd', metavar='AUTOGRAD',
                        help='path to autograd directory')
    args = parser.parse_args()
    gen_autograd(args.declarations, args.native_functions,
                 args.out, args.autograd,
                 SelectiveBuilder.get_nop_selector())


if __name__ == '__main__':
    main()
