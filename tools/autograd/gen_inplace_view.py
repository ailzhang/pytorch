import itertools
from typing import Optional, List, Sequence, Union

from tools.codegen.api.types import *
import tools.codegen.api.cpp as cpp
from tools.codegen.code_template import CodeTemplate
from tools.codegen.api.autograd import *
from tools.codegen.context import with_native_function
from .gen_autograd_functions import uses_single_grad
from tools.codegen.utils import mapMaybe
from tools.codegen.gen import parse_native_yaml, FileManager
from tools.codegen.model import *
from tools.autograd.gen_variable_type import gen_formals, modifies_arguments, \
    unpack_args, is_tensor_type, is_tensor_list_type, REPLAY_VIEW_LAMBDA_FUNC, \
    SETUP_REPLAY_VIEW_IF_NOT_SUPPORT_AS_STRIDED_OR_VIEW_WITH_METADATA_CHANGE, \
    VIEW_FUNCTIONS_WITH_METADATA_CHANGE, CALL_DISPATCH_VIA_NAMESPACE, \
    CALL_DISPATCH_VIA_METHOD, ASSIGN_RETURN_VALUE, ARRAYREF_TO_VEC, OPTIONAL_TO_VAL
from .gen_autograd import VIEW_FUNCTIONS, VIEW_FUNCTIONS_WITH_METADATA_CHANGE, \
    MULTI_OUTPUT_SAFE_FUNCTIONS, RETURNS_VIEWS_OF_INPUT, dispatch_strategy
from .gen_trace_type import (
    MANUAL_AUTOGRAD, declare_returned_variables, tie_return_values, get_return_value, type_wrapper_name,
)

INPLACE_VIEW_DISPATCH = CodeTemplate("""\
${assign_return_values}at::redispatch::${api_name}(${unpacked_args});""")

def emit_inplace_view_body(fn: NativeFunctionWithDifferentiabilityInfo) -> List[str]:
    f = fn.func
    inplace_view_body: List[str] = []

    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    dispatcher_exprs = dispatcher_sig.exprs()

    ret_and_arg_types = ', '.join([dispatcher_sig.returns_type()] + [a.type.cpp_type() for a in dispatcher_exprs])
    # code-generated tracing kernels plumb and recompute dispatch keys directly through the kernel for performance.
    # See Note [Plumbing Keys Through The Dispatcher] for details.
    dispatch_key_set = 'ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Inplace)'
    redispatch_args = ', '.join([dispatch_key_set] + [a.expr for a in dispatcher_exprs])

    # Note that this calls the slow, dispatching variants of manual_cpp_binding ops.
    # We could probably work harder to ensure that the fast variants are called instead, but the perf benefit would be minimal.
    sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=f.manual_cpp_binding)
    if sig_group.faithful_signature is not None:
        api_name = sig_group.faithful_signature.name()
    else:
        api_name = sig_group.signature.name()

    if modifies_arguments(f):
        assign_return_values = f'{tie_return_values(f)} = ' \
            if f.func.kind() == SchemaKind.functional and f.func.returns else ' '
        inplace_view_body.append(INPLACE_VIEW_DISPATCH.substitute(
            assign_return_values=assign_return_values,
            api_name=api_name,
            unpacked_args=redispatch_args,
        ))
        for r in cpp.return_names(f):
            inplace_view_body.append(f'torch::autograd::increment_version({r});')
    else:  # view op
        unpack_args_stats, unpacked_bindings = unpack_args(f)
        var = 'tmp'
        inplace_view_body.append(INPLACE_VIEW_DISPATCH.substitute(
            assign_return_values='auto ' + var + ' = ',
            api_name=api_name,
            unpacked_args=redispatch_args,
        ))
        info = fn.info
        base_name = f.func.name.name.base  # TODO: should be str(f.func.name.name)?
        view_info = VIEW_FUNCTIONS.get(base_name, None)
        if view_info is None and base_name in RETURNS_VIEWS_OF_INPUT:
            view_info = "self"
        def is_differentiable(name: str, type: Type) -> bool:
            return type.is_tensor_like() and (info is None or name not in info.non_differentiable_arg_names)
        def gen_differentiable_outputs(f: NativeFunction) -> List[DifferentiableOutput]:
            outputs: List[DifferentiableOutput] = [
                DifferentiableOutput(name=name, type=ret.type, cpp_type=cpp.return_type(ret))
                for name, ret in zip(cpp.return_names(f), f.func.returns)]

            output_differentiability = info.output_differentiability if info else None
            if output_differentiability is not None:
                differentiable_outputs: List[DifferentiableOutput] = []
                if False in output_differentiability and f.func.kind() == SchemaKind.inplace:
                    raise RuntimeError("output_differentiability=False for inplace operation (version_counter won't get updated)")
                for differentiable, output in zip(output_differentiability, outputs):
                    if differentiable:
                        differentiable_outputs.append(output)
                return differentiable_outputs

            candidate_differentiable_outputs = list(filter(lambda r: is_differentiable(r.name, r.type), outputs))

            if uses_single_grad(info):
                return candidate_differentiable_outputs[:1]
            else:
                return candidate_differentiable_outputs
        differentiable_outputs = gen_differentiable_outputs(f)
        differentiable_output_vars = {r.name for r in differentiable_outputs}

        def emit_view_lambda(unpacked_bindings: List[Binding]) -> str:
            """ Generate an additional lambda function to recover views in backward when as_strided is not supported.
            See Note [View + Inplace update for base tensor] and [View + Inplace update for view tensor] for more details."""
            input_base = 'input_base'
            replay_view_func = ''
            updated_unpacked_args: List[str] = []
            known_view_arg_simple_types: List[str] = ['int64_t', 'c10::optional<int64_t>', 'bool', 'IntArrayRef']
            for unpacked_binding in unpacked_bindings:
                arg, arg_type = unpacked_binding.name, unpacked_binding.type
                if arg == 'self_':
                    updated_unpacked_args.append(input_base)
                    continue
                if arg_type not in known_view_arg_simple_types:
                    known_types_str = ', '.join(known_view_arg_simple_types)
                    raise TypeError(f'You are adding an {arg_type} {arg} argument to op {cpp.name(f.func)} in addition to known types: '
                                    f'{known_types_str}. Please update the list or materialize it so that it can be closed '
                                    'over by value, also add a test in pytorch/xla/test/test_operations.py where this code '
                                    'is exercised.')
    
                if arg_type == 'IntArrayRef':
                    # It's not safe to close over IntArrayRef by value, since this is a
                    # reference type, so materialize a vector to close over by value
                    arg_vec = arg + '_vec'
                    inplace_view_body.append(ARRAYREF_TO_VEC.substitute(arg=arg, vec=arg_vec))
                    updated_unpacked_args.append(arg_vec)
                elif arg_type == 'c10::optional<int64_t>':
                    # Materialize int64_t? to int64_t
                    arg_value = arg + '_val'
                    inplace_view_body.append(OPTIONAL_TO_VAL.substitute(arg=arg, val=arg_value, default='0'))
                    updated_unpacked_args.append(arg_value)
                else:
                    updated_unpacked_args.append(arg)

            def emit_dispatch_call(f: NativeFunction, input_base: str, unpacked_args: Sequence[str]) -> str:
                dispatcher_sig = DispatcherSignature.from_schema(f.func)
                dispatcher_exprs = dispatcher_sig.exprs()
                if Variant.function in f.variants:
                    call = CALL_DISPATCH_VIA_NAMESPACE.substitute(
                        api_name=cpp.name(
                            f.func,
                            faithful_name_for_out_overloads=True,
                        ),
                        unpacked_args=unpacked_args)
                else:
                    call = CALL_DISPATCH_VIA_METHOD.substitute(
                        api_name=cpp.name(f.func),
                        var=input_base,
                        unpacked_method_args=unpacked_args[1:])
                return call

            replay_view_call = emit_dispatch_call(f, input_base, updated_unpacked_args)
            replay_view_func = REPLAY_VIEW_LAMBDA_FUNC.substitute(
                input_base=input_base,
                replay_view_call=replay_view_call)
            name = cpp.name(f.func)
            is_view_with_metadata_change = 'true' if name in VIEW_FUNCTIONS_WITH_METADATA_CHANGE else 'false'
    
            return SETUP_REPLAY_VIEW_IF_NOT_SUPPORT_AS_STRIDED_OR_VIEW_WITH_METADATA_CHANGE.substitute(
                is_view_with_metadata_change=is_view_with_metadata_change,
                replay_view_func=replay_view_func)
        
        if not isinstance(view_info, str):
            raise TypeError(f'The view info should be a string for {base_name}, but it is: {view_info}')
        if len(differentiable_output_vars) == 0:
            # no output is differentiable (.indices() for SparseTensors for example)
            rhs_value = f'torch::autograd::as_view({view_info}, {var}, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false)'
        elif len(differentiable_output_vars) == 1:
            # Single differentiable output (Tensor or Tensor[])
            return_info = differentiable_outputs[0]
            # We only support simple Tensor or a TensorList for functions that return views
            if not is_tensor_type(return_info.type) and not is_tensor_list_type(return_info.type):
                raise RuntimeError(f'{base_name} that return differentiable views can only return Tensor or Tensor[]')
            # Only allow rebasing of the history if we return a single Tensor
            # If we are in a no grad block, raise a warning
            # See NOTE [ View + Inplace detection ] for more details about this logic
            if is_tensor_list_type(return_info.type):
                if base_name in MULTI_OUTPUT_SAFE_FUNCTIONS:
                    creation_meta = 'torch::autograd::CreationMeta::MULTI_OUTPUT_SAFE'
                else:
                    creation_meta = 'torch::autograd::CreationMeta::MULTI_OUTPUT_NODE'
                inplace_view_body.append(f'torch::autograd::as_view(/* base */ {view_info}, /* output */ {var}, /* is_bw_differentiable */ true, '
                         '/* is_fw_differentiable */ true, '
                         f'/* creation_meta */ {creation_meta});')
                rhs_value = f'std::move({var})'
            else:
                inplace_view_body.append(emit_view_lambda(unpacked_bindings))
                creation_meta = 'at::GradMode::is_enabled() ? torch::autograd::CreationMeta::DEFAULT: torch::autograd::CreationMeta::NO_GRAD_MODE'
                rhs_value = (f'torch::autograd::as_view(/* base */ {view_info}, /* output */ {var}, /* is_bw_differentiable */ true, '
                             '/* is_fw_differentiable */ true, '
                             f'/* view_func */ func, /* creation_meta */ {creation_meta})')
        else:
            # This could be supported but we don't need it at the moment, so keeping things simple.
            raise RuntimeError('Function that return multiple differentiable output '
                               'when at least one of them is view is not supported.')
        assert rhs_value is not None
        inplace_view_body.append(ASSIGN_RETURN_VALUE.substitute(return_values=tie_return_values(f),
                                               rhs_value=rhs_value))
    if f.func.returns:
        inplace_view_body.append(f'return {get_return_value(f)};')
    return inplace_view_body

def is_inplace_or_view(fn: NativeFunctionWithDifferentiabilityInfo) -> bool:
    f = fn.func
    if modifies_arguments(f):
        return True
    base_name = f.func.name.name.base  # TODO: should be str(f.func.name.name)?
    view_info = VIEW_FUNCTIONS.get(base_name, None)
    if view_info is None and base_name in RETURNS_VIEWS_OF_INPUT:
        view_info = "self"
    return view_info is not None

METHOD_DEFINITION = CodeTemplate("""\
${return_type} ${type_wrapper_name}(${formals}) {
  ${type_definition_body}
}
""")

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

WRAPPER_REGISTRATION = CodeTemplate("""\
m.impl("${name}",
       TORCH_FN(${class_type}::${type_wrapper_name})
);
""")

def inplace_view_method_registration(fn: NativeFunctionWithDifferentiabilityInfo) -> Optional[str]:
    if not is_inplace_or_view(fn):
        return None
    f = fn.func
    return WRAPPER_REGISTRATION.substitute(
        name=f.func.name,
        type_wrapper_name=type_wrapper_name(f),
        class_type='InplaceView',
    )

def gen_inplace_view_shard(
    fm: FileManager, fns_with_infos: List[NativeFunctionWithDifferentiabilityInfo], suffix: str
) -> None:

    def need(fn: NativeFunctionWithDifferentiabilityInfo) -> bool:
        f = fn.func
        name = cpp.name(f.func)
        return name not in MANUAL_AUTOGRAD and dispatch_strategy(fn) == 'use_derived'

    filtered_fns_with_infos = list(filter(need, fns_with_infos))

    fm.write_with_template('InplaceView%s.cpp' % suffix, 'InplaceView.cpp', lambda: {
        'generated_comment': '@' + f'generated from {fm.template_dir}/InplaceView.cpp',
        'inplace_view_method_definitions': list(mapMaybe(inplace_view_method_definition, filtered_fns_with_infos)),
        'inplace_view_wrapper_registrations': list(mapMaybe(inplace_view_method_registration, filtered_fns_with_infos)),
    })

def gen_inplace_view(out: str, native_yaml_path: str, fns_with_infos: List[NativeFunctionWithDifferentiabilityInfo], template_path: str) -> None:
    # NOTE: see Note [Sharded File] at the top of the VariableType.cpp
    # template regarding sharding of the generated files.
    num_shards = 2
    shards: List[List[NativeFunctionWithDifferentiabilityInfo]] = [[] for _ in range(num_shards)]
 
    # functions are assigned arbitrarily but stably to a file based on hash
    for fn in fns_with_infos:
        x = sum(ord(c) for c in cpp.name(fn.func.func)) % num_shards
        shards[x].append(fn)

    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    for i, shard in enumerate(shards):
        gen_inplace_view_shard(fm, shard, f'_{i}')
    gen_inplace_view_shard(fm, fns_with_infos, 'Everything')
