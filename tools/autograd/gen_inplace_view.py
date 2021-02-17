import itertools
from typing import Optional, List, Sequence, Union

from tools.codegen.api.types import *
import tools.codegen.api.cpp as cpp
from tools.codegen.code_template import CodeTemplate
from tools.codegen.context import with_native_function
from tools.codegen.utils import mapMaybe
from tools.codegen.gen import parse_native_yaml, FileManager
from tools.codegen.model import *


def tie_return_values(f: NativeFunction) -> str:
    if len(f.func.returns) == 1:
        return f'auto {f.func.returns[0].name or "result"}'
    names = cpp.return_names(f)
    return f'std::tie({", ".join(names)})'

def get_return_value(f: NativeFunction) -> str:
    names = cpp.return_names(f)
    if len(f.func.returns) == 1:
        return names[0]
    if f.func.kind() == SchemaKind.out:
        return f'std::forward_as_tuple({", ".join(names)})'
    else:
        moved = ", ".join(f'std::move({name})' for name in names)
        return f'std::make_tuple({moved})'

INPLACE_VIEW_DISPATCH = CodeTemplate("""\
${assign_return_values}at::redispatch::${api_name}(${unpacked_args});""")

def emit_inplace_view_body(f: NativeFunction) -> List[str]:
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

    assign_return_values = f'{tie_return_values(f)} = ' \
                           if f.func.kind() == SchemaKind.functional and f.func.returns else ''

    inplace_view_body.append(INPLACE_VIEW_DISPATCH.substitute(
        assign_return_values=assign_return_values,
        api_name=api_name,
        unpacked_args=redispatch_args,
    ))
    for r in cpp.return_names(f):
        inplace_view_body.append(f'torch::autograd::increment_version({r});')

    if f.func.returns:
        inplace_view_body.append(f'return {get_return_value(f)};')
    return inplace_view_body

METHOD_DEFINITION = CodeTemplate("""\
${return_type} ${type_wrapper_name}(${formals}) {
  ${type_definition_body}
}
""")

def type_wrapper_name(f: NativeFunction) -> str:
    if f.func.name.overload_name:
        return f'{cpp.name(f.func)}_{f.func.name.overload_name}'
    else:
        return cpp.name(f.func)


def modifies_arguments(f: NativeFunction) -> bool:
    return f.func.kind() in [SchemaKind.inplace, SchemaKind.out]

@with_native_function
def method_definition(f: NativeFunction) -> Optional[str]:
    if not modifies_arguments(f):
        return None

    formals = ', '.join(
        # code-generated tracing kernels plumb and recompute dispatch keys directly through the kernel for performance.
        # See Note [Plumbing Keys Through The Dispatcher] for details.
        ['c10::DispatchKeySet ks'] +
        [f'{cpp.argument_type(a, binds="__placeholder__").cpp_type()} {a.name}'
            for a in f.func.schema_order_arguments()]
    )

    return METHOD_DEFINITION.substitute(
        return_type=cpp.returns_type(f.func.returns),
        type_wrapper_name=type_wrapper_name(f),
        formals=formals,
        type_definition_body=emit_inplace_view_body(f),
    )

WRAPPER_REGISTRATION = CodeTemplate("""\
m.impl("${name}",
       TORCH_FN(${class_type}::${type_wrapper_name})
);
""")

@with_native_function
def method_registration(f: NativeFunction) -> Optional[str]:
    if not modifies_arguments(f):
        return None

    return WRAPPER_REGISTRATION.substitute(
        name=f.func.name,
        type_wrapper_name=type_wrapper_name(f),
        class_type='InplaceView',
    )

def gen_inplace_view_shard(
    fm: FileManager, native_functions: Sequence[NativeFunction], suffix: str
) -> None:
    fm.write_with_template('InplaceView%s.cpp' % suffix, 'InplaceView.cpp', lambda: {
        'generated_comment': '@' + f'generated from {fm.template_dir}/InplaceView.cpp',
        'inplace_view_method_definitions': list(mapMaybe(method_definition, native_functions)),
        'inplace_view_wrapper_registrations': list(mapMaybe(method_registration, native_functions)),
    })

def gen_inplace_view(out: str, native_yaml_path: str, template_path: str) -> None:
    # NOTE: see Note [Sharded File] at the top of the VariableType.cpp
    # template regarding sharding of the generated files.
    num_shards = 2
    shards: List[List[NativeFunction]] = [[] for _ in range(num_shards)]

    # functions are assigned arbitrarily but stably to a file based on hash
    native_functions = list(sorted(parse_native_yaml(native_yaml_path), key=lambda f: cpp.name(f.func)))
    for f in native_functions:
        x = sum(ord(c) for c in cpp.name(f.func)) % num_shards
        shards[x].append(f)

    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    for i, shard in enumerate(shards):
        gen_inplace_view_shard(fm, shard, '_%d' % i)
    gen_inplace_view_shard(fm, native_functions, 'Everything')
