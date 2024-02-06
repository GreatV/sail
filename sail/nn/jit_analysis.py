import paddle
from paddle.jit import TracerWarning
import logging
import typing
import warnings
from collections import Counter
from copy import copy
from dataclasses import dataclass
from numbers import Number
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypeVar, Union
import numpy as np
from sail.common.checkpoint import _named_modules_with_dup
from .jit_handles import Handle

T = TypeVar("T", bound="JitModelAnalysis")
_IGNORED_OPS: Set[str] = {""}


@dataclass
class Statistics:
    """
    For keeping track of the various model statistics recorded during
    analysis.
    """

    counts: "Dict[str, Counter[str]]"
    unsupported_ops: "Dict[str, Counter[str]]"
    uncalled_mods: "Set[str]"


def _named_modules_without_dup(
    model: paddle.nn.Layer,
) -> Iterator[Tuple[str, paddle.nn.Layer]]:
    """
    Like .named_modules(), but the results are slightly different for
    some wrapped models.
    """
    seen = set()
    for name, mod in _named_modules_with_dup(model):
        if mod not in seen:
            seen.add(mod)
            yield name, mod


def _get_scoped_trace_graph(
    module: paddle.nn.Layer,
    inputs: Union[paddle.Tensor, Tuple[paddle.Tensor, ...]],
    aliases: Dict[Union[str, paddle.nn.Layer], str],
) -> paddle._C.Graph:
    """
    Traces the provided module using torch.jit._get_trace_graph, but adds
    submodule scope information to each graph node. The resulting graph
    is in-lined and has all model parameters treated as inputs. The input
    model has the scope name '', while its descendants have names of the
    form 'child.grandchild.grandgrandchild...'.

    Args:
        model (nn.Module) : The module to trace
        inputs (tuple) : Inputs used during the trace of the model
        aliases (dict(str or nn.Module, str) : maps modules and module
            names to the canonical name to be used as the scope for
            that module.

    Returns:
        graph (torch._C.Graph) : The pytorch JIT trace of the model
    """

    class ScopePushHook:
        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, module: paddle.nn.Layer, inputs: Any) -> Any:
            tracing_state = paddle._C._get_tracing_state()
            if tracing_state:
                tracing_state.push_scope(self.name)
            return inputs

    class ScopePopHook:
        def __call__(self, module: paddle.nn.Layer, inputs: Any, outputs: Any) -> Any:
            tracing_state = paddle._C._get_tracing_state()
            if tracing_state:
                tracing_state.pop_scope()
            return outputs

    hook_handles: List[Any] = []

    def register_hooks(mod: paddle.nn.Layer, name: str) -> None:
        prehook = mod.register_forward_pre_hook(hook=ScopePushHook(name))
        posthook = mod.register_forward_post_hook(hook=ScopePopHook())
        hook_handles.append(prehook)
        hook_handles.append(posthook)

    if isinstance(module, (paddle.DataParallel, paddle.DataParallel)):
        root_name = aliases[module]
        module = module.module
        register_hooks(module, root_name)
    for name, mod in _named_modules_without_dup(module):
        name = aliases[mod]
        register_hooks(mod, name)
    graph, _ = paddle.jit._get_trace_graph(module, inputs)
    for handle in hook_handles:
        handle.remove()
    return graph


class JitModelAnalysis:
    """
    Provides access to per-submodule model statistics obtained by
    tracing a model with pytorch's jit tracing functionality. Calculates
    a statistic on a per-operator basis using the provided set of functions
    that acts on the inputs and outputs to the operator, then aggregates
    this over modules in the model. Can return the aggregate statistic for
    any submodule in the model. Is lazily evaluated, and will perform the
    trace when a statistic is first requested. Changing the operator handles
    will cause the trace to be rerun on the next request.

    Submodules may be referred to using the module's name. The input model has
    name "", while its descendants have names of the form
    "child.grandchild.grandgrandchild...".

    An operator is treated as within the scope of a module if calling that
    module directly resulted in that operator being run. In particular,
    this means that calls to other functions owned by a module or explicit
    calls to module.forward(...) will not register resulting operators as
    contributing statistics to that module.
    """

    def __init__(
        self,
        model: paddle.nn.Layer,
        inputs: Union[paddle.Tensor, Tuple[paddle.Tensor, ...]],
    ) -> None:
        """
        Args:
            model: The model to analyze
            inputs: The inputs to the model for analysis.

        We will trace the execution of `model.forward(inputs)`. This means
        inputs have to be tensors or tuple of tensors (see
        https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace).
        In order to trace other methods or unsupported input types, you may need
        to implement a wrapper module.
        """
        self._model = model
        self._inputs = inputs
        self._op_handles: Dict[str, Handle] = {}
        self._named_modules: Dict[str, paddle.nn.Layer] = dict(
            _named_modules_with_dup(model)
        )
        self._aliases: Dict[Union[paddle.nn.Layer, str], str] = self._get_aliases(model)
        self._stats: Optional[Statistics] = None
        self._ignored_ops: Set[str] = copy(_IGNORED_OPS)
        self.unsupported_ops_warnings(True)
        self.uncalled_modules_warnings(True)
        self.tracer_warnings("no_tracer_warning")
        self.ancestor_mode("owner")

    def total(self, module_name: str = "") -> int:
        """
        Returns the total aggregated statistic across all operators
        for the requested module.

        Args:
            module_name (str) : The submodule to get data for. Defaults to
                the entire model.
        Returns:
            int : The aggregated statistic.
        """
        stats = self._analyze()
        module_name = self.canonical_module_name(module_name)
        total_count = sum(stats.counts[module_name].values())
        return total_count

    def by_operator(self, module_name: str = "") -> typing.Counter[str]:
        """
        Returns the statistics for a requested module, grouped by operator
        type. The operator handle determines the name associated with each
        operator type.

        Args:
            module_name (str) : The submodule to get data for. Defaults
                to the entire model.
        Returns:
            Counter(str) : The statistics for each operator.
        """
        stats = self._analyze()
        module_name = self.canonical_module_name(module_name)
        return stats.counts[module_name]

    def by_module_and_operator(self) -> Dict[str, typing.Counter[str]]:
        """
        Returns the statistics for all submodules, separated out by
        operator type for each submodule. The operator handle determines
        the name associated with each operator type.

        Returns:
            dict(str, Counter(str)):
                The statistics for each submodule and each operator.
                Grouped by submodule names, then by operator name.
        """
        stats = self._analyze()
        return stats.counts

    def by_module(self) -> typing.Counter[str]:
        """
        Returns the statistics for all submodules, aggregated over
        all operators.

        Returns:
            Counter(str): statistics counter grouped by submodule names
        """
        stats = self._analyze()
        summed_counts = Counter()
        for mod, results in stats.counts.items():
            summed_counts[mod] = sum(results.values())
        return summed_counts

    def unsupported_ops(self, module_name: str = "") -> typing.Counter[str]:
        """
        Lists the number of operators that were encountered but unsupported
        because no operator handle is available for them. Does not include
        operators that are explicitly ignored.

        Args:
            module_name (str) : The submodule to list unsupported ops.
                Defaults to the entire model.

        Returns:
            Counter(str) : The number of occurences each unsupported operator.
        """
        if self._stats is None:
            raise RuntimeError(
                "Analysis results should be computed before calling unsupported_ops()"
            )
        module_name = self.canonical_module_name(module_name)
        return self._stats.unsupported_ops[module_name]

    def uncalled_modules(self) -> Set[str]:
        """
        Returns a set of submodules that were never called during the
        trace of the graph. This may be because they were unused, or
        because they were accessed via direct calls .forward() or with
        other python methods. In the latter case, statistics will not be
        attributed to the submodule, though the statistics will be included
        in the parent module.

        Returns:
            set(str) : The set of submodule names that were never called
                during the trace of the model.
        """
        stats = self._analyze()
        return stats.uncalled_mods

    def set_op_handle(self, *args, **kwargs: Optional[Handle]) -> "JitModelAnalysis":
        """
        Sets additional operator handles, or replaces existing ones.

        Args:
            args: (str, Handle) pairs of operator names and handles.
            kwargs: mapping from operator names to handles.

        If a handle is ``None``, the op will be explicitly ignored. Otherwise,
        handle should be a function that calculates the desirable statistic
        from an operator. The function must take two arguments, which are the
        inputs and outputs of the operator, in the form of ``list(torch._C.Value)``.
        The function should return a counter object with per-operator statistics.

        Examples
        ::
            handlers = {"aten::linear": my_handler}
            counter.set_op_handle("aten::matmul", None, "aten::bmm", my_handler2)
                   .set_op_handle(**handlers)
        """
        self._stats = None
        if len(args) % 2 != 0:
            raise TypeError(
                "set_op_handle should be called with pairs of names and handles!"
            )
        for name, handle in zip(args[::2], args[1::2]):
            kwargs[name] = handle
        for name, handle in kwargs.items():
            if handle is None:
                self._ignored_ops.add(name)
            else:
                self._op_handles[name] = handle
        return self

    def clear_op_handles(self) -> "JitModelAnalysis":
        """
        Clears all operator handles currently set.
        """
        self._op_handles = {}
        self._ignored_ops = copy(_IGNORED_OPS)
        self._stats = None
        return self

    def canonical_module_name(self, name: str) -> str:
        """
        Returns the canonical module name of the given ``name``, which might be
        different from the given ``name`` if the module is shared.
        This is the name that will be used as a key when statistics are
        output using .by_module() and .by_module_and_operator().

        Args:
            name (str) : The name of the module to find the canonical name for.
        Returns:
            str : The canonical name of the module.
        """
        assert isinstance(name, str), "Module name must be a string."
        if name in self._aliases:
            return self._aliases[name]
        else:
            raise KeyError(
                "Requested module name is not among the descendants of the analyzed model."
            )

    def copy(
        self,
        new_model: Optional[paddle.nn.Layer] = None,
        new_inputs: Union[None, paddle.Tensor, Tuple[paddle.Tensor, ...]] = None,
    ) -> "JitModelAnalysis":
        """
        Returns a copy of the :class:`JitModelAnalysis` object, keeping all
        settings, but on a new model or new inputs.

        Args:
            new_model (nn.Module or None) : a new model for the new
                JitModelAnalysis. If None, uses the original model.
            new_inputs (typing.Tuple[object, ...] or None) : new inputs
                for the new JitModelAnalysis. If None, uses the original
                inputs.
        Returns:
            JitModelAnalysis : the new model analysis object
        """
        model = self._model if new_model is None else new_model
        inputs = self._inputs if new_inputs is None else new_inputs
        return (
            JitModelAnalysis(model=model, inputs=inputs)
            .set_op_handle(**self._op_handles)
            .unsupported_ops_warnings(self._enable_warn_unsupported_ops)
            .uncalled_modules_warnings(self._enable_warn_uncalled_mods)
            .tracer_warnings(self._warn_trace)
        )

    def tracer_warnings(self: T, mode: str) -> T:
        """
        Sets which warnings to print when tracing the graph to calculate
        statistics. There are three modes. Defaults to 'no_tracer_warning'.
        Allowed values are:

        * 'all' : keeps all warnings raised while tracing
        * 'no_tracer_warning' : suppress torch.jit.TracerWarning only
        * 'none' : suppress all warnings raised while tracing

        Args:
            mode (str) : warning mode in one of the above values.
        """
        if mode not in ["all", "no_tracer_warning", "none"]:
            raise ValueError(f"Unrecognized tracer warning mode {mode}.")
        self._warn_trace = mode
        return self

    def ancestor_mode(self: T, mode: str) -> T:
        """
        Sets how to determine the ancestor modules of an operator. Must be one of
        "owner" or "caller".

        * "caller": an operator belongs to all modules that is currently executing
          `forward()` at the time the operator is called.
        * "owner": an operator belongs to the last module that's executing
          `forward()` at the time the operator is called, plus this module's recursive
          parents.  If an module has multiple parents (e.g. a shared module), only one
          will be picked.

        For most cases, a module only calls submodules it owns, so both options would
        work identically. In certain edge cases, this option will affect the hierarchy
        of results, but won't affect the total count.
        """
        if mode not in ["owner", "caller"]:
            raise ValueError(f"Unrecognized ancestor mode: {mode}")
        self._ancestor_mode = mode
        return self

    def unsupported_ops_warnings(self: T, enabled: bool) -> T:
        """
        Sets if warnings for unsupported operators are shown. Defaults
        to True. Counts of unsupported operators may be obtained from
        :meth:`unsupported_ops` regardless of this setting.

        Args:
            enabled (bool) : Set to 'True' to show unsupported operator
                warnings.
        """
        self._enable_warn_unsupported_ops = enabled
        return self

    def uncalled_modules_warnings(self: T, enabled: bool) -> T:
        """
        Sets if warnings from uncalled submodules are shown. Defaults to true.
        A submodule is considered "uncalled" if it is never called during
        tracing. This may be because it is actually unused, or because it is
        accessed via calls to ``.forward()`` or other methods of the module.
        The set of uncalled modules may be obtained from
        :meth:`uncalled_modules` regardless of this setting.

        Args:
            enabled (bool) : Set to 'True' to show warnings.
        """
        self._enable_warn_uncalled_mods = enabled
        return self

    def _warn_unsupported_ops(self, ops: typing.Counter[str]) -> None:
        if not self._enable_warn_unsupported_ops:
            return
        logger = logging.getLogger(__name__)
        for op, freq in ops.items():
            logger.warning(
                "Unsupported operator {} encountered {} time(s)".format(op, freq)
            )

    def _warn_uncalled_mods(self, uncalled_mods: Set[str]) -> None:
        if not self._enable_warn_uncalled_mods:
            return
        uncalled_mods = {x for x in uncalled_mods if self._has_forward(x)}
        if len(uncalled_mods) == 0:
            return
        logger = logging.getLogger(__name__)
        logger.warning(
            """The following submodules of the model were never called during the trace of the graph. They may be unused, or they were accessed by direct calls to .forward() or via other python methods. In the latter case they will have zeros for statistics, though their statistics will still contribute to their parent calling module.
"""
            + ", ".join(sorted(uncalled_mods))
        )

    def _get_aliases(
        self, model: paddle.nn.Layer
    ) -> Dict[Union[str, paddle.nn.Layer], str]:
        aliases = {}
        for name, module in _named_modules_with_dup(model):
            if module not in aliases:
                aliases[module] = name
            aliases[name] = aliases[module]
            if "/" in name:
                sub_name = name.split("/")[-1]
                aliases[sub_name] = aliases[module]
        return aliases

    def _get_all_ancestors(self, module_name: str) -> Set[str]:
        """
        Get all ancestors of the given module, defined by ownership.
        If the given module has multiple owners, use its canonical name.
        """
        parts = self.canonical_module_name(module_name).split(".")
        res = {""}
        for k in range(len(parts) + 1):
            res.add(".".join(parts[:k]))
        return res

    def _analyze(self) -> "Statistics":
        stats = self._stats
        if stats is not None:
            return stats
        with warnings.catch_warnings():
            if self._warn_trace == "none":
                warnings.simplefilter("ignore")
            elif self._warn_trace == "no_tracer_warning":
                warnings.filterwarnings("ignore", category=TracerWarning)
            graph = _get_scoped_trace_graph(self._model, self._inputs, self._aliases)
        counts = {}
        unsupported_ops = {}
        for _, mod in _named_modules_with_dup(self._model):
            name = self._aliases[mod]
            counts[name] = Counter()
            unsupported_ops[name] = Counter()
        all_seen = set()
        for node in graph.nodes():
            kind = node.kind()
            if kind == "prim::PythonOp":
                kind = kind + "." + node.pyname()
            scope_names = node.scopeName().split("/")
            all_seen.update(scope_names)
            if self._ancestor_mode == "caller":
                ancestors = set(scope_names)
            else:
                ancestors = self._get_all_ancestors(scope_names[-1])
                all_seen.update(ancestors)
            if kind not in self._op_handles:
                if self._should_ignore_node(node):
                    continue
                for name in ancestors:
                    unsupported_ops[name][kind] += 1
            else:
                inputs, outputs = list(node.inputs()), list(node.outputs())
                op_counts = self._op_handles[kind](inputs, outputs)
                if isinstance(op_counts, Number):
                    op_counts = Counter({self._simplify_op_name(kind): op_counts})
                for v in op_counts.values():
                    if not isinstance(v, (int, float, np.float64, np.int64)):
                        raise ValueError(
                            f"Invalid type {type(v)} for the flop count! Please use a wider type to avoid overflow."
                        )
                for name in ancestors:
                    counts[name] += op_counts
        uncalled_mods = set(self._aliases.values()) - all_seen
        stats = Statistics(
            counts=counts, unsupported_ops=unsupported_ops, uncalled_mods=uncalled_mods
        )
        self._stats = stats
        self._warn_unsupported_ops(unsupported_ops[""])
        self._warn_uncalled_mods(uncalled_mods)
        return stats

    def _simplify_op_name(self, full_op_name: str) -> str:
        """
        Get simplified name of the op without the preceding namespace, e.g.
        aten::batch_norm -> batch_norm
        """
        p = full_op_name.find("::")
        if p != -1:
            return full_op_name[p + 2 :]
        else:
            return full_op_name

    def _has_forward(self, mod_name: str) -> bool:
        module = self._named_modules.get(mod_name)
        if module is None:
            return False
        module_type = type(module)
        no_forward_mods = {
            paddle.nn.LayerList,
            paddle.nn.LayerDict,
            paddle.nn.Layer,
            paddle.nn.Identity,
        }
        for mod in no_forward_mods:
            if module_type.forward is mod.forward:
                return False
        return True

    def _should_ignore_node(self, node) -> bool:
        kind = node.kind()
        if kind in self._ignored_ops:
            return True
        if kind.startswith("prim::PythonOp") or kind.startswith("prim::CallFunction"):
            return False
        if kind.startswith("prim::"):
            return True
        return False
