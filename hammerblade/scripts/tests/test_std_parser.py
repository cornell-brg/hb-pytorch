import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()) + '/..')
import std_parser

log = """
Emulating CUDALite...
Emulation barrier init'ed with 1 threads
PyTorch configed with 1 * 1 HB device
Emulation layer enqueued kernel tensorlib_hb_startup
  Emulation layer launched 1 threads to simulate the tile group
HB startup config kernel applied
 ATen profiler collecting ...
at top level kernel at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)
should I redispatch? 1/0
redispatching...
@#ACTUALS#@__self;[10678, 16, 4]<|>other;[10678, 16, 4]<|>
Emulation layer enqueued kernel tensorlib_add
  Emulation layer launched 1 threads to simulate the tile group
#TOP_LEVEL_FUNC#__at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar);36.904
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@CPU_LOG@;16.59
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@CPU_LOG@<|>at::Tensor at::CPUType::{anonymous}::empty(c10::IntArrayRef, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>);0.027
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@CPU_LOG@<|>at::native::add_stub::add_stub();16.515
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@HB_LOG@;15.449
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@HB_LOG@<|>at::Tensor at::HammerBladeType::{anonymous}::empty(c10::IntArrayRef, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>);0.021
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@HB_LOG@<|>at::native::add_stub::add_stub();15.31
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@HB_LOG@<|>at::native::add_stub::add_stub()<|>@OFFLOAD_KERNEL@__tensorlib_add;15.247
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@HB_LOG@<|>at::native::add_stub::add_stub()<|>@OFFLOAD_KERNEL@__tensorlib_add<|>@TRIM@;0
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>at::Tensor at::CPUType::{anonymous}::llcopy(const at::Tensor&);3.816

#TOP_LEVEL_FUNC_END#__at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)"""


def test_std_parser_1():
    actuals, stack = std_parser.parse(log)
    assert actuals == "self;[10678, 16, 4]<|>other;[10678, 16, 4]<|>"
    assert stack == """#TOP_LEVEL_FUNC#__at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar);36.904
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@CPU_LOG@;16.59
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@CPU_LOG@<|>at::Tensor at::CPUType::{anonymous}::empty(c10::IntArrayRef, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>);0.027
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@CPU_LOG@<|>at::native::add_stub::add_stub();16.515
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@HB_LOG@;15.449
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@HB_LOG@<|>at::Tensor at::HammerBladeType::{anonymous}::empty(c10::IntArrayRef, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>);0.021
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@HB_LOG@<|>at::native::add_stub::add_stub();15.31
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@HB_LOG@<|>at::native::add_stub::add_stub()<|>@OFFLOAD_KERNEL@__tensorlib_add;15.247
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@HB_LOG@<|>at::native::add_stub::add_stub()<|>@OFFLOAD_KERNEL@__tensorlib_add<|>@TRIM@;0
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>at::Tensor at::CPUType::{anonymous}::llcopy(const at::Tensor&);3.816

#TOP_LEVEL_FUNC_END#__at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)"""
