import tvm._ffi
import tvm.driver
# import tvm.relay.testing as testing
# from tvm import relay

# from ..backend_operator.record import backendop_lib
from ..backend_operator.op_config import MeasuredConfigs
from ..backend_operator.target import Target
from ..backend_operator.op_type import OpType
from ..backend_operator.backend_op_lib import BackendOpLib
from ..backend_operator.utils import *

#from ..utility.visualize import visualize_network
from ..utility.profile_ops_in_net import profile_ops_in_net
from .optimizer_utils import print_matching_final

from .comp_graph import ComputationGraph
from .comp_graph_optimizer import *
from .ext_compiler_op_merger import *
from .backend_op_state import *
from .evolutionary_searcher import EvolutionarySearcher
from .op_match_logger import *

def setup_backend_op_lib(network_expr, targets, batch_size):
    backendop_lib = BackendOpLib.get()
    # backendop_lib.measure_backend_ops(network_expr, targets, batch_size)

    return backendop_lib

@tvm._ffi.register_func("relay.transform.optimizer.get_user_fusion")
def get_user_fusion(relay_expr):
    print("User-defined fusion", file=sys.stderr)
    relay_expr = get_function_body(relay_expr)
    opt_match = OpMatchReader().read(relay_expr)
    return opt_match

def run_op_level_opt(relay_expr):
    relay_expr = get_function_body(relay_expr)

    comp_graph = ComputationGraph(relay_expr)
    n_relay_nodes = comp_graph.n_relay_nodes
    print(f"# of relay nodes in comp graph: {n_relay_nodes}")

    # Sanity check: Only AutoTVM
    targets = [Target.TVM_GPU_AUTOTVM]

    # Sanity check: Enable all backends except for TensorRT
    # targets = [Target.TVM_GPU_AUTOTVM, Target.CUDNN, Target.CUBLAS]

    batch_size = 1
    backendop_lib = setup_backend_op_lib(relay_expr, targets, batch_size)

    # Optimizing graph
    print("Computation graph created")
    optimizer = CompGraphOptimizer(backendop_lib, targets)
    print("Optimizer created")
    optimizer.optimize(comp_graph)
    print("It's optimized")
    optimized_match, post_order_match_result = optimizer.get_optimized_match(comp_graph)

    # print("Match result: ", optimized_match)
    # post_order_match_result is for debugging to check if it matches the final result from the TVM DP fusion pass
    print("Match result")
    for idx, pair in enumerate(post_order_match_result):
        print(idx, pair)
    # Debug (@soo)
    print_matching_final(comp_graph, optimizer.loc2match)
    print("-"*40)

    return optimized_match, relay_expr, backendop_lib, n_relay_nodes


@tvm._ffi.register_func("relay.transform.optimizer.run_two_level_opt")
def run_two_level_opt(relay_expr):
    """
    Two-level optimization pass
    First level is for backend operators from TVM and CuDNN, which are not external compilers.
    Second level is for external compilers such as TensorRT.
    We have two separate levels because
    1) First level remove need to enumerate all possible extenral compiler patterns
    2) First level provide us with optimal solutions for backend operators.
    Thus, for each backend operators, second level only needs to consider between
    TensorRT operators and chosen optimal backend operators from first level.

    Parameters
    ----------
    relay_expr : tvm.relay.expr
        Relay IR for computation graph

    Returns
    -------
    matched_relay_expr : tvm.relay.expr
        The result matching between backend operators and Relay operators
    """

    # It is a function if you get it from last pass of Relay build
    print("[Python side] Run two-level optimization")

    # op-level optimization: DP with all backends but external compilers, e.g., TensorRT
    optimized_match, relay_expr, backendop_lib, n_relay_nodes = run_op_level_opt(relay_expr)
    print("[Python side] Op-level optimization is done")

    # Debug
    # OpMatchLogger().save(relay_expr, optimized_match)
    # print(OpMatchReader().read(relay_expr))

    # subgraph-level optimization for external compilers
    # Translate optimized_match into OpState class that can be used for evolutionary search
    state_id_to_exprs_anno = MatchToOpStateTranslator().translate(relay_expr, optimized_match)

    # Prepare OpStateToMatchTranslator
    # This translates Op State to optimized_match with the following two dictionaries
    op_state_to_match_translator = OpStateToMatchTranslator(optimized_match, state_id_to_exprs_anno)

    # Run evolutionary search
    n_ops = len(state_id_to_exprs_anno.keys())
    print(f"# of matched operators after first level : {n_ops}")

    # Warning(@soo): Network name is hardcoded for now. We can fix it later
    net_name = "conv2d+relu_x2"
    ev_searcher = EvolutionarySearcher(op_state_to_match_translator, relay_expr, net_name, n_ops=n_ops,
                                       pop_size=5, max_iter=1)
    second_opt_match = ev_searcher.search(rnd_seed=64)

    # print(f"fusion dic (before merge): {optimized_match}")
    # optimized_match = ExtCompilerOpMerger(optimized_match).merge(relay_expr)
    # print(f"fusion dic (after  merge): {optimized_match}")

    backendop_lib.save_to_log()

    return second_opt_match

@tvm._ffi.register_func("relay.transform.optimizer.run_dp")
def run_dp(relay_expr):
    """Optimizing pass for computation graph representation (Relay IR).

    Parameters
    ----------
    relay_expr : tvm.relay.expr
        Relay IR for computation graph

    Returns
    -------
    matched_relay_expr : tvm.relay.expr
        The result matching between backend operators and Relay operators
    """

    # It is a function if you get it from last pass of Relay build
    print("Optimizing on the Python side")
    # print("Relay expression")
    # print(relay_expr)
    # profile_ops_in_net(relay_expr, "bert", "tensorrt")
    # import sys
    # sys.exit(0)
    # visualize_network(relay_expr, "o3_bert")
    relay_expr = get_function_body(relay_expr)

    comp_graph = ComputationGraph(relay_expr)

    # Warning: ResNet-8 doesn't have tuned operators / CuDNN doesn't work for ResNet-8
    # target_backend = None # Consider all targets

    # Sanity check: AutoTVM
    targets = [Target.TVM_GPU_AUTOTVM]

    # Sanity check: Only CuDNN
    # targets = [Target.TVM_GPU_AUTOTVM, Target.CUDNN]

    # Enable all backends except for CuDNN
    # targets = [Target.TVM_GPU_AUTOTVM, Target.CUBLAS, Target.TENSORRT]

    # Enable all backends
    # targets = [Target.TVM_GPU_AUTOTVM, Target.CUBLAS, Target.CUDNN, Target.TENSORRT]

    batch_size = 1
    backendop_lib = setup_backend_op_lib(relay_expr, targets, batch_size)

    # Optimizing graph
    print("Computation graph created")
    optimizer = CompGraphOptimizer(backendop_lib, targets)
    print("Optimizer created")
    optimizer.optimize(comp_graph)
    print("It's optimized")
    optimized_match, post_order_match_result = optimizer.get_optimized_match(comp_graph)

    # print("Match result: ", optimized_match)
    # post_order_match_result is for debugging to check if it matches the final result from the TVM DP fusion pass
    print("Match result")
    for idx, pair in enumerate(post_order_match_result):
        print(idx, pair)
    # Debug (@soo)
    print_matching_final(comp_graph, optimizer.loc2match)
    print("-"*40)
    # print("Optimized match")
    # print(optimized_match)

    # print(f"fusion dic (before merge): {optimized_match}")
    # optimized_match = ExtCompilerOpMerger(optimized_match).merge(relay_expr)
    # print(f"fusion dic (after  merge): {optimized_match}")

    backendop_lib.save_to_log()

    return optimized_match

"""
This is still work in progress.

Pros: it gives us upper bound
Cons: How do you consider all possible TensorRT operators? I don't have good answers to that yet.

"""
@tvm._ffi.register_func("relay.transform.optimizer.run_exhaustive_search")
def run_exhaustive_search(relay_expr):
    # It is a function if you get it from last pass of Relay build
    print("Optimizing on the Python side")
    # print("Relay expression")
    # print(relay_expr)
    # profile_ops_in_net(relay_expr, "bert", "tensorrt")
    # import sys
    # sys.exit(0)
    # visualize_network(relay_expr, "o3_bert")
    relay_expr = get_function_body(relay_expr)

    comp_graph = ComputationGraph(relay_expr)

    # Warning: ResNet-8 doesn't have tuned operators / CuDNN doesn't work for ResNet-8
    # target_backend = None # Consider all targets

    # Sanity check: AutoTVM
    # targets = [Target.TVM_GPU_AUTOTVM]

    # Sanity check: Only CuDNN
    # targets = [Target.TVM_GPU_AUTOTVM, Target.CUDNN]

    # Enable all backends
    targets = [Target.TVM_GPU_AUTOTVM, Target.CUBLAS, Target.CUDNN, Target.TENSORRT]
    batch_size = 1
    backendop_lib = setup_backend_op_lib(relay_expr, targets, batch_size)

    # Optimizing graph
    print("Computation graph created")
    optimizer = ExhaustiveSearcher(backendop_lib, targets)
    print("Optimizer created")
    optimizer.optimize(comp_graph)
    print("It's optimized")
    optimized_match, post_order_match_result = optimizer.get_optimized_match(comp_graph)

    # print("Match result: ", optimized_match)
    # post_order_match_result is for debugging to check if it matches the final result from the TVM DP fusion pass
    print("Match result")
    for idx, pair in enumerate(post_order_match_result):
        print(idx, pair)
    # Debug (@soo)
    print_matching_final(comp_graph, optimizer.loc2match)
    print("-" * 40)
    # print("Optimized match")
    # print(optimized_match)

    # print(f"fusion dic (before merge): {optimized_match}")
    # optimized_match = ExtCompilerOpMerger(optimized_match).merge(relay_expr)
    # print(f"fusion dic (after  merge): {optimized_match}")

    backendop_lib.save_to_log()

    return optimized_match

# # Test script with ResNet-8
# if __name__ == "__main__":
#     batch_size = 1
#     num_class = 1000

#     # Resnet-8
# #     image_shape = (3, 28, 28)
# #     mod, params = testing.resnet.get_workload(num_layers=8, batch_size=batch_size, image_shape=image_shape)
# #     relay_expr = mod["main"].body

#     # Chain graph
#     out_channels = 16
#     batch_size = 1

#     data = relay.var("data", relay.TensorType((batch_size, 3, 224, 224), "float32"))
#     conv_weight = relay.var("weight")
#     dense_weight = relay.var("weight")
#     bn_gamma = relay.var("bn_gamma")
#     bn_beta = relay.var("bn_beta")
#     bn_mmean = relay.var("bn_mean")
#     bn_mvar = relay.var("bn_var")

#     simple_net = relay.nn.conv2d(
#         data=data, weight=conv_weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
#     )
# #     simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
# #     simple_net = relay.nn.relu(simple_net)
# #     simple_net = relay.nn.relu(simple_net)
#     relay_expr = simple_net

#     optimize_comp_graph(relay_expr)

# Deprecated get_user_fusion
# fusion_dic[relay_expr] = "0-tvmgpu-autotvm_relu"  # Relu
# fusion_dic[relay_expr.args[0]] = "1-tvmgpu-autotvm_conv2d"  # Conv
# fusion_dic[relay_expr.args[0].args[0]] = "1-tvmgpu-autotvm_conv2d"  # Relu
# fusion_dic[relay_expr.args[0].args[1]] = "1-tvmgpu-autotvm_conv2d"  # Param
# fusion_dic[relay_expr.args[0].args[0].args[0]] = "3-cudnn_conv2d"  #
# fusion_dic[relay_expr.args[0].args[0].args[0].args[1]] = "3-cudnn_conv2d"  # Param
# fusion_dic[relay_expr.args[0].args[0].args[0].args[0]] = "3-cudnn_conv2d"  # Data

# fusion_dic[relay_expr] = "0-tvmgpu-no-tuning_relu" # Relu
# fusion_dic[relay_expr.args[0]] = "1-tvmgpu-no-tuning_conv2d" # Conv
# fusion_dic[relay_expr.args[0].args[0]] = "2-tvmgpu-no-tuning_relu"  # Relu
# fusion_dic[relay_expr.args[0].args[1]] = "1-tvmgpu-no-tuning_conv2d" # Param
# fusion_dic[relay_expr.args[0].args[0].args[0]] = "3-tvmgpu-no-tuning_conv2d"  #
# fusion_dic[relay_expr.args[0].args[0].args[0].args[1]] = "3-tvmgpu-no-tuning_conv2d" # Param
# fusion_dic[relay_expr.args[0].args[0].args[0].args[0]]= "3-tvmgpu-no-tuning_conv2d" # Data

# Enable External compiler merging or not
# print(f"fusion dic (before merge): {fusion_dic}")
# fusion_dic = ExtCompilerOpMerger(fusion_dic).merge(relay_expr)
# print(f"fusion dic (after  merge): {fusion_dic}")
