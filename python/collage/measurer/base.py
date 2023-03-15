import logging
import sys
from collage.utils import (is_data_tensor)
import numpy as np
from collage.utils import printe

# only collect results whose standard deviation is below this
MAX_STD_MEASURE_RTX = 5E-04
MAX_STD_MEASURE_XEON = 0.1 

# This is for operator measurement
NUM_REPEATS = 20 
NUM_MEASUREMENTS_PER_REPEAT = 100

# This is for network measurement
NUM_REPEATS_E2E = 20
NUM_MEASUREMENTS_PER_REPEAT_E2E = 100


def setup_mod_inputs(mod):
  for i in range(mod.get_num_inputs()):
    input = mod.get_input(i)
    if is_data_tensor(input):
      input_shape = input.asnumpy().shape
      logging.info(f"Data shape: {i}, {input_shape}")

      if input.dtype == "float32":
        mod.set_input(i, np.random.uniform(-1, 1, size=input_shape).astype("float32"))
      elif input.dtype == "uint8":
        mod.set_input(i, np.random.randint(0,1, size=input_shape).astype("uint8")) 
      elif input.dtype.startswith("int"):
        mod.set_input(i, np.random.randint(0,100, size=input_shape).astype(input.dtype))
      else:
        print(dir(input), input.dtype)
        assert 0
        mod.set_input(i, np.random.randint(0,100, size=input_shape).astype("long"))


# Raw performance numbers vary significantly depending on the hardware. 
# Adjust measurement option depending on the build target.
# NOTE: This is a temporary solution
def get_max_std_for_measurement(target, mean_perf):
    if "cuda" in target:
        max_std = max(MAX_STD_MEASURE_RTX, MAX_STD_MEASURE_RTX*mean_perf)
    elif "llvm" in target:
        max_std = max(MAX_STD_MEASURE_XEON, MAX_STD_MEASURE_XEON*mean_perf)
    else:
        raise Exception(f"{target} is unexpected hw, we need to set default backends for this hw.")

    return max_std

def measure(ftimer, target, *args):
    # Dummy run to check whether it runs correctly e.g., segfault due to large workspace
    import sys

    try:
        ftimer(*args)
    except Exception as E:
        #printe("It errors out when measuring; likely during op measurement")
        printe(E)
        print(sys.exc_info()[0])
        return sys.maxsize, 0

    # Warm-up Phase: Run without measurement
    # TimeEvaluator itself come with the warmup,
    # so we don't need this part technically.
    for i in range(3):
         ftimer(*args)

    mean_perf, std_perf = None, None
    # Measure performance. Continue until we get results within the max standard deviation

    # Warning(@Soo): We may want to investigate more on how to measure op perf
    # Example: in BERT, where we measure a lot of light-weighted kernel (e.g., add, multiply, relu+add),
    # we found that op perf measurement has a lot of variance within/across runtimes.
    # Plus, there is also a gap between op perf and real perf in network inference, specifically
    # for such light-weighted kernels, e.g., TensorRT op is on par with AutoTVM in op perf,
    # but it is always worse than AutoTVM in actual network inference runtime
    max_trials = 50
    i = 0
    while i<max_trials:
        perfs = np.array(ftimer(*args).results) * 1000  # convert to millisecond
        mean_perf, std_perf = np.mean(perfs), np.std(perfs)
        logging.info(f"Mean, std of perf : {mean_perf}, {std_perf}")
        i += 1

        # If mean_perf is more than 1 ms, then we should reduce threshold not to take too long,
        # e.g., BERT or Conv3D ops
        # Otherwise, we keep MAX_STD_MEASURE_RTX no matter how small the mean_perf is.
        # MAX_STD_MEASURE_RTX much of variance shouldn't matter anyway for end-to-end perf.
        threshold = get_max_std_for_measurement(target, mean_perf)
        if std_perf <= threshold:
            break

    return mean_perf, std_perf
