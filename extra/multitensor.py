import numpy as np
from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.helpers import Timing

d0, d1 = f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2"
N = 256
FLOPS = N * N * N * 2

class LazyBuffer:
    """
    A LazyBuffer manages tensor shards across multiple devices without unnecessary copies.
    It maintains lists for tensor shards (`st`), their realization status (`realized`), and their assigned devices (`device`).
    """
    def __init__(self, tensor, num_shards=2):
        self.num_shards = num_shards
        self.st = [tensor[:, i * (N // num_shards):(i + 1) * (N // num_shards)].to(f"{Device.DEFAULT}:{i+1}") for i in range(num_shards)]
        self.realized = [False] * num_shards
        self.device = [f"{Device.DEFAULT}:{i+1}" for i in range(num_shards)]

    def realize_all(self):
        for i, shard in enumerate(self.st):
            if not self.realized[i]:
                shard.realize()
                self.realized[i] = True

    def to_default(self):
        """
        Combine all shards back to the default device without copying.
        Assumes that shards are already realized.
        """
        combined = Tensor.cat(*self.st, dim=1).to(Device.DEFAULT)
        return combined

def explicit_shard_W_axis_1(X, W):
    # Initialize LazyBuffer for W to manage shards across devices d0 and d1
    W_buffer = LazyBuffer(W)

    # Access the shards without making copies
    Ws = W_buffer.st  # These are views/shards on d0 and d1

    # Pad them to form the correct size
    Ws = [Ws[0].pad((0, N - Ws[0].shape[1], 0, 0)), 
          Ws[1].pad((0, 0, 0, N - Ws[1].shape[1]))]

    for w in Ws:
        assert w.shape == W.shape, "Sharded weights do not match original shape after padding."

    # Ensure that the input tensors are on the correct devices
    Xs = [X.to(d0), X.to(d1)]
    for x in Xs:
        assert x.shape == X.shape, "Sharded inputs do not match original shape."

    # Realize tensors to avoid lazy evaluation affecting performance
    W_buffer.realize_all()
    for x in Xs:
        x.realize()

    def lm(x: Tensor, w: Tensor):
        # Perform local matrix multiplication on each device
        x = x.reshape(N, 1, N).expand(N, N, N)
        w = w.T.reshape(1, N, N).expand(N, N, N)
        m = x * w
        assert m.lazydata.st.views[0].mask is not None, "Mask should not be None."
        ret = m.sum(2)
        return ret

    # Perform matrix multiplication on each shard
    Os = [lm(Xs[0], Ws[0]), lm(Xs[1], Ws[1])]
    for o in Os:
        o.realize()

    # Combine the results back to the default device without additional copies
    combined_O = Tensor.cat(*[o.to(Device.DEFAULT) for o in Os], dim=1).sum(dim=1)
    return combined_O

def matmul(X, W):
    return explicit_shard_W_axis_1(X, W)
    # Alternatively, for non-sharded matmul:
    # return X @ W

if __name__ == "__main__":
    with Timing("init devices: "):
        # Initialize devices (this might trigger device allocation)
        Device[d0], Device[d1]

    with Timing("create tensors: "):
        # Initialize input tensors on the default device
        X = Tensor.kaiming_uniform(N, N).realize()
        W = Tensor.kaiming_uniform(N, N).realize()

    # Warmup can be useful for benchmarking but is commented out
    # with Timing("warmup: "):
    #     O = matmul(X, W).numpy()

    GlobalCounters.reset()
    print("******** multiply start")
    with Timing("******** multiply done: ", lambda x: f"  {FLOPS / x:.2f} GFLOPS"):
        O = matmul(X, W).realize()
        Device[Device.DEFAULT].synchronize()

    with Timing("testing: "):
        val = X.numpy() @ W.numpy()
        np.testing.assert_allclose(val, O.numpy(), atol=1e-5)
