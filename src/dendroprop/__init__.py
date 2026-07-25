"""dendroprop — modular, data-driven core for the two-compartment dendritic SNN.

One analytic gradient (never autodiff), organized into three levels of abstraction:
the single neuron is absent (vectorized), ``layers.TwoCompLayer`` is the hidden unit,
``network.Network`` composes hidden layers + an ``layers.ReadoutLayer`` and carries the
credit routing. Per-layer components: the surrogate and the dendritic-credit option;
network-level: the credit routing (bptt/dfa/rfa). SoA state, plus a unified dataset
interface. See docs/architecture.md.

Module map:

    config.py     NeuronConfig (fixed neuron parameters + soma_grad_scale), RunConfig
    state.py      DynamicsState, EligState, LayerCredit/LayerIngredients/TrainAccum/TrainableParams (SoA)
    dynamics.py   forward state-update equations (eqs 17-23), shared by inference + training
    surrogate.py  fast_sigmoid | ste                              (the surrogate)
    feedback.py   e_n = delta @ M; fixed random B init            (credit-routing helpers)
    credit.py     credit routing: bptt | dfa | rfa
    layers.py     TwoCompLayer, ReadoutLayer, HiddenLayer Protocol; eligibility
    network.py    Network (compose + train/infer scans, grad assembly), smooth_targets
    optim.py      hand-rolled AdamW/SGD (adam_step/sgd_step) — applies grads, outside credit
    train.py      train() -> (net, MetricsHistory); evaluate()
    io.py         write_run/load_run + save_model/load_model (run-dir artifacts)
    data/         SpikeDataset + count-bin preprocessor + adapters
"""

from .layers import TwoCompLayer, ReadoutLayer, HiddenLayer
from .network import Network, smooth_targets, build_network
from .train import train, evaluate, MetricsHistory, EpochMetrics

__all__ = [
    "TwoCompLayer", "ReadoutLayer", "HiddenLayer",
    "Network", "smooth_targets", "build_network",
    "train", "evaluate", "MetricsHistory", "EpochMetrics",
]

__version__ = "0.0.0"
