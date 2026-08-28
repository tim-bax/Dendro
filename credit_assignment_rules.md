# Credit Assigment Rules for Dendrites.

A quick note on how credit assignment works for the Dendroprop model.

## A model summary.
First, a reminder on the model:
We have a dendritic and a somatic leaky integrator compartment. For the soma:
$$
\nu(t) = \alpha_s (1 - o(t-1)) \nu(t-1) + \sum_i^M \sum_f w^S_i \kappa(t - t_i^f)
$$
where $f$ counts the spikes arriving on input channel $i$. This can be abbreviated to
accesses of the discrete input pattern, after kernel application to an input spike train,
to:
$$
\nu(t) = \alpha_s (1 - o(t-1)) \nu(t-1) + \sum_i^M w^S_i N_i(t)
$$
$N_i(t)$ is the input of channel $i$ in time bin $t$ (in the code, a spike *count*: the
exponential kernel $\kappa$ is the membrane itself, so binning is all the input needs).
The gate $(1 - o(t-1))$ is the hard reset: after an output spike $o$ the membrane restarts
from rest.

Likewise, the dendrite integrates the same inputs, but it stops integrating while it is
producing a plateau:
$$
\mu(t) = \alpha_d \mu(t-1) + (1 - h(t-1)) \sum_i^M \sum_f w^D_i \kappa(t - t_i^f)
$$
again abbreviated to:
$$
\mu(t) = \alpha_d \mu(t-1) + (1 - h(t-1)) \sum_i^M w^D_i N_i(t)
$$

The plateau state $h(t) \in \{0,1\}$ is the dendrite's spiking non-linearity. It reads the
dendritic voltage $\mu(t')$ *latched at threshold crossing*, and stays up for a plateau
duration $\tau$ (drawn once per neuron, so plateau durations are heterogeneous across a
layer):
$$
h(t) = H(\mu(t') - \theta_d) \cdot \mathbf{1}\{0 \leq t - t' \leq \tau\}
$$
Here, $H$ is the Heaviside function and $\theta_d$ the dendritic threshold. The plateau
onset time $t'$ is frozen for as long as the plateau is up, and otherwise tracks
simulation time:
$$
t' = \begin{cases}
t' & \text{if } h(t-1) = 1 \\
t  & \text{if } h(t-1) = 0
\end{cases}
$$
In words: while no plateau is running, $t' = t$ and $\mu(t')$ is just the live dendritic
voltage, so the threshold test is the ordinary one. The moment it crosses, $t'$ (and with
it $\mu(t')$) is frozen, and $h$ stays 1 for $\tau$ time units regardless of what the
dendrite does afterwards. Note that no explicit termination condition is needed: the gate
$(1 - h(t-1))$ blocks input during the plateau and $\tau$ is much longer than the
dendritic membrane time constant, so $\mu$ has decayed away by the time the neuron looks
at it again.

The two compartments interact at the somatic non-linearity, where an active plateau
lowers the somatic threshold $\theta_s$ by $\gamma$:
$$
o(t) = H(\nu(t) + \gamma h(t) - \theta_s)
$$
so the somatic threshold is $\theta_s$ at rest and $\theta_s - \gamma$ for the duration of
a plateau.

## Credit in a LIP/LIF 2-compartment-neuron.

In the following, we will always assume the readout of a network of LIP/LIF neurons is
a non-spiking 1 compartment somatic membrane, and the average of the membrane potential
of every readout neuron enters a softmax function. The loss is the cross-entropy loss.
We will also write the derivations for single neurons, or chains of single neurons, first
as they direclty extend to the multiple neurons per layer case.
Indices are $j$ for the readout class, $n$ for a hidden neuron, and $i$ for an input
channel. The readout layer does not have a dendrite compartment, and has its own membrane
decay $\alpha_r$:
$$
\nu^R_j(t) = \alpha_r \nu^R_j(t-1) + \sum_n w^R_{j,n} o_n(t),
\qquad z_j = \frac{1}{T} \sum_t^T \nu^R_j(t)
$$
$z_j$ is the $j$-th component entering the soft-max, and with $p = \text{softmax}(z)$ and
a one-hot target $y$ the cross-entropy loss gives the top-down error signal
$$
\frac{\partial L}{\partial z_j} = p_j - y_j \equiv \delta_j
$$

### Surrogate Backpropagation through time.
The gradients of the weights of the readout layer are:
$$
\frac{\partial L}{\partial w^R_{j,n}} = \frac{\partial L}{\partial z_j}
                                        \frac{\partial z_j}{\partial \nu^R_j}
                                        \frac{\partial \nu^R_j}{\partial w^R_{j,n}}
$$
which is
$$
\frac{\partial L}{\partial w^R_{j,n}} = \frac{\delta_j}{T} \sum_{t=0}^T \underbrace{\sum_{s=0}^t \alpha_r^{t-s} o_n(s)}_{\text{pre-synaptic trace } \varepsilon^R_n(t)}
$$
Note already, that backprop through time results in a pre-synaptic trace that can be
calculated in the forward pass:
$$
\varepsilon^R_n(t) = \alpha_r \varepsilon^R_n(t-1) + o_n(t)
$$
This theme of capturing backprop through time in traces was a key contribution of the paper
'A solution to the learning dilemma for recurrent networks of spiking neurons' by Bellec et al.
and will be used throughout our approach.

The chain of derivatives from the readout to a somatic weight $w^S_{n,i}$ of a neuron $n$
in the last hidden layer $L$ is:
$$
\frac{\partial L}{\partial w^{S}_{n,i}} = \sum_j \delta_j \sum_{t=0}^T
                                       \frac{\partial z_j}{\partial o_n(t)}
                                       \frac{\partial o_n(t)}{\partial \nu_n(t)}
                                       \frac{\partial \nu_n(t)}{\partial w^S_{n,i}}
$$
The first factor is exactly
$$
\frac{\partial z_j}{\partial o_n(t)} = \frac{w^R_{j,n}}{T} \sum_{u \geq t} \alpha_r^{u-t}
                                     = \frac{w^R_{j,n}}{T} \frac{1 - \alpha_r^{T-t}}{1 - \alpha_r}
$$
i.e. a spike is credited with the whole future of the readout membrane it charges. We
drop the readout leak and use $\frac{\partial z_j}{\partial o_n(t)} \approx
\frac{w^R_{j,n}}{T}$: the discarded factor is $\frac{1}{1-\alpha_r} \approx 5.5$ for every
$t$ except the last few steps of the trial, so it is a near-uniform gain that the learning
rate absorbs. What survives is a single scalar per hidden neuron, the *learning signal*
$$
e_n = \sum_j \delta_j w^R_{j,n}
$$
The second factor is the non-differentiable $\frac{\partial o_n(t)}{\partial \nu_n(t)}$,
for which we use the fast sigmoid surrogate proposed by Zenke and Ganguli:
$$
\sigma'(x; \beta) = \frac{1}{(1 + \beta |x|)^2}
$$
evaluated at the *distance to threshold*. That distance is the dynamic one — an active
plateau has lowered it by $\gamma$ — so the plateau already shows up in the somatic
gradient:
$$
\sigma'_s(t) = \sigma'(\nu_n(t) + \gamma h_n(t) - \theta_s; \beta_s)
$$
The third factor is the pre-synaptic trace again, now at the somatic decay. Strictly, the
reset makes the recursion
$\frac{\partial \nu_n(t)}{\partial \nu_n(t-1)} = \alpha_s [1 - o_n(t-1) -
\nu_n(t-1) \sigma'_s(t-1)]$; we drop both reset terms and keep the plain leak
$\alpha_s$, which is the standard E-prop treatment. Putting the three together:
$$
\frac{\partial L}{\partial w^S_{n,i}} = \frac{e_n}{T}
        \underbrace{\sum_{t=0}^T \sigma'_s(t) \; \varepsilon^S_i(t)}_{\text{somatic eligibility}}
$$
where the pre-synaptic part is again computable on the fly during the feed-forward
computation as:
$$
\varepsilon^S_i(t) = \alpha_s \varepsilon^S_i(t-1) + N^{L-1}_i(t)
$$
with $N^{L-1}_i(t)$ the activity arriving from the layer below (the external input for the
first hidden layer, the previous layer's spikes otherwise). Note that this trace carries
no neuron index $n$: the soma integrates its input ungated, so one trace per input channel
serves the whole layer.

### Relation to E-prop.
The derivation above is E-prop, minus the parts E-prop needs and we do not. Their extra
work is (i) *recurrence*: with recurrent connections the state-to-state Jacobian is a
matrix over neurons, and the whole argument is about dropping its off-diagonal part so
that the gradient factorises into learning signal $\times$ eligibility trace — in a
feed-forward network that factorisation is exact from the start; (ii) *adaptive
thresholds*: an ALIF neuron has a second state variable, so their eligibility becomes a
two-component vector with a cross-term; and (iii) their filter notation
$\mathcal{F}_\alpha\{\cdot\}$, which is nothing but shorthand for $\sum_s \alpha^{t-s}(\cdot)$.
With a feed-forward architecture and a scalar somatic state, all three collapse and the
result is the equation above.

### below is the plan to finish the document.

Next up: dendrites, where we now have the $t'$ jump, which means that parts
of the derivative are constants for multiple time steps and attribute credit solely via
inputs and somatic feedback to the beginning to what happened at the beginning of plateaus.

(basically the same thing as the somatic weight gradient, but there is an extra surrogate,
and one of the surrogates is $t'$ linked, and the inputs are then in $t'$ time)

We can extend this soma + dendrite gradient directly to direct feedback alignment because
DFA only ever does these local gradients and then gets a direct map from $\delta$.

Update notation to vectors and matricies (matricies capital letters, vectors lower case fat printed).
(leave the one neuron examples as educational.

Then, do the "backprop through time" path through multiple layers. Here is what is happening here
(too lazy to write it all out here!):
    a) the t' path through the dendrite carries through to the prev. layer soma. that soma
       then has a t gradient path from soma, t' from dendrite. through the next dendrite, 
       this will add two paths each (t,t) (t',t) (t,t') (t',t'), and technically require either
       2^L growing traces OR keeping rolled out activity and properly BPTT-ing after the fact.
    b) "proper BPTT" might be an option to compare but not very interesting i think?
       might be inefficient
    c) the other option: we only backprop through the soma into the dendrite and into the
       prev. layer, but not from the dendrite into the prev. layer. "pruned".

This solution also admits random feedback alignment with non-local weights.

Finally, a short section listing what the code does that these equations do not.

This should finish all the methods to solve credit assignment for our model.
