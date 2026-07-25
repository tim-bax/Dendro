# Credit Assigment Rules for Dendrites.

A quick note on how credit assignment works for the Dendroprop model.

## A model summary.
First, a reminder on the model (more details in the methods pdf):
We have dendritic and somatic leaky integrator compartment. For the soma:
$$
\nu(t) = \alpha_s \nu(t-1) + \sum_i^M \sum_j^N w^S_i \kappa(t - t_i^j)
$$
which can be abbreviated to accesses of the discrete input pattern N, 
after kernel application to an input spike train, to:
$$
\nu(t) = \alpha_s \nu(t-1) + \sum_i^M w^S_i N(t)
$$

Likewise, the dendrite integrates the same inputs:
$$
\mu(t) = \alpha_d \mu(t-1) + \sum_i^M \sum_j^N w^D_i \kappa(t - t_i^j)
$$
again abbreviated to:
$$
\mu(t) = \alpha_d \mu(t-1) + \sum_i^M w^D_i N(t)
$$

The two comartment interact via their spiking non-linearities to produce
the spike train output:
$$
o(t) = H( \nu(t) - (\theta - H(\mu(t')) ) )
$$
Here, H is the Heaviside function.
The dendrite time $t'$ holds the dendrite voltage at threshold crossing for the
duration of an active dendritic plateau, reducing the spike threshold at the soma for
the same duration. When the dendrite does not generate a plateau, $t' = t$. The update
for $t'$ with plateau duration $\tau$ is:
```
if H(\mu(t')) = 0
    t' = t
elseif H(\mu(t')) = 1
    if t - t' < tau
        t' = t'
    elseif t - t' > tau
        t' = t
    end
end
```
In words: if there was no threshold crossing, or the threshold crossing is a time longer
than one plateau away, plateau time equals simulation time. If, however, the dendrite did
cross the threshold less than $\tau$ in the past, simply freeze $t'$. Thus, 
$H(\mu(t')) = 1$ for $\tau$ time units after every initial threshold crossing.

## Credit in a LIP/LIF 2-compartment-neuron.

In the following, we will always assume the readout of a network of LIP/LIF neurons is
a non-spiking 1 compartment somatic membrane, and the average of the membrane potential
of every readout neuron enters a softmax function. The loss is the cross-entropy loss.
We will also write the derivations for single neurons, or chains of single neurons, first
as they direclty extend to the multiple neurons per layer case.
$z_j$ will be the $j$-th component of the soft-max, which includes a mean over
the membrane potential of integration operation via $\frac{1}{T} \sum_t^T \nu^R_j$.
The readout layer does not have a dendrite compartment.

### Surrogate Backpropagation through time.
The gradients of the weights of the readout layer are:
$$
\frac{\partial L}{\partial w^r_{j,i}} = \frac{\partial L}{\partial z_j}
                                        \frac{\partial z_j}{\partial \nu^R_j}
                                        \frac{\partial \nu^R_j}{\partial w^R_{j,i}}
$$
which is 
$$
\frac{\partial L}{\partial w^r_{j,i}} = \frac{\del_j}{T} \sum_{t=0}^T \underbrace{\sum_{s=0}^t \alpha_s^{t-s} N(s)}_{\text{pre-synaptic eligibility}}
$$
Note already, that backprop through time results in a pre-synaptic eligibility trace can be calculated in the forward pass.
This theme of capturing backprop through time in traces was a key contribution of the paper 
'A solution to the learning dilemma for recurrent networks of spiking neurons' by Bellec et al.
and will be used throughout our approach.

The chain of derivatives from a readout to a weight $w^L_{j,i}$ of a neuron $k$ in the last hidden layer $L$ is:
$$
\frac{\partial L}{\partial w^L_{j,i} = \sum_j \frac{\del_j}{T} \sum_{t=0}^T
                                       \frac{\partial \nu^R_j}{\partial o^L_j(t)}
                                       \frac{\partial o^L_j(t)}{\partial \nu_j(t)}
                                       \frac{\partial \nu_j(t)}{\partial w_{j,i}}
$$
Here, we use the simplified sigmoid surrogate gradient proposed by Zenke and Ganguli 
to replace the non-differentiable $\frac{\partial o^L_j(t)}{\partial \nu_j(t)}$:
$$
\sigma'(\nu) = \frac{1}{1+\beta {|\nu|}^2}
$$
to derive the gradient as:
$$
\frac{\partial L}{\partial w^r_j} = \sum_j \frac{\del_j}{T}  
        \underbrace{\sum_{t=0}^T \sigma'(\nu^L_j(t)) \sum_{s=0}^t \alpha_s^{t-s} N(s)}_{\text{post-synaptic eligibility}}
$$
where we have now identified the full E-prop soma-synaptic eligibility trace.
At this point, we can note that the pre-synaptic part is computable on the fly during feed-forward computation
as:
$$
\varepsilon^{L,S}_{j,i}(t) = \alpha_s \varepsilon^{L,S}_{j,i}(t-1) + N^{L-1}_i(t)
$$

Note: Up to here it is worth cleanly double-checking against E-prop where we might have made
a mistake in our derivatoin, because the E-prop paper does a lot of work to get here and 
I don't fully get why. Includes the filter notation they use.





