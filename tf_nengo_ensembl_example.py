# Example: Training a Nengo ensembl on Tensorflow
#
#
# nengo-dl is somewhat behind the current scene and is finicky about packages/python version
# will need python 3.10 and older package versions 
# uv venv --python 3.10 nengo-dl
# uv pip install  "nengo-dl==3.6.0" "nengo==3.2.0"   "tensorflow==2.11.0" "keras==2.11.0" "numpy==1.23.5"

import matplotlib.pyplot as plt
import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

def nengo_ensembl(tau, n_neurons = 80):
    with nengo.Network(seed=42) as net:
        nengo_dl.configure_settings(trainable=False)

        stimulus = nengo.Node([0.0], label="input")
        population = nengo.Ensemble(
            n_neurons,
            dimensions=1,
            neuron_type=nengo.LIF(amplitude=0.01),
            label="LIF population",
        )
        output = nengo.Node(size_in=1, label="readout")
        nengo.Connection(stimulus, population, synapse=None)

        readout = nengo.Connection(
            population.neurons,
            output,
            transform=np.zeros((1, n_neurons)),
            synapse=None,
        )
        net.config[readout].trainable = True
        prediction_probe = nengo.Probe(output, synapse=tau)
        spike_probe = nengo.Probe(population.neurons)

    return (net, stimulus, (prediction_probe, spike_probe))


def simulate(net, dt, test_xy, probes, stimulus, train_xy):
    (x_train, y_train) = train_xy
    (x_test, y_test) = test_xy
    (prediction_probe, spike_probe) = probes
    with nengo_dl.Simulator(net, dt=dt, minibatch_size=8, device="/cpu:0") as sim:
        sim.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
            loss={prediction_probe: tf.keras.losses.MeanSquaredError()},
        )

        test_input = {stimulus: x_test}
        before = sim.predict(test_input, verbose=0)[prediction_probe]

        sim.fit(
            x={stimulus: x_train},
            y={prediction_probe: y_train},
            epochs=15,
            verbose=2,
        )

        result = sim.predict(test_input, verbose=0)
        after = result[prediction_probe]
        spikes = result[spike_probe]

    print(f"Held-out MSE before: {np.mean((before - y_test)**2):.5f}")


def test_data(n_examples, t, dt, tau, rng):
    frequency = rng.uniform(1.0, 3.0, size=(n_examples, 1))
    phase = rng.uniform(0, 2 * np.pi, size=(n_examples, 1))

    x = np.sin(2 * np.pi * frequency * t + phase)
    x = x[..., None].astype(np.float32)

    # filter along the time axis of each example independently.
    y = nengo.Lowpass(tau).filt(x**2, dt=dt, axis=1)
    return x, y.astype(np.float32)


def main():
    dt = 0.001          # one simulation step is 1 ms
    n_steps = 400
    tau = 0.020         # output filter time constant, in seconds
    t = dt * np.arange(1, n_steps + 1)
    rng = np.random.default_rng(42)

    x_train, y_train = test_data(64, t, dt, tau, rng)
    x_test, y_test = test_data(16, t, dt, tau, rng)
    print(x_train.shape, y_train.shape)  # (64, 400, 1) (64, 400, 1)


    (net, stimulus, (prediction_probe, spike_probe)) = nengo_ensembl(tau, 80)
    simulate(net, dt, (x_test, y_test), (prediction_probe, spike_probe), stimulus, (x_train, y_train))


if __name__ == "__main__":
    main()
