import { assertEquals } from "https://deno.land/std@0.89.0/testing/asserts.ts";

import { Neuron, Layer, MLP, loadCsv, SGD, MaxMargin, Scalar } from '../src/mod.ts';

Deno.test("NN - Neuron", () => {
    const n = new Neuron(4);

    assertEquals(n.w.length, 4);
    assertEquals(n.b.data, 0);
    assertEquals(n.toString(), 'ReLU Neuron(4)');
});

Deno.test("NN - Layer", () => {
    const l = new Layer(4, 4);

    assertEquals(l.neurons.length, 4);
    assertEquals(l.toString(), 'Layer of [ReLU Neuron(4), ReLU Neuron(4), ReLU Neuron(4), ReLU Neuron(4)]');
});

Deno.test("NN - MLP", () => {
    const nn = new MLP(2, [2, 2, 1]);

    assertEquals(nn.layers.length, 3);
    assertEquals(nn.toString(), 'MLP of [Layer of [ReLU Neuron(2), ReLU Neuron(2)], Layer of [ReLU Neuron(2), ReLU Neuron(2)], Layer of [Linear Neuron(2)]]');
});

Deno.test("NN - Train", () => {
    const data = loadCsv('./test/data/moons.csv');
    const X = data.map(l => [l[0], l[1]]);
    const y = data.map(l => l[2]);

    // 2 Inputs, 2 Hidden Layers, 2 Outputs
    const model = new MLP(2, [8, 8, 2]);

    const opt = new SGD();
    const cost = new MaxMargin();

    opt.train(model, cost, X, y, { niter: 100 });

    const fit: number[][] = [];
    for (let x = -2; x <= 2; x += 0.1) {
        for (let y = -2; y <= 2; y += 0.1) {
            const p = model.call([new Scalar(x), new Scalar(y)]);
            fit.push([x, y, p[0].data, p[1].data]);
        }
    }

    Deno.writeTextFileSync('./test/data/moon_fit.csv', fit.map(c => c.join(', ')).join('\n'));
});
