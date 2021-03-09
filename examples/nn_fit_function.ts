import { MLP, SGD, L2, Scalar } from '../src/mod.ts';

// x^2 - 2x + 1
const f = (x: number) => x * x - x * 2 + 1;

const X: number[][] = [];
const y: number[] = [];
for (let i = -1.0; i <= 1.0; i += 0.1) {
    // Give the model x and x^2 as features
    X.push([i, i * i]);
    y.push(f(i));
}

// 2 Inputs, 2 Hidden Layers with 4 neurons, 1 Output
const model = new MLP(2, [4, 4, 1], false);
console.log(model.toString())

const opt = new SGD();
const optArgs = {
    baseStep: 0.1,
    niter: 100,
}
const cost = new L2();

opt.train(model, cost, X, y, optArgs);

// Test the trained model at a bunch of points
const xTest = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5];
for (const x of xTest) {
    const yp = model.call([new Scalar(x), new Scalar(x * x)]);
    console.log(`f(${x})=${f(x)}, fp(${x})=${yp[0].data}, ${Math.abs((f(x) - yp[0].data) / f(x)) * 100}% error`);
}
