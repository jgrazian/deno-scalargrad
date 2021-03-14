import { Scalar } from './scalar.ts';

/** Standard interface for NN type objects. */
export interface Model {
    call(this: this, x: Scalar[]): Scalar[],
    parameters(this: this): Scalar[],
    zeroGrad(this: this): void,
}

/** A single neuron.
 * 
 * Calculates sum(w * x) + b with optional ReLU.
 * All values are initialized with random values between -1.0 and 1.0.
 * 
 *      let n = new Neuron(3);
 *      let input = [new Scalar(1.0), new Scalar(-1.0), new Scalar(4.0)];
 *      let output = n.call(input);
 *      assert(output.length == 1);
 * @param nin Number of incoming connections to this Neuron
 * @param nonlin If true use ReLU during forward pass
 */
export class Neuron implements Model {
    w: Scalar[];
    b: Scalar;
    nonlin: boolean;

    constructor(nin: number, nonlin: boolean = true) {
        this.w = Array(nin).fill(0).map(() => new Scalar(Math.random() * 2 - 1));
        this.b = new Scalar(0);
        this.nonlin = nonlin;
    }

    call(this: this, x: Scalar[]): Scalar[] {
        // sum(w * x) + b
        const act = this.w.map((w, i) => w.mul(x[i])).reduce((p, c) => p.add(c)).add(this.b);
        return [(this.nonlin) ? act.relu() : act];
    }

    parameters(this: this): Scalar[] {
        return this.w.concat(this.b);
    }

    zeroGrad(this: this) {
        for (const p of this.parameters()) {
            p.grad = 0;
        }
    }

    toString(this: this): string {
        return `${(this.nonlin) ? 'ReLU' : 'Linear'} Neuron(${this.w.length})`;
    }
}

/** A collection of Neurons.
 * 
 *      let l = new Layer(3, 1);
 *      let input = [new Scalar(1.0), new Scalar(-1.0), new Scalar(4.0)];
 *      let output = l.call(input);
 *      assert(output.length == 1);
 * @param nin Number of incoming connections to each Neuron in the layer
 * @param nout Number of outputs from this layer. This is equivalent to the number of Neurons in the layer
 * @param relu If true use ReLU during forward pass
 */
export class Layer implements Model {
    neurons: Neuron[];

    constructor(nin: number, nout: number, relu: boolean = true) {
        this.neurons = Array(nout).fill(0).map(() => new Neuron(nin, relu));
    }

    call(this: this, x: Scalar[]): Scalar[] {
        const out = this.neurons.map(v => v.call(x)[0]);
        return out;
    }

    parameters(this: this): Scalar[] {
        return this.neurons.flatMap(n => n.parameters());
    }

    zeroGrad(this: this) {
        for (const p of this.parameters()) {
            p.grad = 0;
        }
    }

    toString(this: this): string {
        return `Layer of [${this.neurons.map(n => n.toString()).join(', ')}]`;
    }
}

/** A collection of Layers.
 * 
 * The example below creates a MLP with 2 inputs, 2 hidden layers both with 8 neurons and 1 output.
 * 
 *      let nn = new MLP(2, [8, 8, 1]);
 *      let input = [new Scalar(1.0), new Scalar(4.0)];
 *      let output = nn.call(input);
 *      assert(output.length == 1);
 * @param nin Number of Neurons in input layer
 * @param nouts Number of Neurons in subsequent layers
 */
export class MLP implements Model {
    layers: Layer[];

    constructor(nin: number, nouts: number[], relu: boolean = true) {
        const sz = [nin].concat(nouts);
        this.layers = Array(nouts.length).fill(0).map((_, i) => new Layer(sz[i], sz[i + 1], (i != nouts.length - 1) && relu));
    }

    call(this: this, x: Scalar[]): Scalar[] {
        for (const layer of this.layers) {
            x = layer.call(x);
        }
        return x;
    }

    parameters(this: this): Scalar[] {
        return this.layers.flatMap(n => n.parameters());
    }

    zeroGrad(this: this) {
        for (const p of this.parameters()) {
            p.grad = 0;
        }
    }

    toString(this: this): string {
        return `MLP of [${this.layers.map(l => l.toString()).join(', ')}]`;
    }
}
