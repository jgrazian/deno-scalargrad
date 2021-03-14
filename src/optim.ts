import { Scalar } from './scalar.ts';
import { Model } from './nn.ts';
import { randomPermutation } from './utils.ts';

/** Standard interface for loss functions. */
interface LossFunction {
    loss(model: Model, X: number[][], y: number[], batchSize?: number): [Scalar, number]
}

interface OptArgs {
    baseStep?: number,
    niter?: number,
    batchSize?: number,
}

/** Standard interface for optimizers. */
interface Optimizer {
    train(model: Model, loss: LossFunction, X: number[][], y: number[], args?: OptArgs): void
}

/** Stochastic Gradinet Descent Optimizer */
export class SGD implements Optimizer {
    train(model: Model, lf: LossFunction, X: number[][], y: number[], args?: OptArgs): void {
        args = args || {};
        const baseStep = args.baseStep || 1.0;
        const niter = args.niter || 100;
        const batchSize = args.batchSize || undefined;

        for (let i = 0; i < niter; i++) {
            // Forward
            const [totalLoss, acc] = lf.loss(model, X, y, batchSize);

            // Backward
            model.zeroGrad();
            totalLoss.backward();

            // Gradient Descent
            const learningRate = baseStep * (1.0 - 0.9 * i / niter);
            for (const p of model.parameters()) {
                p.data -= learningRate * p.grad;
            }
            if (i % 1 == 0) {
                console.log(`step ${i} loss ${totalLoss.data}, accuracy ${acc * 100}%`);
            }
        }
    }
}

/** Support Vector Machine (SVM) loss function
 * 
 * Used for multiclassification.
 * @param X Features. X[i] gets a single sample. X[i][j] gets a single feature from a
 *  single sample. X[0].length should be equal to the number of inputs to the model
 * @param y Labels. Should contain m integers valued -1 or 1 where m = X.length
 */
export class MaxMargin implements LossFunction {
    loss(model: Model, X: number[][], y: number[], batchSize?: number): [Scalar, number] {
        const ri = randomPermutation(X.length - 1).slice(0, batchSize);

        const Xb: number[][] = (!batchSize) ? X : ri.map(v => X[v]);
        const yb: number[] = (!batchSize) ? y : ri.map(v => y[v]);

        // Get training inputs as [Scalar, Scalar][]
        const inputs = Xb.map(xy => xy.map(v => new Scalar(v)));

        // Feed forward
        const scores = <Scalar[][]>inputs.map(xy => model.call(xy));

        // SVM max-margin loss ReLu(sp - st + 1)
        let accuracy = 0;
        const losses = scores.map(s => new Scalar(1.0));
        for (let n = 0; n < scores.length; n++) {
            const yi = yb[n]; // Expected
            const pi = scores[n][0]; // Predicition

            losses[n] = losses[n].add(pi.mul(-yi)).relu();

            if ((yi > 0.0) == (pi.data > 0.0)) {
                accuracy += 1;
            }
        }

        accuracy /= losses.length;
        const dataLoss = losses.reduce((p, c) => p.add(c)).mul(1 / losses.length);

        // L2 Reg
        const alpha = 1e-4;
        const regLoss = model.parameters().map(p => p.pow(2)).reduce((p, c) => p.add(c)).mul(alpha);

        const totalLoss = dataLoss.add(regLoss);

        return [totalLoss, accuracy];
    }
}

/** Mean Square Error (L2) loss function
 * 
 * Used for regression.
 * @param X Features. X[i] gets a single sample. X[i][j] gets a single feature from a
 *  single sample. X[0].length should be equal to the number of inputs to the model
 * @param y Labels. Should contain m numbers where m = X.length
 */
export class L2 implements LossFunction {
    loss(model: Model, X: number[][], y: number[], batchSize?: number): [Scalar, number] {
        const ri = randomPermutation(X.length - 1).slice(0, batchSize);

        const Xb: number[][] = (!batchSize) ? X : ri.map(v => X[v]);
        const yb: number[] = (!batchSize) ? y : ri.map(v => y[v]);

        // Get training inputs as Scalar[][]
        const inputs = Xb.map(xy => xy.map(v => new Scalar(v)));

        // Feed forward
        const scores = <Scalar[][]>inputs.map(xy => model.call(xy));

        // L2 loss (si - sp)^2
        let accuracy = 0;
        const losses = scores.map(s => new Scalar(0));
        for (let n = 0; n < scores.length; n++) {
            const si = new Scalar(yb[n]);
            const sp = scores[n][0];
            losses[n] = si.sub(sp).pow(2);
            if (losses[n].data <= 0.01) {
                accuracy += 1;
            }
        }

        accuracy /= losses.length;
        const dataLoss = losses.reduce((p, c) => p.add(c)).mul(1 / losses.length);

        // L2 Reg
        const alpha = 1e-4;
        const regLoss = model.parameters().map(p => p.pow(2)).reduce((p, c) => p.add(c)).mul(alpha);

        const totalLoss = dataLoss.add(regLoss);

        return [totalLoss, accuracy];
    }
}
