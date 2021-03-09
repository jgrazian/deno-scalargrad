import { Scalar } from '../src/mod.ts';

const a = new Scalar(2.0);
const b = a.pow(3.0);
const c = b.mul(new Scalar(4.0));
c.backward();

console.log(`a's gradient with respect to c is ${a.grad}`);
