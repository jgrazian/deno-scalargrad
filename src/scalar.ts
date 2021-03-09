/** Possible operations performed on Scalars. */
enum Op {
    Nop = "",
    Add = "+",
    Mul = "*",
    Pow = "^",
    Relu = "ReLU",
}

/**
 * Holds a single scalar value and references to children to form a graph.
 * @param data Value that this Scalar represnets
 * @param children Previous Scalars that formed this
 * @param op The operator that created this
 */
export class Scalar {
    data: number;
    grad = 0.0;
    children: Scalar[] = [];
    _backward: () => void;
    op: Op;

    constructor(data: number, children: Scalar[] = [], op: Op = Op.Nop) {
        this.data = data;
        this.op = op;
        this.children = children;
        this._backward = () => { };
    }

    // ----Tier 1 Ops----
    /** Scalar Addition.
     * 
     *      let a = new Scalar(1.0);
     *      let b = new Scalar(2.0);
     *      let c = a.add(b);
     *      assert(c.data == 3.0);
     */
    add(this: Scalar, other: Scalar | number): Scalar {
        const _other = (other instanceof Scalar) ? other : new Scalar(other);
        const out = new Scalar(this.data + _other.data, [this, _other], Op.Add);
        out._backward = () => {
            this.grad += out.grad;
            _other.grad += out.grad;
        };
        return out;
    }

    /** Scalar Multiplication
     * 
     *      let a = new Scalar(2.0);
     *      let b = new Scalar(3.0);
     *      let c = a.mul(b);
     *      assert(c.data == 6.0);
     */
    mul(this: Scalar, other: Scalar | number): Scalar {
        const _other = (other instanceof Scalar) ? other : new Scalar(other);
        const out = new Scalar(this.data * _other.data, [this, _other], Op.Mul);
        out._backward = () => {
            this.grad += _other.data * out.grad;
            _other.grad += this.data * out.grad;
        };
        return out;
    }

    /** Scalar Exponentiation
     * 
     *      let a = new Scalar(2.0);
     *      let b = a.pow(3.0);
     *      assert(b.data == 8.0);
     */
    pow(this: Scalar, other: number): Scalar {
        const out = new Scalar(Math.pow(this.data, other), [this], Op.Mul);
        out._backward = () => {
            this.grad += (other * Math.pow(this.data, other - 1)) * out.grad;
        };
        return out;
    }

    /** ReLU function
     * 
     * x = {x if x > 0; 0 otherwise}
     * 
     *      let a = new Scalar(2.0);
     *      let b = a.relu();
     *      assert(b.data == 2.0);
     *      let c = new Scalar(-2.0);
     *      assert(c.relu().data == 0.0);
     */
    relu(this: Scalar): Scalar {
        const out = new Scalar((this.data < 0) ? 0 : this.data, [this], Op.Relu);
        out._backward = () => {
            this.grad += Number(out.data > 0) * out.grad;
        };
        return out;
    }

    /** Builds a DAG of this Scalar's ancestors.
     * 
     * Uses chain rule to calcuate derivative of each wrt this.
     * 
     *      let a = new Scalar(2.0);
     *      let b = a.pow(3.0);
     *      let c = b.mul(new Scalar(4.0));
     *      c.backward();
     *      assert(a.grad == 48.0);
     */
    backward(this: Scalar) {
        const topo: Scalar[] = [];
        const visited = new Set();

        function buildTopo(v: Scalar) {
            if (!visited.has(v)) {
                visited.add(v);
                for (const child of v.children) {
                    buildTopo(child);
                }
                topo.push(v);
            }
        }

        buildTopo(this);

        this.grad = 1;
        for (const v of topo.reverse()) {
            v._backward();
        }
    }

    // ----Tier 2 Ops----
    /** Self Negation */
    neg(this: Scalar): Scalar {
        return this.mul(-1);
    }

    /** Scalar Subtraction */
    sub(this: Scalar, other: Scalar | number): Scalar {
        const _other = (other instanceof Scalar) ? other : new Scalar(other);
        return this.add(_other.neg());
    }

    /** Scalar Division */
    div(this: Scalar, other: Scalar | number): Scalar {
        const _other = (other instanceof Scalar) ? other : new Scalar(other);
        return this.mul(_other.pow(-1));
    }
}
