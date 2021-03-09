import { assertEquals } from "https://deno.land/std@0.89.0/testing/asserts.ts";

import { Scalar } from '../src/mod.ts';

Deno.test("Scalar - Add", () => {
    const a = new Scalar(1);
    const b = new Scalar(2);
    const c = a.add(b);
    assertEquals(c.data, 3);
    assertEquals(c.children, [a, b]);

    c.backward();
    assertEquals(a.grad, 1);

    assertEquals(a.add(2).data, 3);
});

Deno.test("Scalar - Mul", () => {
    const a = new Scalar(1);
    const b = new Scalar(2);
    const c = a.mul(b);
    assertEquals(c.data, 2);
    assertEquals(c.children, [a, b]);

    c.backward();
    assertEquals(a.grad, 2);

    assertEquals(a.mul(2).data, 2);
});

Deno.test("Scalar - Pow", () => {
    const a = new Scalar(2);
    const b = a.pow(3);
    assertEquals(b.data, 8);
    assertEquals(b.children, [a]);

    b.backward();
    assertEquals(a.grad, 12);
});

Deno.test("Scalar - Relu", () => {
    const a = new Scalar(2);
    const b = a.relu();
    assertEquals(b.data, 2);
    assertEquals(b.children, [a]);

    b.backward();
    assertEquals(a.grad, 1);

    assertEquals(new Scalar(-1).relu().data, 0);
});

Deno.test("Scalar - Backwards", () => {
    const a = new Scalar(1.5); // 1.5
    const b = new Scalar(-4.0); // -4.0
    const c = a.pow(3).div(5); // a^3 / 5
    const d = b.pow(2).relu().add(c); // Relu(b^2) + c

    d.backward();
    assertEquals(d.data, 16.675);
    assertEquals(a.grad, 1.35);
    assertEquals(b.grad, -8.0);
});

Deno.test("Scalar - Big Backwards", () => {
    //24.70408163265306
    //138.83381924198252
    //645.5772594752186
    const a = new Scalar(-4.0);
    const b = new Scalar(2.0);

    let c = a.add(b);
    let d = a.mul(b).add(b.pow(3));
    c = c.add(c).add(1);
    c = c.add(1).add(c).add(a.neg());
    d = d.add(d.mul(2)).add(b.add(a).relu());
    d = d.add(d.mul(3)).add(b.sub(a).relu());
    const e = c.sub(d);
    const f = e.pow(2);
    let g = f.div(2);
    g = g.add(f.pow(-1).mul(10));

    const tol = 0.001;
    g.backward();
    assertEquals(24.70408 - g.data <= tol, true, 'Forward');
    assertEquals(138.8338 - a.grad <= tol, true, 'Backprop - A gradient');
    assertEquals(645.5772 - b.grad <= tol, true, 'Backprop - B gradient');

});
