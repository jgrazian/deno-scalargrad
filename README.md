# Deno Scalar AutoGrad

A scalar (single number) autograd engine based off of the great [micrograd](https://github.com/karpathy/micrograd) written for Deno in Typescript. Not super fast or anything but relatively easy to understand, especially with the Typescript annotations.

The biggest issue that Deno is going to have in breaking into the ML market is the lack of operator overloading. I apologize for the grossness of method chaining.


## Instructions
1. Clone this repo
	```
	git clone
	cd deno-scalargrad
	```
2. Run the tests
	`deno test -A`
3. Run the examples
	```
	deno run ./examples/calc_grad.ts
	deno run ./examples/nn_fit_function.ts
	```