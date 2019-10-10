## stacks
Neural network implementation from scratch in C++.

### Current operations:
* __Dense:__ Most basic operation. Every output takes in every input.
* __Local2:__ Dense and conventional mix. Instead of connecting the output to every input, it connects to the input and its local neighbours in a given radius. Input and output can be in different width and height. If input is smaller than the output, same input will be reused for close by outputs. If input is bigger than the output, closeby inputs will be skipped.

### Notes:
* Compile time optimization with constexpr at cost of no runtime configuration. Almost twice as fast as original version. Ex: sx::OpLocal2<sx::Func::RELU, RADIUS, IN_WIDTH, IN_HEIGHT, OUT_WIDTH, OUT_HEIGHT>()
* No multithreading.
* Conv2 does not work.
