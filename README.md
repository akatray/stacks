## stacks
Neural network implementation in C++.

### Current operations:
* __Dense:__ Most basic operation. Every output takes in every input.
* __Local2:__ Dense and conv2 mix. Input and output are 3d objects [width] [height] [depth]. Every output takes in input in the same position and surrounding neighbours. Input and output can be in different width and height dimensions, but must be same in depth dimension. If input is smaller than the output, same input will be reused for close by outputs. If input is bigger than the output, closeby inputs will be skipped.

### On multithreading:
* Dense is currently memory bound on my computer. (I only have one memory stick. Effectively halving bandwidth.)
* Local2 need testing.

### Future plans:
* Conv2 - Maybe. Local2 in theory should be better for what I need. And scales good compared to Dense, which after 128x128 can't even fit in ram nor will be trained in this millennium.
* Maxpool2 - Necessity for Conv2.
