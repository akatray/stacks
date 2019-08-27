## stacks
Neural network implementation in C++.

### On multithreading.
After some testing I am on the assumption that dense operation is already memory bound.
Therefore, it's not planned until someone proves that this assumption is wrong or other
type of operation would benefit from it.

*Dividing fit() function in fragments and executing them in N threads will result in performance divided by N.
*Running the same model on two different threads divides performance in half or more.
*Running two programs that run same model will half performance.

### Future plans.
*Convoliution operation: Somewhat hard to implement when I don't have math education to understand wtf I am doing.
