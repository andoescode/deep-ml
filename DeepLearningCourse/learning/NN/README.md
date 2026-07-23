Deep learning - Theories
---
What is Neural Network?
---

**Definition**

Given: x_i = features, y = output (prediction target)

Combine x -> different group of 'neuron' -> output y
neuron => create linear relationship between x, y

**Supervised learning**

Simple prediction -> Standard NN
Photo/ image tagging -> CNN
NLP -> RNN
Hybrid form of inputs -> Custom hybrid approach

Scale size of labeled data vs size of NN: more data + bigger NN = Better performance, more data + traditional learning alg = will stop learning at some points

---
Neural Network Basic + Shallow
---

Logistic regression

Cost function = average of loss function = Sum (Loss function for y_i) /m (m= number of y)

Gradient descent -> go closer to the global optimal point

```J(w,b) -> Derivatives w:= w - lr(dJ(w,b)/dw); b:= b - lr(dJ(w,b)/db)```

Flow = Input layer -> Hidden layer(s) -> Output layer

Propagation

<forward>
  
  z[layer 1] = np.dot(w[1].T, X) + b[1] -> A[layer 1] = sigmoid(z) = 1/(1+np.exp(-z)) (Activation function can be changed based on requirements - eg.ReLu, Leaky ReLu or tanh)

  z[layer 2] = np.dot(w[2].T, A[layer 1]) + b[2] -> A[layer 2] = sigmoid(z[layer 2]) -> loss(A[layer 2], Y)
... till layer n

cost = (1/m) * L(A,Y) = (1/m) * -(np.dot(Y, np.log(A).T) + np.dot((1 - Y), np.log(1 - A).T)) = -(1/m) * (np.dot(Y, np.log(A).T) + np.dot((1 - Y), np.log(1 - A).T))

<backward>
dz[2] = A[2] - Y
  
dw[2] = 1/m * np.dot(dz[2], A[2].T)

db[2] = 1/m * np.sum(dz[2], axis=1, keepdims=True) # sum by rows: axis = 1, sum by cols: axis = 0

dz[1] = np.dot(W[2].T, dz[2]) * g'(z[1]) (element-wise product)

dw[1] = 1/m * np.dot(dz[1], X.T)

db[1] = 1/m * np.sum(dz[1], axis=1, keepdims=True)

dz = da * g'(z)

dw = dz * x

db = dz


Notation

a_i{l] with i = index of node in layer, l = index of layer (e.g a_1[1] = activate node 1 of layer 1

b_i = np.array(# of nodes, 1)

W.shape = (w_x, w_y) where w_x = number of total nodes in hidden layer l, w_y = number of input features

How to choose activation function

----------------------------------------------------------------------------------
| Problem                   | Activation function | Output                   	 |
----------------------------------------------------------------------------------
| Regression                | 	    Linear        | Unbounded real number    	 |
| Binary Classification     | 	    Sigmoid       | P in range 0,1           	 |
| Multiclass Classification | 	    SoftMax       | sum (P_distribution) = 1 	 |
| Multilabel Classification |       Sigmoid	  | independent P for each label |
----------------------------------------------------------------------------------


---
Deep Neural Network
---

Notation

L = m (number of layers)

n[l] = number of units/ nodes in layer l

a[l] = g[l](z[l]); w[l], b[l] = weight, bias for z[l]

x = a[0]

Propagation

Forward propagation

x=a[0] -> z[1] = w[1].x + b[1], a[1] = g[1](z[1])

...

output layer yhat -> z[L] = w[L].a[L-1] + b[L], a[L] = g[L](z[L])

n_x -- size of the input layer

n_h -- size of the hidden layer

n_y -- size of the output layer

W1 = np.random.randn(n_h, n_x) * 0.01
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(n_y, n_h) * 0.01
b2 = np.zeros((n_y, 1))

Z[l].shape = A[l].shape = (n[l], m)
dW[l].shape = W[l].shape = (n[l], n[l-1])
db[l].shape = b[l].shape = (n[l], 1)

Backward propagation

da[l] -> da[l-1], dw[l], db[l]
(cache z[l])

dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

Backpropagation calculates gradients so the model can learn = learn where it is wrong and how much that affected the cost/learning process
-> knowing dw, db -> model can update itself to adjust the approach based on the feedback from backward propagation.

Others

Hyperparameters = learning rate, # iteration, # hidden layers, # hidden units, choice of activation functions

Activation function = introduce non-linearity while still allowing gradients to be calculated

Hidden layers (n[1] .. n[L-1]) -> ReLU (mostly, unless it returned too many negative outputs -> change to different function)
Output layer (n[L]) -> based on the purpose of the learning (e.g. sigmoid for binary or multilabel classification aka independent/non exclusive cases, SoftMax for multiclass classification aka competing probabilities that sum to 1 between exclusive classes)

---
Train/Dev/Test and Bias/Variance
---

Train/Dev/Test

Small dataset -> 60/20/20 or 70/15/15
Big dataset -> 98/1/1 or 99/0.5/0.5
dev and test sets should come from the same distribution where possible
Test -> size = big enough to give high confidence

human level <-- 	avoidable bias	 --> training error <-- 	variance	 --> dev error
(Train bigger model, longer optimization alg, NN architecture) (More data, regularization, NN architecture)

Bias and variance

= explain why model is making errors.

bias = how strong the model is (training error to Bayes/human error)
variance = how generalised the model is (training error to dev/evaluate error)

High bias = low learning = underfitting (high error in train and dev)

Fix =
bigger neural network
more layers
more hidden units
train longer
better architecture
lower regularisation
better features/input representation

High variance = good at train but fails on unseen data = overfitting (low errors in train but significantly higher in dev)

Fix =
more training data
regularisation
dropout
data augmentation
early stopping
smaller model
better train/dev distribution matching

High bias + high variance (bad in train and much worse in dev) => BAD
Low bias + low variance (low errors in both train and dev, not much different in the e_train/e_dev ratio) => GOOD

Regularization = make the model less dependent on the training data

J reg = J + lambda/2m * sum(||W[l]||**2) with W large -> sum(||W[l]||**2) is large
dw_reg = dw + (lambda/m) * w


Gradient Checking - check where the weight update go unreasonable


He init parameters

parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])
parameters['b' + str(l)] = np.zeros((layers_dims[l],1))

---
Optimization
---

Minibatch gradient descent

=> fast learning + make progress without training through the whole set
=> only use for bigger training set ( m> 2000), mini batch size should be a power of 2
=> consider GPU/CPU memory capability

Minibatch size s -> # of minibatches = T = total data size/ s

loop through all t in T:
epoch = forward -> backward

Exponentially weighted average

= smooth noisy values overtime by giving more weights to the recent data

vt​=(1−β)θt​+(1−β)βθt−1​+(1−β)β2θt−2​+...

|                         (\beta) | Effect                      |
| ------------------------------: | --------------------------- |
|         Small (\beta), e.g. 0.5 | reacts quickly, less smooth |
| Large (\beta), e.g. 0.9 or 0.98 | reacts slowly, smoother     |

Bias correction = fixes the early-time bias caused by initializing v0 = 0

-> v_t = (v_t)/(1−β^t)

Momentum
Momentum behaves like a ball rolling downhill.
If gradients keep pointing in the same direction, momentum speeds up learning.
If gradients oscillate back and forth, momentum smooths the movement.
This is useful when the cost surface is shaped like a long narrow valley.
=> Smooth gradient

RMSProp
If a parameter has large gradients, RMSProp reduces its update size.
If a parameter has small gradients, RMSProp allows a relatively larger update.
So RMSProp helps control unstable directions.
It reduces oscillation and makes learning more stable.
=> More oriented direction

ADAM = Adaptive Moment Estimation = Momentum + RMSProp

| Method           | Uses gradient average? | Uses squared gradient average? | Main benefit                      |
| ---------------- | ---------------------: | -----------------------------: | --------------------------------- |
| Gradient descent |                     No |                             No | Simple baseline                   |
| Momentum         |                    Yes |                             No | Faster, smoother updates          |
| RMSProp          |                     No |                            Yes | Adaptive step sizes               |
| Adam             |                    Yes |                            Yes | Fast and stable default optimizer |

learning rate decay alpha = (1/(1 + decayRate * epochNumber)) * alpha_0
learning rate decay helps model to learn faster and closer to the optimal point.

---
Structure Machine Learning projects
---

ML Strategy

Precision = how many true positive of all positive predictions (prevent false alarm)
Recall = how many true positive of all actual positives (prevent missed real positives)
F1 Score = balance both Precision and Recall

Cost = accuracy (optimising) - 0.5 * running_time (satisfying)

Analyse error + Cleaning up incorrected labeled data

Examine and analyse both the right and wrong labels

If 2 different quality of data -> dont shuffle 2 different quality of data, train = all good + small subset of not as good, dev/test = not as good

Human error - Training error - Training Dev error - Dev error - Test error
	(avoidable bias)  (variance)     (data mismatch)   (degree of overfitting to dev set)

If data mismatch -> make training set similar to dev/test set

Transfer learning (reuse pretrained model + finetuning) (have to apply from task with more data -> train task with smaller data)

Transfer from task A to task B:
Both task got same input x
Task A data > Task B data
Low level features from A can help learning B

Multi-task learning:
Train on the set of tasks that could be benefit from having shared lower level features
Similar data distribution between tasks
Big enough nn to do well on all tasks

End to end learning:
If doesn't have enough data -> breakdown into multiple steps
Identify clearly what the input and output end

---
Convolutional Neural Networks
---

Foundation

Definition

CNN = Early layers learn local visual patterns like edges and corners, deeper layers combine them into higher-level features, and final layers use those features to make a prediction.

Edge detection
Apply filter size 3x3 to image size nxn -> abs(image * filter)
vertical = [[1 0 -1][1 0 -1][1 0 -1]]
Horizontal = [[1 1 1][0 0 0][-1 -1 -1]]
Sobel = [[1 0 -1][2 0 -2][1 0 -1]]
Scharr = [[3 0 -3][10 0 -10][3 0 -3]]

Filter 3x3 = [[w1 w2 w3][w4 w5 w6][w7 w8 w9]]

Padding
n x n (img) * f x f (filter) -> (n - f + 1) x (n - f + 1) (output)
Add padding to make sure all the details will be covered
e.g. add padding p = 1 around the border of the image
=> (n + 2p) x (n + 2p) * f x f -> (n + 2p - f + 1) x (n + 2p - f + 1)
Make sure output size == input size after applying filter: f is odd normally (center), p = (f-1)/2

Stride
Given padding p, stride s: n x n (img) * f x f (filter) -> ((n + 2p - f)/s + 1) x ((n + 2p - f)/s + 1) (output)
(n + 2p - f)/s -> floor((n + 2p - f)/s) (round down if not int)

Convolution vs Cross-Correlation in CNN
| Aspect            | Convolution                                                                 | Cross-Correlation                                                   |
| ----------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Kernel flipping   | Yes, the kernel is flipped horizontally and vertically before applying.      | No, the kernel is used as-is without flipping.                          |
| Operation         | Mathematical convolution includes a kernel flip.                            | Similar operation, but without the kernel flip.                         |
| Use in theory     | Important in signal processing due to properties such as time-invariance.    | Also used in signal/image processing, but without convolution flipping.  |
| Use in practice   | Often mentioned in CNNs, but standard CNNs usually do not use true flipping. | The operation actually used in most CNN implementations.                 |
| Efficiency        | Slightly less efficient because of the flipping step.                       | More computationally efficient because it skips the flipping step.       |
| Pattern detection | Detects patterns while following the formal convolution definition.          | Detects patterns directly by matching the kernel with input regions.     |

Convolution on RGB image
height x width x #channels (image) * f x f x #channels (filter) -> (n - f + 1) x (n - f + 1) x number of filters (output)
output = W[i]a[i-1] => on one layer l: z[l] = W[l]a[l-1] + b[l] = output l + b[l]
number of parameters in a layers of nf (number of filters) = f X f X #channels X nf

Convolution layer l
f[l] = filter w/h, p[l] = padding, s[l] = stride, n_c[l] = number of filters
input size = n_H[l-1] x n_W[l-1] x n_c[l-1]
output size = n_H[l] x n_W[l] x n_c[l] (number of filters in layer l)
n_(W/H)[l] = (n_(W/H)[l-1] + 2p[l] - f[l])/s[l] + 1)
filter size = f[l] x f[l] x n_c[l-1] (number of channels of filter has to be the same as one from input)
activation a[l] size = n_H[l] x n_W[l] x n_c[l]
weights (all filters) w[l] size = f[l] x f[l] x n_c[l-1] x n_f[l] (number of filters in layer l)
bias b[l] size = (1,1,1,n_c[l])

Types of layer in CNN

| Layer type            | Main job                                        | Learns parameters? | Typical position         |
| --------------------- | ----------------------------------------------- | ------------------ | ------------------------ |
| Convolution layer     | Extract local features from images              | O                  | Early and middle layers  |
| Pooling layer         | Downsample feature maps                         | X                  | After convolution blocks |
| Fully connected layer | Combine extracted features for final prediction | O                  | Near the output          |

Convolution layer

| Benefit                 | Explanation                                                        |
| ----------------------- | ------------------------------------------------------------------ |
| Local feature detection | Small filters look at local image regions.                         |
| Parameter sharing       | The same filter (features detector) is reused across the whole image.|
| Translation awareness   | A filter can detect the same feature in different image locations. |
| Fewer parameters        | Much fewer weights than a fully connected layer on raw pixels.     |


Pooling layer
(given f, s, not padding)
Max pooling: max value of all values under the region area that fxf filter cover
Avg pooling: avg value of all values under the region area that fxf filter cover
No parameters to learn
Use when features map is large

| Reason                          | Explanation                                                 |
| ------------------------------- | ----------------------------------------------------------- |
| Reduce computation              | Smaller feature maps mean fewer calculations.               |
| Reduce memory usage             | Downsampled feature maps use less memory.                   |
| Add small translation tolerance | If a feature moves slightly, pooling can still preserve it. |
| Control overfitting             | Fewer activations can reduce model complexity.              |


FC (Fully Connected)

A fully connected layer, also called a dense layer, connects every input neuron to every output neuron.

Conv layer       ->  learns visual features
Pooling layer    ->  reduces size and keeps important activations
Fully connected  ->  combines features for classification (final output)

Architecture

Input image -> Convolution layers -> Pooling layers -> Flatten -> Fully connected layers -> Prediction

Exercise:

Architecture: input -> conv -> gelu -> maxpool(2,2) -> conv -> gelu -> dropout(0.2) -> maxpool(2,2) -> conv -> gelu -> dropout(0.3) -> maxpool(2,2) -> conv -> gelu -> Flatten -> Linear -> (CrossEntropyLoss) <output>

Scheduler types and their placements

| Scheduler           | Placement                                 |
| ------------------- | ----------------------------------------- |
| `StepLR`            | End of each epoch                         |
| `ExponentialLR`     | End of each epoch                         |
| `MultiStepLR`       | End of each epoch                         |
| `CosineAnnealingLR` | Usually end of each epoch                 |
| `OneCycleLR`        | After every `optimizer.step()`            |
| `CyclicLR`          | After every `optimizer.step()`            |
| `ReduceLROnPlateau` | After validation, passing validation loss |
