# Neural Network Exam Prep

## Intro
- History of "classical" NNs
	- Pitts & McCulloch => Neuron Definition
	- Rosenblatt => Convergence Theorem
	- Widrow => Kind of Backpropagation
	- LeCuns first conv nn 1989 for AT&T digit recognition
	- 2006 => restricted boltzmann machine building block for DNN
	- 2013 => deepminds alpha go zero etc.
- Early applications (NetTalk / Alvin)
	- 1987 nettalk => reads text aloud
	- 1989 => ALVINN autonomous driving with 30x32px camera
- Enabling factors for the "Deep Learning" revolution
	- le cuns OCR 1989
	- mnist / imagenet perviously these big datasets where not available
	- more poreful hardware through gpu (and soon tpus)
	- new algorithms and architectures (not limited to MLP)
	- many other applications suddenly beating existing approaches (deep belief network, stacked auto-encoder, deep recurrent network)
- Why more layers
	- one layer sufficient for any function in theory
	- number of nodes and weights grow exponentially
	- consecutive layers "learn" more complicated features building upon the previous ones
- Traditional Machine Learning Problems
	- Classification => predict class label
	- regression => predict a value
	- clustering => find clusters in data
	- associations => find frequent patterns in data
- to adjust results one can
	- increase the bias
	- increase the weights
	- change the result of the previous activation by changing their weights
- hebbian theory = neurons that fire together wire together

## Statistical Pattern Recognition
- Bayes Theorem
	- probability of an event based on prior knowledge of conditions
	- P(A | B) => likelyhood of A if B is true
	- $P(A|B) = \frac{P(B|A) * P(A)}{P(B)}$
	- enables estimation of posteriror probabilities
- Optimal Decision Boundary
	- joint probability densities
	- e.g. 2 classes. each of them produces a classification value
	- draw bar graphs in different color per class
	- find lowest overlap for minimal amount of misclassifications
	- best fit is where they cross
- Loss Matrix and Risk Minimization
	- Risk => cost of misclassification
	- use loss matrix to quantify cost
	- use the one with smallest loss
	- cost => if higher cost for one parameter it's change impact is higher!
- recheck "apple banana"
- Overfitting
	- usually means training set works well but test set doesn't
	- more parameters => higher risk of overfitting
- Regularization
	- prevent overfitting by imposing constraints on values or number of model parameters
	- done by e.g. penalizing large cofficient values (shrinkage ridge regression weight decay)
- Cross-validation
	- monitor error botn in training and test set

## Linear Models
- Linear separability
	- two classes are linearly seperable if there exists a hyperplane that separates them
	- formulas always need to be bigger than each other
	- doesn't imply convexness
- Cover's Theorem (1965)
	- what is the chance that randomly labeled set of N Points in d-dimensional space is linearly seperable?
	- if number of points in d dimensional space smaller than 2*d then probably seperable
	- $= \frac{N}{d-1}$ => 500 images with 16*16px => d = 16*16; N = 500 =>$\frac{500}{257} \approx 2$
- Perceptron Learning Algo
	- find suitable values for wheights such that training set gets classified correctly
	- geometrically => find hyper-plane that separates two classes
	- works by => initialize weight randomly; for each misclassified sample adjust weights
	- convergence => if linearly seperable the learning will terminate successfully after a finite number of iterations
- Gallant's Pocket Algo
	- initial idea => see if weight change improved => expensive to recheck
	- improvement => count consecutive correct classifications
	- if best run then keep weights saved for later
	- converges with probability 1 BUT might be multiple separating hyperplanes and it picked a bad one
	- if around 2 probably seperable
- Adaline
	- main idea => minimize Mean Squared Error
	- supports arbitrary values (not limited to -1 & 1)
	- can solve linear regression
- Logistic Regression
	- replace sign function by smooth approximation and use steepest descent to find weights that minize error
- Stochastic Gradient Descent
	- finds minimum of a function (e.g. error function)
- Multi-class linear separability
	- there exists c linear discriminant functions such that each x is assigned to a class
- Multi-class Perceptron (Slide 44)
- Histogram
	- data binning leads to loss of information
	- number of buckets increases exponentially with data dimensionality (feature amount) hence amount of records to classify also exponentially grows

## MLP and Backpropagation
- MLP (Multi-layer Perceptron)
	- single perceptrons => linear decision boundary => XOR unsolvable
	- via multi layers this is overcome
	- require training through backpropagation
- Expressive Power of MLP (Slide 5)
	- each layer allows for more convex regions to be drawn for classification
	- single node => single line
	- one hidden layer => one convex regions
	- two hidden layers => convex open / closed regions
	- three layer => arbitrary shapes
	- every boolean function can be described by single hidden layer
	- continous functions any bounded continuous function can be approximated by two hidden layers
- Backpropagation Algo
	- forward pass => in this step the network is activated on one example and the error of each neuron of the output layer is computed
	- backward pass => in this step the network error is used for updating the weights. Starting at the output layer, the error is propagated backwards through the network, layer by layer with help of the generalized delta rule. Finally, all weights are updated
	- doesn't guarranteee convergence, weights initialized randomly
	- backpropagation => find wanted weight adjustment for a single example
	- for each example look at activation result for each neuron and desired output
	- accumulate changes to previous weights
	- recursively do so until reach start
	- average together desired changes of each example => negative gradient of the cost function
	- because very expenssive following is done do stochastig gradient descent
		- random order of training samples put together into mini-batches
		- for each mini-patch do backprop => approximation
    - disatvantages
    	- no convergence guaranee
    	- only local minimum found
- Backpropagation update modes
	- batch mode => all inputs at once, takes longest
	- (stochastic gradient descent) mini-batch mode => use small random subsamples to process => quicker but approximated
	- on-line mode => update weights using one example at a time
- Backpropagation when to stop
	- total mean squared error change => when MSE change is sufficiently small
	- when reaching certain percentage for training or test set (helps against overfitting)
- Alternative Error Messures
	- (SSE) for regression problems use linear outputs and sum quared error
	- (Logistic) for binary classification use logistic output unit and minimze cross entropy
	- (Cross Entropy + Softmax) for multiclass use softmax and minimize cross entropy


## Convolutional Networks
- On more Layers
	- adding more layers is harmful because
		- lack of generalization (increased number of weights quickly overfits data)
		- huge number of (bad) local minima to trap the gradient descent algo in
		- vanishing or exploding gradients (update rule involves products of many numbers) cause additional problems
- Convolution
	- convolve one matrix into a smaller one by applying a so called kernel
	- results in "degree of overlap"
	- helps "feature detectors" => returns high values when corresponding patch is similar to filter matrix
- bias => every operation (e.g. 5x5 conv has ONE bias so 5x5+1)
- Weight Sharing (???)
	- weights in convolutional layers are shared to reduce memory and improve performance
- Subsampling (???)
- local receptive fields (???)
- Pooling
	- progressively reduce spatial size of representation
	- reduce amount of features and hence complexity
	- reduces e.g. 3x3 to 2x2 etc.
	- Reduction of resolution (and size) of feature maps, enforcing generalization by losing some information about location of features, filtering out noise.
- dropout
	- at each training stage node can be dropped with a probability
	- for altering weights ignore dropped nodes but reinsert htem after
	- by not changing all nodes chances of overfitting are decreased#
	- During each cycle of the training process a fraction of neurons (e.g., 50%) is disabled and corresponding weights are not used or affected by the training algorithm. It is a powerful technique of preventing overfitting by reduction of complex co-adaptations of neurons and forcing network to learn more robust features that are useful in conjunction with many different random subsets of the other neurons.
- ReLU's (???)
	- alternative to sigmoid, hyperbolic tangent etc.
	- increases non-linear properties of decision function
	- preferred over sigmoid etc. because it trains faster without big accuracy changes
- Be able to count weights between layers
	- filter verkleinerung => Filtergröße - 1 bsp. 32 orig 5x5 => 32-4 = 28
	- Shared Weights => (FilterGrößer*FilterGröße + 1) * amount of feature maps
	- Connections => 28x28 + FilterGröße*FilterGröße+1
	- fully connected => (input*input+1)*(reduced*reduced)*amount of feature maps

## GPU learning
- know theano tensorflow keras
- limit gpu

## Recurrent Neural Networks
- made for sequential data
	- one to one ( no recurrence)
	- one to many (e.g. image captioning)
	- many to one (e.g. sentiment analysis)
	- many to many
- RNN (plain)
	- problems
		- exploding vanishing gradients
		- speed at which past actions are forgotten
	- make use of sequential information
	- traditionally all inputs and outputs independent of each other
	- called recurrent cause same task for every element of sequences is done
	- plain rnns would use backpropagation through time
	- because longer sequence => longer network and more problems with error poropagation limited to 5-10 sequences but lstm helps
- LSTM (Long Short Term Memory) Networks
	- let the network decide which info requires preserving
	- kept in a cell state vector
	- hidden layer replaced by specially designed network
	- consists of 3 gates (Explain architecture of LSTM Layer)
		- forget gate => decides whcih info of state should be neglected (sigmoid takes 0,1 activation)
		- input gate => decide which info should be added to internal state (tanh generates new info; sigma filters it)
		- output gate => computes the final output by combining new input, previous output and cell state
- GRU basically LSTM but simplified => no cell state
- Leading example of text generation
- Calculate amount of trainable params
- Understand difference between stateful and stateless mode
	- *stateful* => reuse cell states from last run / after batch is processed DON'T reset network state
	- *stateless* => don't reuse cell states from last run / reset network state after batch process
- Know various usages of RNNs (slide 5)
	- image captioning
	- sentiment analysis
	- machine translation
	- video frame classification

## Autoencoders
- Key Idea => unsupervised learning of efficient codings
- Stacked => stacked sparse autoencoder layers are used for image recognition
- Denoising
	- train network with broken to healed pictures
	- apply result => works quite well
- Other Applications
	- varational autoencoder
	- sparse => reducing dimensionality in a lossy way

## Optimization
- Learning as optimization => mimize expected loss over traning dataset
- Gradient descent
	- classic gradient descent => always points in the direction of steepest increase
	- stochastic gradient descent => use small random batches for faster calculation
		- faster
		- avoids local minima
		- better generalization, prevents overfitting => kind of regularization
	- momentum => dampened oscillations and faster convergence
	- nesterov accelerated gradient
		- accounts future momentum
		- in practice slightly better than momentum
- Batch Normalization
	- normalize the output of a previous activation layer by substracting the batch mean and dividing by batch standard deviation
	- adds two trainable parameters => standard deviation (gamma) and mean (beta)
	- reduces the amount by which hidden unit values shift around (covariance shift)
	- helps independence between layers
	- enables higher learning rates because activations don't go really high or low anymore
	- reduces overfitting through its regularization effects (similar to dropout but less aggressive
- Be able to calculate diretion (a = learning rate; gradient )
	- SGD => $p_{new} = p_{old} - a * gradient$
	- Momentum => $p_{new} = p_{old} - a * gradient + b * last direction$
	- Nesterov => $p_{new} = momentumRes + b * (momentumRes - momentumResPrev)$

## RBMs and Deep Belief Networks
- Restricted Boltzmann Machine
	- 2 layered unsupervised network
	- nodes take values 0 or 1
	- model of probability distribution of P(x, h)
	- Energy, generative model, likelihood
	- train via gradient descent
	- used for (recommendations, feature extraction, dimensionality reduction, collaborative filtering)
	- optimized using log likelihood during training
	- as heuristic use constrastive divergence
- Contrastive Divergence Algo
	- training undirected graphical models
	- relies on approximation of gradient
- Deep Belief Networks
	- stack RBMs in a greedy manner
	- extract deep hierarchical representations of training data
	- uses unlabeled data
- Auto Encoders
	- use to cluster topcis
- Netflix RBM challenge
	- 1mil price money, 100000000 records, 5000000 customers 18000 movies, each has a rating
	- blend together different models for optimal results
	- calculation includes Ratings*Movies*Hiddenlayers NOT USERS! they are data

## Convolutional Networks 2
- Details of Alex & ResNet
- Treating image boundaries
- Weight initialization strategies (Glorot, GLorot uniform)
- Data Augmentation
- Applications

## Q-Learning / Go / Alpha Go Zero


## PARAMETRIC DENSITY ESTIMATION NN 3 Slide 2
- maximize product and maximize logarithm sum
- to find optimum
	- uniform distribution => a = min(X) b = MAX(X)
	- normal distribution => a = mean(X); b = std(X)
- understand discriminant functions for sets a und b

# know which activation / error functions are good for which thing
- binary classification => sigmoid + cross entropy
- multi class => cross entropy + softmax
- regression problems => linear + square error

# overfitting prevention
- early stopping
- regularization L2
- dropout
- data augmentation
	- e.g. use known image and apply various filters to it like color, saturation etc.


# to learn
- bayes nicht nur umstellen sondern angepasste wahrscheinlichkeiten!
	- related to density
- log likelihood => use values from X insert into function, multiply each result. apply log. Log will get bigger with bigger numbers
	- used in RBM for optimization
	- in parametric denstiy estimation maximize => product of likelihoods, sum of logarithms of likelihood
- schätzen wie viel iterationen von gradient descent
	- $xNeu = xAlt - learningRate* f'(xAlt)$
- XOR-Problem
- if all weights are 1 or 0, changes always same => never converges because backpropagation takes derivative from previous layer
- linear seperability
- anzahl weights etc. FÜR ALLE NETZWERKE RNN, CNN etc.
	- cnn
        - filter verkleinerung => Filtergröße - 1 bsp. 32 orig 5x5 => 32-4 = 28
        - Shared Weights => (FilterGrößer*FilterGröße + 1) * amount of feature maps
        - Connections => 28x28 + FilterGröße*FilterGröße+1
        - fully connected => (input*input+1)*(reduced*reduced)*amount of feature maps
    - rnn
    	- bsp. 8000 input 100 hidden 8000 output
    	- 2 * 100 * 8000 + 100 * 100
    	- (input + output) * hidden + hidden * hidden
- contrastive divergence
	- single update of weights for single input vector => 5 * M * N
	- anything added increase the 5
	- The three passes up, down, and up require 3mn multiplications, updating weights (calculation of x0h0-x1h1) requires 2mn more multiplications; thus in total 5mn multiplications are required.
- gaussian distirbution parameters
	- means and covariances
	- amount of means = amount of features
	- amount of covariances = fakultät von features => für 10 => 10+9+8+7+6+5+4+3+2+1
	- biggest limiation => data is never normally distributed
- RBMS + contrastive divergence in detail!!!
- deep belief network
- ALLES FÜR RNNS