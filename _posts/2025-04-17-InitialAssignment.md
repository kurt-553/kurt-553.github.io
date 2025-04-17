# Machine Learning - Loss Functions
Loss functions are used in machine learning as a measure for how far away a given sample passed into the model is from the predicted result. 
During training, the loss function is used to reinforce the model's predictions, as the goal of the model is to minimise the magnitude of loss on the set of training data.  
After the model has been trained, the loss function is a good measure of how difficult a given input to the machine learning model is. A high loss function is an indicator of an input far away from what the model is trained on, and can be a predictor of uncertainty on the prediction result.

## Cross Entropy in Classification
### Equations
### $$loss = -\sum_{n=1}^{N} y_n log(p_n)$$   
Where:   
| n is classifier group and N is number of groups  
| $p_n$ = prediction probability for the sample  
| $y_n =$ binary 1/0, 1 for the correct classification, 0 for all others. 

### Cross Entropy Concept
The cross entropy loss function is quite simple in its concept, but also very powerful and extensible (see [focal loss](#focal-loss) section). The equation for cross entropy essentially defines a logarithmic loss on the prediction certainty. That is for any given classifier which is supposed to be of type $n$ the loss is given by $-log(p_n)$, where $p_n$ is the probability that the input is assigned to this classifier. Furthering this concept, if a sample is supposed to be of class X but is prediced as X with low certainty, this will produce a higher loss on the model, which the model aims to minimise. The higher the certainty the lower the loss or pentalty of the model. This simple concept in some sense assumes that all classifier groups are equal in the difficulty, size of dataset and quality of dataset. Which may not always be the case, however produces good results in most cases.

<h2 id="focal-loss"> Focal Loss as an Alternative to Cross Entropy </h2>   

A good article (which taught me about this topic) is [here](https://towardsdatascience.com/focal-loss-a-better-alternative-for-cross-entropy-1d073d92d075/).   

### Equations
### $$loss = -\sum_{n=1}^{N} \alpha_n (1 - p_n)^{\gamma} log(p_n)$$
Where:   
| n is classifier group and N is number of groups  
| $p_n$ = prediction probability for the sample  
| $y_n =$ binary 1/0, 1 for the correct classification, 0 for all others.   
| $\alpha_n =$ tunable linear scaling (0,1] variable for each group    
| $\gamma =$ tunable weighting for more difficult classifications     
### Focal Loss Concept
Focal loss aims to use a 'down weighting' approach to tune each individual classifier/group based on the quality, size and difficulty of training on their dataset. Down weighting allows for tuning of the loss per group, to reduce the effect of groups that are easier to classify on the model results. By prioritising (or give a more dominant effect) to the more difficult classification problems, this allows for a more accurate model in the places where it has most affect on the outcomes.
