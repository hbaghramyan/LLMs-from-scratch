28/03/2024 - հաջորդ հանդիպման համար ու էն հերթականությամբ, ոնց էստեղ նշված ա
1.⁠ ⁠էս վիդեոն https://www.youtube.com/watch?v=zduSFxRajkE մինչև 01:11:382 վերջացնել
2. ⁠կուրսից՝ 2.1 - 2.7
3. օնլայն գրքից 2.6 միայն
4. օնլայն գրքից App A. Introduction to PyTorch - A1, A2, A3 - սա լինելու ա կուրսի 2.1 - 2.7 կետերի որոշ չափով կրկնություն, բայց կարդացածը թարմացնելու համար պետք ա։ Արդյունքը լավ ստանալու համար տարբեր օրերի կկարդաս Pytorch-ի մասին օնլայն գրքից ու կուրսից։
5. https://rezaborhani.github.io/mlr/blog_posts/Linear_Supervised_Learning/Part_3_Perceptron.html
6. ⁠⁠Հավանականության գրքից մինչև Proof That the Real Numbers Are Uncountable


08/04/2024 - հաջորդ հանդիպման համար ու էն հերթականությամբ, ոնց էստեղ նշված ա
1. Քո արածները ավելացրու մեր https://github.com/hbaghramyan/LLMs-from-scratch github-ին։ Հենց ավելացնես, ինձ ասա - HB TODO
2. ⁠կուրսից՝ 2-րդ գլխի առաջադրանքները - HB TODO
3. օնլայն գրքից 2.6-2.8
4. օնլայն գրքից App A. Introduction to PyTorch - A4-A6
5. https://rezaborhani.github.io/mlr/blog_posts/Linear_Supervised_Learning/Part_3_Perceptron.html
6. ⁠⁠Հավանականության գրքից - Proof That the Real Numbers Are Uncountable, Exercises - 1-4

to discuss - 
1. how to count tensor dimensions
2. on contiguousity
    import torch

    # Create a contiguous tensor
    tensor = torch.tensor([[1, 2], [3, 4]])

    # Check if the tensor is contiguous
    print(tensor.is_contiguous())  # This will print True

    # Create a non-contiguous tensor by selecting a column, which results in a stride that does not cover the entire storage.
    non_contiguous_tensor = tensor[:, 1]

    # Check if the non-contiguous tensor is contiguous
    print(non_contiguous_tensor.is_contiguous())  # This will print False

    PyTorch stores tensors in a row-major order, meaning that it stores all the elements of a row contiguously in memory before moving to the next row.

    .view works only on contiguous tensors and reshape also on not contiguous
3. Appendix A.5
    1. model.layers[0] to see the number of inputs and outputs

    2. To manually count the number of trainable parameters in a neural network model like the one you've defined, you would calculate the number of weights and biases in each layer that has parameters. 

    Here's how you do it for your `NeuralNetwork`:

    1. **1st Hidden Layer:**
        - Weights: The first hidden layer has a `Linear` module with `num_inputs` coming in and `30` neurons, so it has `num_inputs * 30` weights.
        - Biases: There's one bias term per neuron in the layer, so there are `30` biases.

    2. **2nd Hidden Layer:**
        - Weights: This layer connects `30` neurons from the first hidden layer to `20` neurons of the second hidden layer, totaling `30 * 20` weights.
        - Biases: Similarly, this layer has `20` biases, one for each neuron.

    3. **Output Layer:**
        - Weights: The output layer has `20` incoming connections per output neuron, and there are `num_outputs` neurons, so there are `20 * num_outputs` weights.
        - Biases: There's one bias term per output neuron, adding up to `num_outputs` biases.

    Now, let's plug in the numbers for `num_inputs = 50` and `num_outputs = 3`:

    1. **1st Hidden Layer:**
        - Weights: `50 * 30 = 1500`
        - Biases: `30`

    2. **2nd Hidden Layer:**
        - Weights: `30 * 20 = 600`
        - Biases: `20`

    3. **Output Layer:**
        - Weights: `20 * 3 = 60`
        - Biases: `3`

    So the total number of trainable parameters would be the sum of all weights and biases across the layers:


    $(1500 \text{ weights} + 30 \text{ biases}) + (600 \text{ weights} + 20 \text{ biases}) + (60 \text{ weights} + 3 \text{ biases}) = 1500 + 30 + 600 + 20 + 60 + 3 = 2213$

    Therefore, your model has 2,213 trainable parameters.

    3. X = torch.rand((1, 50)) is important because 1 stands for the batch number

    4. ON logits

    Yes, you're right in connecting the term "logits" with the logit function in statistics, which is indeed related to the reason why we use this term in the context of neural networks.

    The logit function is defined in logistic regression and is used to map probability values $p$ (ranging between 0 and 1) to the entire real number line (ranging from $-\infty$ to $+\infty$). The function is given by:
    $$
    \text{logit}(p) = \log\left(\frac{p}{1-p}\right)
    $$

    This function is the inverse of the logistic (sigmoid) function:

    $$
    \sigma(x) = \frac{1}{1 + e^{-x}}
    $$
    where $\sigma(x)$ maps a real number $x$ (the logit) to a probability $p$.

    In the context of neural networks, especially those designed for classification tasks, the final layer’s outputs (before any activation like softmax or sigmoid) are called logits because they can be seen as being the input to the sigmoid or softmax function, which will map them onto a probability scale. These logits represent the unbounded real numbers, which are the direct counterparts to the bounded probabilities produced by the sigmoid or softmax functions.

    Thus, the term "logits" reflects this link, as these are the values that you would apply the logit function to if you were going backward from probabilities to raw scores. The naming helps clarify that these values are yet to be passed through the final activation function to produce probabilities.

    Also to read

    https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81

4. Appendix 6

    To see the order change in the epochs

    for i in range(3):
    for idx, (x, y) in enumerate(train_loader):
        print(f"Batch {idx+1}:", x, y)
______________________________________
        self.input_ids = []
        self.target_ids = []

16/04/2024 -

Հավանականության գրքից վերցրու 6-11 խնդիրները:

To discuss

1. vocab = {idx: bytes([idx]) for idx in range(256)} why square brackets in bytes([idx])

26/04/2024 -

1. օնլայն գրքից - A7, A8, Chapter 3 - 3.1, 3.2, 3.3.1
2. from the Course - Unit 3.1, 3.2
3. Հավանականության գրքից - 12 - 14

To discuss

29/04/2024 - 


1. https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
2. https://datascience.stackexchange.com/questions/23159/in-softmax-classifier-why-use-exp-function-to-do-normalization - second response only
3. [Wikipedia article on the Entropy in information theory](https://en.wikipedia.org/wiki/Entropy_%28information_theory%29) - till the example inclusive
4. օնլայն գրքից - A7, A8, Chapter 3 - 3.3.2
5. from the Course - 3.3

To discuss
1. all_context_vecs = attn_weights @ inputs
    print(all_context_vecs)
2. def compute_accuracy(model, dataloader):

    model = model.eval() - this doesn't required

3. torch.isclose(context_vec_2, all_context_vecs[1, :], atol=1e-4, rtol=0)

To discuss

06/05/2024 - 

1. https://www.youtube.com/watch?v=zduSFxRajkE&t=2266s էս վիդեոն շարունակել մինչև 01:25:28
2. օնլայն գրքից Chapter 3 - 3.4, Appendix 9.1, 9.2 - սրա համար պետք ա օգտագործես, enter runtime and choose the accelerator to be T4 GPU
3. Հավանականության գրքից - 1.5 առանց խնդիրները լուծելու
4. from the Course - 3.4

To discuss

1. torch.sum(attn_weights, dim=0)

13/05/2024 - 

1. օնլայն գրքից Chapter 3 - 3.4, 3.5.1
2. Հավանականության գրքից - 1.5 պարագրաֆի 1-6 խնդիրները
3. from the Course - 3.5

To discuss

1. why the dimensions should be (3, 5) for the weight. Can you show it with matrix multiplications?

Sure! I'll explain why the weight matrix in a `nn.Linear` layer has the shape `(3, 5)` for transforming an input of `5` dimensions into an output of `3` dimensions, using matrix multiplication as the basis.

### The Setup

- **Input Dimensions (`d_in`)**: 5
- **Output Dimensions (`d_out`)**: 3

### Linear Transformation in Neural Networks

In neural networks, a linear transformation can be represented as:

\[ Y = XW^T \]

where:
- \( X \) is the input matrix.
- \( W \) is the weight matrix.
- \( Y \) is the output matrix after applying the linear transformation.

### Shapes of the Matrices

1. **Input Matrix \( X \)**:
   - Shape: `(batch_size, d_in)`
   - Example: For a batch size of 2, \( X \) has shape `(2, 5)`.

2. **Weight Matrix \( W \)**:
   - The rows of \( W \) represent the output features, and the columns represent the input features.
   - Shape: `(d_out, d_in)`
   - Example: \( W \) has shape `(3, 5)`.

3. **Output Matrix \( Y \)**:
   - Shape: `(batch_size, d_out)`
   - Example: For a batch size of 2, \( Y \) has shape `(2, 3)`.

### Matrix Multiplication

Given the shapes:
- \( X \) is `(2, 5)`
- \( W \) is `(3, 5)`

To multiply \( X \) and \( W \), and align the dimensions for valid matrix multiplication (where the inner dimensions must match), \( W \) must be transposed. Therefore, \( W^T \) (the transpose of \( W \)) is `(5, 3)`.

Now, the matrix multiplication \( XW^T \) works as follows:
- \( X \) has shape `(2, 5)`
- \( W^T \) has shape `(5, 3)`

The result \( Y \) after multiplication will then have the shape `(2, 3)`, fitting our expectation of the output dimensions.

### Conclusion

The reason the weight matrix \( W \) in `nn.Linear` is initialized with the shape `(d_out, d_in)` (in this example, `(3, 5)`) is so that its transpose \( W^T \) aligns correctly with the input \( X \) for matrix multiplication. This transpose operation is part of the underlying mechanics in many neural network frameworks, including PyTorch, to facilitate straightforward linear transformations using matrix multiplication.

I hope this clarifies the rationale behind the dimensions of the weight matrix in PyTorch's `nn.Linear` layer!

2. dim=-1

3. see scaling_rationale.py

### 27/05/2024 - 

1. օնլայն գրքից Chapter 3 - վերջացնել, չմոռանաս առաջադրանքները
2. from the Course - 3.6, 3.7 առանց առաջադրանքների
3. Հավանականության գրքից - 1.5 պարագրաֆի 7-11 խնդիրները

### to discuss 

1. https://pytorch.org/docs/stable/notes/randomness.html

### 03/06/2024 - 

1. օնլայն գրքից Chapter 3 - վերջացնել, չմոռանաս առաջադրանքները
2. from the Course - 3.6, 3.7 առանց առաջադրանքների
3. https://huggingface.co/learn/nlp-course/en/chapter6/2?fw=pt սա հայերենի համար ա
3. Հավանականության գրքից - 1.5 պարագրաֆի 7-11 խնդիրները

### 10/06/2024 -

մնում են նույնը ինչ անցած հանդիպմանը

to discuss
 1. each self.heads is an instance of CausalAttention

 ### 24/06/2024 -

մնում են նույնը ինչ անցած հանդիպմանը

to discuss

* attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

**Causal attention** is crucial for autoregressive tasks, where the model generates text one token at a time, predicting the next token based on the previous ones. This is essential for maintaining the chronological order of text generation, ensuring that the model doesn’t use future information that hasn’t been generated yet.

**Conventional attention** is useful for tasks like text classification, where understanding the entire context (both past and future tokens) is important. Models like BERT (Bidirectional Encoder Representations from Transformers) use this type of attention.

.view() can be applied only to contiguous Tensors 

.transpose() usually results in non-contiguous tensors

 ### 29/07/2024 -

 1. mha-implementations.ipynb from the 02_bonus_efficient-multihead-attention
 սա ուղղակի նայի, առանց շատ խորանալու, պետք ա աշխատեցնես google colab-ում

 2. օնլայն գրքից - 4.1, 4.2 

 3. Հավանականության գրքից - 1.6 խնդիրներով հանդերձ

 4. from the Course - Unit 3 Exercises

 to discuss

    1. 

    self.trf_blocks = nn.Sequential(
    *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
    
    self.trf_blocks = nn.Sequential(
    DummyTransformerBlock(cfg),
    DummyTransformerBlock(cfg),
    DummyTransformerBlock(cfg),
    ...
)

    2. the need for to_device()

    To ensure that the positional embeddings are on the same device as the input indices and token embeddings, you specify device=in_idx.device when creating the positional indices tensor. This guarantees that the positional indices tensor and, consequently, the output of pos_emb will be on the correct device.

    3. 

    https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

    4. 

    https://en.wikipedia.org/wiki/Bessel%27s_correction


 ### 05/08/2024 -

 1. mha-implementations.ipynb from the 02_bonus_efficient-multihead-attention
 սա ուղղակի նայի, առանց շատ խորանալու, պետք ա աշխատեցնես google colab-ում

 2. օնլայն գրքից - 4.1, 4.2, 4.3, 4.4

 3. Հավանականության գրքից - 1.7 առանց խնդիրների

 4. from the Course - Unit 3 Exercises + Units 4.1, 4.2, 4.3

  ### 12/08/2024 -

  1. օնլայն գրքից - 4.3, 4.4, 4.5

  2. հավանականության գրքից - 1.7 խնդիրներ

  3. from the Course - Unit 3 Exercises + Units 4.1, 4.2

  4. tokenization study - only for me Henrikh

  To discuss

  Layer Normalization in LLMs:

	•	Normalization Process:
	•	Layer Norm normalizes across the feature dimension for each token independently.
	•	For each token, it computes the mean and variance across all 768 features (not across the batch).
	•	This means for each token, it normalizes the 768 features using the mean and variance computed across these features.
	•	Intuitive Example:
	•	Imagine you are processing the word “cat” in the sentence. The word “cat” is represented by a vector of 768 features. Layer Norm will normalize this vector by computing the mean and variance across these 768 features and then normalizing and scaling them.
	•	Each token has its own normalization, independent of other tokens or other examples in the batch.

Batch Normalization in CNNs:

	•	Normalization Process:
	•	For each filter (out of the 16 filters), Batch Norm normalizes the activations across the batch dimension.
	•	This means for each filter, it looks at all the activations across the batch and computes the mean and variance for each spatial location (i.e., each pixel position in the 32 \times 32  grid).
	•	The normalization is performed across all images in the batch for each filter separately.
	•	Intuitive Example:
	•	Imagine you have 32 images (batch size  N = 32 ). For each filter in the CNN, you would compute the mean and variance across these 32 images for each pixel location in the  32 \times 32  feature map.
	•	After normalization, the pixel values are scaled and shifted using the learned gamma and beta parameters, but each filter has its own pair of these parameters.

https://de.wikipedia.org/wiki/Convolutional_Neural_Network

https://youtu.be/DtEq44FTPM4?si=wIl8-RlKH13qqaGJ&t=382