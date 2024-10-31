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

### 28/08/2024 -

1. օնլայն գրքից - 4.3, 4.4, 4.5, 4.6, 4.7

2. from the Course - Unit 3 Exercises + Units 4.1, 4.2 - 4.6

3. Հավանականության գրքից, 1.8

### 02/09/2024 -

1. օնլայն գրքից - 4.6, 4.7 - նույնն եմ թողնում էս անգամ

2. from the Course - Unit 3 Exercises + Units 4.4, 4.5, 4.6 - նույնն եմ թողնում էս 

3. Հավանականության գրքից, 1.8 - միայն կարդալ

4. https://www.youtube.com/watch?v=CS4cs9xVecg&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0

առաջին շաբաթվա նյութրը նայի - բոլորը, որոնք C1W1 ունեն իրենց անվանման մեջ,
ավելի ուշ կավելացնեմ կոդերի նյութերը, նաև C1_W1.pdf կարդա

5. 

https://youtu.be/4u81xU7BIOc?si=GtSSKSsCtFr1O74d&t=3980

սա եթե կհասցնես նայի էն վարկյանից, որը նշել եմ

https://youtu.be/4u81xU7BIOc?si=Ufs_O74n7Imh5kYT&t=948

ու սա մինչև 30։27

6. https://huggingface.co/learn/nlp-course/en/chapter6/2?fw=pt 

To discuss



2. sum(p.numel() for p in block.att.parameters())
2360064
[p.numel() for p in block.att.parameters()]
[589824, 589824, 589824, 589824, 768]
sum([p.numel() for p in block.att.parameters()])
2360064

### 16/09/2024 -

0. see the todo.md in dl fundamentals

1. օնլայն գրքից - 4.6 - exercises, ch04/02_performance-analysis/flops-analysis.ipynb, 4.7

2. [stratificatied vs cluster vs random vs convinient vs systematic sampling](https://www.youtube.com/watch?v=9PaR1TsvnJs) 

3. To understand how you can approximate a function using neural nets 
    https://stackoverflow.com/questions/55170460/neural-network-for-square-x2-approximation#:~:text=15-,The%20problem,-is%20that%20x

4. [GPU vs CPU](https://stackoverflow.com/questions/53325418/pytorch-speed-comparison-gpu-slower-than-cpu#:~:text=39-,GPU,-acceleration%20works%20by) կարող ես փորձարկել Colab-ում - Henrikh check!!!!

5. http://neuralnetworksanddeeplearning.com/chap1.html մինչև Sigmoid neurons

6. Հավանականության գրքից, 1.8 - առաջադրանքները

7. https://www.youtube.com/watch?v=CS4cs9xVecg&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0 - միայն W02 վիդեոները նայել

8. sys.path.insert(0, os.getcwd())

9. https://huggingface.co/learn/nlp-course/en/chapter6/2?fw=pt 

10. էն հայերեն տեքստերից մեկը փորձի բաց անել ու տեքտը որպես txt պահել


### 23/09/2024 - 

1. [__name__ == "__main__"](https://builtin.com/articles/name-python#:~:text=%E2%80%9CIf%20__name__%3D%3D%20',main%20method%20should%20be%20executed.&text=If%20you%20are%20new%20to,or%20without%20a%20main%20method.) in python

2. Implications with multiprocessing

    Let’s break down the behavior of Python's multiprocessing on macOS and Windows more clearly, focusing on why not using the `if __name__ == "__main__":` block can lead to problems.

    ### Understanding Multiprocessing and Script Execution

    When using multiprocessing in Python (like when you set `num_workers > 0` in a `DataLoader`), the operating system needs to create new processes to handle these additional tasks.

    1. **Linux (Fork Method):**
    - On Linux, the new processes are created using the fork method.
    - This method duplicates the parent process, including its current state (variables, code execution point, etc.).
    - No re-importing of the script occurs, so the new process doesn't execute the script from the start.

    2. **macOS and Windows (Spawn Method):**
    - On macOS and Windows, the default method for creating new processes is spawn.
    - This method starts a brand new, fresh Python interpreter process.
    - The new process needs to know what code to run, so it **re-imports the script** that created it.

    ### Why the `if __name__ == "__main__":` Block is Crucial

    When the new interpreter spawns (on macOS/Windows), it starts by re-importing your script to understand what it should execute. Here’s what happens step-by-step if your script doesn't use `if __name__ == "__main__":`:

    1. **Script Import:** When the new process starts, it imports your script, running from the top.
    
    2. **Uncontrolled Execution:** If the DataLoader and dataset creation code is outside of `if __name__ == "__main__":`, it runs as soon as the script is imported, not just when intended.

    3. **Recursive Spawning:** Because the DataLoader with `num_workers > 0` creates new processes, those processes will also re-import and re-execute the script from the top, trying to create their own DataLoader instances.

    4. **Infinite Loop:** This leads to a chain reaction where every new process spawns more processes, which again import and execute the script, trying to spawn even more processes. This results in infinite recursion of process creation.

    ### How the `if __name__ == "__main__":` Block Prevents This

    - The `if __name__ == "__main__":` block ensures that certain code only runs when the script is executed directly, not when it is imported.
    - When a new process spawns and imports the script, it does not execute the code inside `if __name__ == "__main__":`, preventing the recursive spawning issue.
    
    For example:

    ```python
    if __name__ == "__main__":
        # This code will only run when the script is executed directly.
        train_loader = DataLoader(...)
    ```

    In a new process, this block is ignored because `__name__` is not `"__main__"` (it’s the module name). This prevents unintended and repeated execution of the DataLoader creation code, keeping the multiprocessing controlled and functional.

    ### Key Point

    The `if __name__ == "__main__":` block ensures that multiprocessing doesn’t lead to runaway process creation on platforms that use the spawn method, making your script work correctly and efficiently on all operating systems.

3. The `transforms.Normalize((0.5,), (0.5,))` step is included to normalize your image data so that its pixel values fall within the \([-1, 1]\) range instead of the default \([0, 1]\) range produced by `transforms.ToTensor()`. Here's why this is beneficial:

* **Normalization Process**: The `transforms.Normalize(mean, std)` function adjusts the pixel values using the formula:

   \[
   \text{output} = \frac{\text{input} - \text{mean}}{\text{std}}
   \]

   By setting `mean` to \((0.5,)\) and `std` to \((0.5,)\), the transformation becomes:

   \[
   \text{output} = \frac{\text{input} - 0.5}{0.5} = 2 \times \text{input} - 1
   \]

   This maps the input pixel values from the \([0, 1]\) range to \([-1, 1]\).

* **Benefits of \([-1, 1]\) Range**:
   - **Neural Network Performance**: Many neural network architectures, especially those using activation functions like `tanh`, perform better when inputs are centered around zero. This can lead to faster convergence during training.
   - **Stability**: Normalizing data can improve numerical stability and make the training process less sensitive to the scale of input features.
   - **Consistency**: If pre-trained models (e.g., those trained on ImageNet) expect inputs in a specific range, normalizing your data accordingly ensures compatibility.

* **Channel-wise Normalization**: The `(0.5,)` tuple indicates that this normalization is applied to each channel individually. For grayscale images (single-channel), this is straightforward. For RGB images, you would provide a mean and standard deviation for each channel.

**In summary**, the `transforms.Normalize((0.5,), (0.5,))` step scales your image data to a \([-1, 1]\) range, which is often preferred for training neural networks due to improved performance and stability.

**Answer:**

Because it scales image pixels from [0, 1] to [–1, 1]; using Normalize((0.5,), (0.5,)) centers and scales the data so neural networks train better with inputs in the [–1, 1] range

4. օնլայն գրքից

ch04/02_performance-analysis/flops-analysis.ipynb

ch05 5.1-ը լրիվ

5. Հավանականության գրքից, 1.9

6. nn_and_dl/course_01_nn_and_dl/w02/A01_done/Python_Basics_with_Numpy_final.ipynb
սա առաջին կուրսի երկրոդ շաբաթվա առաջադրանքներն են

https://drive.google.com/drive/folders/1BYE-XGL0csgQfaRbImZhDMcY9uXWV2vS

էստեղ են, քեզ հետ կիսվել էի

7. https://huggingface.co/learn/nlp-course/en/chapter6/2?fw=pt 

### To discuss

1. BASE_CONFIG.update(model_configs[size]) 
In this config 'emb_dim': 1024, 'n_layers': 24, 'n_heads': 16 get updated every time the loop
runs because there can't be 2 keys with the same name in the dictionaries.

2. add notebooks from the previous chapters to discuss

3. lightning course unit 4 exercises

4. id(idx_cond)
5808296944
id(idx)
5797275792
idx_cond is idx
False

5. 

Remember that figure 5.6 displays the softmax probabilities for a compact seven-token vocabulary to fit everything into a single figure. This implies that the starting random values will hover around 1/7, which equals approximately 0.14. However, the vocabulary we are using for our GPT-2 model has 50,257 tokens, so most of the initial probabilities will hover around 0.00002 (1/50,257).

This applies only to the outputs of the model and not the embedding vectors. Each value of the embedding matrix
is independent on the others. 

torch.sum(tok_emb, dim=-1, keepdim=True).squeeze(dim=-1)
tensor([[  0.4252,  13.5739, -26.4600,  -5.1018]])
torch.sum(tok_emb, dim=-1, keepdim=True).squeeze(dim=-1).shape

import torch
torch.manual_seed(0)  # Set random seed for reproducibility

# Create an embedding layer and get embeddings for specific indices
embedding_layer = torch.nn.Embedding(10, 5)  # 10 is the max index, 5 is the embedding dimension
idx = torch.tensor([2, 3, 1])
embeddings_from_layer = embedding_layer(idx)

# Now create a manual embedding matrix with the same random seed
torch.manual_seed(0)  # Reset random seed to ensure the same random numbers are generated
manual_embedding_matrix = torch.randn(10, 5)  # Same dimensions as the embedding layer
embeddings_from_manual_matrix = manual_embedding_matrix[idx]

6. : and [0, 1, 2] give different results:  

probas[text_idx, :, targets[text_idx]]
tensor([[7.4541e-05, 2.6072e-05, 1.8191e-05],
        [2.5497e-05, 3.1061e-05, 2.7802e-05],
        [3.2404e-05, 1.0943e-05, 1.1563e-05]])


probas[text_idx, [0, 1, 2], targets[text_idx]]
tensor([7.4541e-05, 3.1061e-05, 1.1563e-05])

targets[0]
tensor([3626, 6100,  345])

the reason is to adjust the probabilities for each target in tensor([3626, 6100,  345]). 
0th row in probas[0] contains probabilities for token 3626, 1st - for token  6100, 2nd for 345. Each row has 50257
elements but only the 3626th element of 0th row is considered, 6100th for 1st row, and 345th for the 3rd row.  

probas[0][0, 3626]
tensor(7.4541e-05)
probas[0][1, 6100]
tensor(3.1061e-05)

These targets respresent tokens in " effort moves you" 

see the link to understand why

https://tiktokenizer.vercel.app/?model=gpt2

which means we need to 

7. appendix B on logarithm

8. how do solve the probability problems


id(idx0)
13093152112
id(idx)
5228297360
idx000 = idx[:]
idx[0] = 1
idx000
tensor([[1, 1, 1, 1]])
idx
tensor([[1, 1, 1, 1]])

30/09/2024

0. nn_and_dl/course_01_nn_and_dl/w02/A02_done/Logistic_Regression_with_a_Neural_Network_mindset.ipynb

1. ch05 5.1-ը լրիվ online գրքից, 

ch02/03_bonus_embedding-vs-matmul/embeddings-and-linear-layers.ipynb

2. Հավանականության գրքից, 1.9

3. WHatsapp-ով ուղարկածս խնդիրը քննարկել

4. Unit 4 Exercises 
https://lightning.ai/courses/deep-learning-fundamentals/training-multilayer-neural-networks-overview/unit-4-exercises/


The trivial result for perplexity in this context can be understood as follows:

1. **Perplexity Definition**: The perplexity is defined as $( 2^H )$, where $( H )$ is the entropy of the probability distribution. Perplexity gives an idea of how many possible outcomes there are for the random variable.

2. **Calculating Entropy**: For a fair 6-sided die, the probability distribution is uniform, so each face has a probability of $( \frac{1}{6} )$. The entropy $( H )$ is given by:

   $$
   H = -\sum_{i=1}^{6} p(X = i) \log_2 p(X = i) = -6 \times \left(\frac{1}{6} \times \log_2 \frac{1}{6}\right).
   $$
   Simplifying this, we get:

   $$
   H = -\log_2 \frac{1}{6} = \log_2 6 \approx 2.58.
   $$

3. **Finding Perplexity**: Using the formula for perplexity:

   $$
   \text{perplexity} = 2^H = 2^{\log_2 6} = 6.
   $$

The result is trivial because, for a uniform distribution over $ n $ equally likely outcomes (in this case, $ n = 6 $ for the die), the perplexity directly equals the number of possible outcomes. Thus, for a fair die with 6 faces, the perplexity is simply 6, which reflects the six possible choices that the die could land on. This relationship holds generally for any uniform distribution over $ n $ outcomes, making it a straightforward result.

10/10/2024

1. https://www.youtube.com/watch?v=fXOsFF95ifk&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=25 - C1W3 բոլոր վիդեոները նայել

2. ch05 5.1, 5.2 online գրքից

3. ch02/03_bonus_embedding-vs-matmul/embeddings-and-linear-layers.ipynb

4. WHatsapp-ով ուղարկածս խնդիրը քննարկել

5. Unit 4 Exercises 
https://lightning.ai/courses/deep-learning-fundamentals/training-multilayer-neural-networks-overview/unit-4-exercises/

to discuss

1. torch.tensor vs torch.Tensor
https://stackoverflow.com/questions/51911749/what-is-the-difference-between-torch-tensor-and-torch-tensor

2. 

shuffle=False,
drop_last=False,

for validation

3. encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
if you expect the text to contain "<|endoftext|>".

4.

Reasons for Setting drop_last=True in the Training Data Loader

	1.	Consistent Batch Sizes:
	•	Why: Having all batches of the same size simplifies many aspects of training.
	•	Benefits:
	•	Computational Efficiency: Consistent batch sizes ensure that each forward and backward pass has the same computational load, which can be more efficient for GPUs.
	•	Memory Management: It helps in managing GPU memory usage, preventing unexpected spikes that could lead to out-of-memory errors.
	2.	Stable Training Dynamics with Batch-Dependent Layers:
	•	Batch Normalization and Layer Normalization:
	•	Issue: Layers like BatchNorm compute statistics (mean and variance) across the batch.
	•	Problem with Smaller Batches:
	•	A smaller last batch can produce unreliable statistics.
	•	This can lead to instability in training or degraded model performance.
	•	Solution: Dropping the last incomplete batch ensures all batches have the same number of samples, providing consistent statistics.
	3.	Avoiding Bias from Incomplete Batches:
	•	Why: The last, smaller batch may not be representative of the data distribution.
	•	Issue:
	•	Including this batch can introduce bias, especially if the data is not perfectly shuffled.
	•	The model might overfit to these few samples during the last update of each epoch.
	•	Benefit of Dropping:
	•	Ensures that each parameter update is based on a full batch, maintaining consistency in the training process.
	4.	Simplified Loss Calculation and Metrics:
	•	Consistent Averaging:
	•	Loss functions often assume batches of the same size for averaging.
	•	With varying batch sizes, you need to adjust loss calculations to account for the smaller batch, adding complexity.
	•	Simplification:
	•	Dropping the last incomplete batch avoids the need for special handling in loss computation.
	5.	Optimizing Training Performance:
	•	Hardware Utilization:
	•	Modern GPUs are optimized for operations on fixed-size tensors.
	•	Variable batch sizes can lead to suboptimal utilization and reduced throughput.
	•	Parallelism:
	•	Consistent batch sizes allow for better parallelization across multiple GPUs or CPU cores.
	6.	Ease of Implementing Certain Training Strategies:
	•	Batch-wise Operations:
	•	Some training strategies involve operations that assume a fixed batch size (e.g., certain types of data augmentation or batch-wise regularization techniques).
	•	Having consistent batch sizes simplifies these implementations.


13/11/2024

1. Unit 4 Exercises 
https://lightning.ai/courses/deep-learning-fundamentals/training-multilayer-neural-networks-overview/unit-4-exercises/

1. C1W3 առաջադրանքները

2. 5.3 online գրքից

3. ch02/03_bonus_embedding-vs-matmul/embeddings-and-linear-layers.ipynb

4. WHatsapp-ով ուղարկածս խնդիրը քննարկել

to discuss