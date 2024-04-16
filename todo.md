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