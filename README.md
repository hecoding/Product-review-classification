# Binary Classification of Product Reviews based on Recurrent Neural Networks
Authors: Héctor Laria Mantecón, Aditya Kaushik Surikuchi and Maximilian Proll

<img src="https://github.com/hecoding/DL-project/blob/master/report/pics/bad.png" width="700"/>

Filtering out product reviews and focussing on relevant reviews is an important trait from which both e-commerce retailers and consumers can mutually benefit.
A quick web-search resulted in numerous existing solutions to this problem. Some of them are based on deep-learning architectures whereas others are conventional classification methods that leverage NLP techniques.

We decided to tackle this task by implementing and tweaking the best existing deep-learning model, which apparently performs binary classification of product reviews as good or bad, based on the recurrent neural network architecture. Our results are compared to other similar approaches and publications in terms of time and performance.

Eventually, we demonstrate that using GRUs instead of LSTMs improves the prediction accuracy to __75.1__%, __77.5__% for bad and good reviews respectively. Considering the baseline model we relied upon reaches the accuracy of __74.7__%, __76.3__% for bad and good reviews respectively, under the same datasets and runtime environments.

Furthermore, we evaluate and discuss the implications (advantages and disadvantages) of variations in the architecture, importance of certain features and details on how they impact the network flow (like weights and the descent). We conclude with detailed discussions and comparisons.

Network architecture:
<img src="https://github.com/hecoding/DL-project/blob/master/report/pics/arch.png" />
