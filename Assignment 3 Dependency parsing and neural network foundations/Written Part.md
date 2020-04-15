### CS224N-2019 Assignment 3 Written Part ###

#### Machine Learning & Neural Networks ####

##### (a) Adam Optimizer #####

$$\mathrm{i.}$$  **$m$** consists of two parts. We could see the first part $\beta_1m$ as a push to maintain its origin trend while the other $(1-\beta_1)\delta$ as a push to a variance. $\beta_1$ is often set to 0.9, which implies that **$m$** tends to keep its direction preventing from overshooting. Hence, The low variance could help $J$ reduce the vibration on the vertical direction while downward to the convergence point.

$$\mathrm{ii.}$$ 

##### (b) Dropout #####

$$\mathrm{i.}$$ It's obvious that we have $\mathbb{E}[X]_i = \mathbb{E}[X_i]$. Hence,
$$
\begin{equation} \nonumber
\begin{split}
\mathbb{E}_{p_{drop}}[h_{drop}]_i & = \mathbb{E}_{p_{drop}}[\gamma d \cdot h]_i\\
& = \mathbb{E}_{p_{drop}}[\gamma d_i \cdot h_i]\\
& = \gamma \mathbb{E}_{p_{drop}}[d_i \cdot h_i]\\
& = \gamma [p_{drop} \times 0 +(1-p_{drop}) \times1]h_i\\
& = \gamma (1-p_{drop})h_i
\end{split} 
\end{equation}
$$
According to the topic, we have 
$$
\nonumber 
\begin{split}
\mathbb{E}_{p_{drop}}[h_{drop}]_i & = hi \\
\gamma (1-p_{drop})h_i & = hi \\
\gamma & = \frac{1}{1-p_{drop}}
\end{split}
$$
$$\mathrm{ii.}$$ Dropout is design for preventing overfitting and improving robustness of model. However, applying dropout in evaluation  actually adds occasionality in it and decrease model's reliability in turn. 

#### Neural Transition-Based Dependency Parsing ####

(a) Transition process is as following:

| Stack                          | Buffer                                 | New Dependency               | Transition          |
| ------------------------------ | -------------------------------------- | ---------------------------- | ------------------- |
| [ROOT]                         | [I, parsed, this, sentence, correctly] |                              | Initial Conguration |
| [ROOT, I]                      | [parsed, this, sentence, correctly]    |                              | SHIFT               |
| [ROOT, I, parsed]              | [this, sentence, correctly]            |                              | SHIFT               |
| [ROOT, parsed]                 | [this, sentence, correctly]            | parsed$\rightarrow$I         | LEFT-ARC            |
| [ROOT, parsed, this]           | [sentence, correctly]                  |                              | SHIFT               |
| [ROOT, parsed, this, sentence] | [correctly]                            |                              | SHIFT               |
| [ROOT parsed,  sentence]       | [correctly]                            | sentence$\rightarrow$this    | LEFT-ARC            |
| [ROOT, parsed]                 | [correctly]                            | parsed$\rightarrow$sentence  | RIGHT-ARC           |
| [ROOT, parsed, correctly]      | []                                     |                              | SHIFT               |
| [ROOT, parsed]                 | []                                     | parsed$\rightarrow$correctly | RIGHT-ARC           |
| [ROOT]                         | []                                     | ROOT$\rightarrow$parsed      | RIGHT-ARC           |

(b) 2$n$. For each word in sentence, it takes one step to shift word from buffer to stack and another one to pop from stack. Hence the total step is double the number of words.



