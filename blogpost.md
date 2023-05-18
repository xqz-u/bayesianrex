## Introduction
Our project is based on the paper "_Safe Imitation Learning via Fast Bayesian Reward Inference from Preferences_" by Brown, Coleman, Srinivasan, Niekum (2020).

This study proposes Bayesian Reward Extrapolation (Bayesian REX), an imitation learning algorithm that uses ranked demonstrations and Markov Chain Monte-Carlo (MCMC) to efficiently sample a learned reward function from a pretrained network. This reward function can then be used to train high performing reinforcement learning agents.
Because of the efficiency of Bayesian REX, the authors are able to use high dimensional environments like Atari games for their demonstation.



## Glossary
### Imitation Learning
Imitation Learning aims to use demonstrations for policy learning. It can be split into behavioral cloning and inverse reinforcement learning.

### Safe Imitation & Reinforcement Learning
The safe aspect of imitation and reinforcement learning refers to employing safe exploration strategies, thereby minimising the risk of the learnt policy. 

### Trajectory Reward Extrapolation (T-REX)
Brown et al. (2019b) propose Trajectory Reward Extrapolation (T-REX), a method that uses pairwise ranking of demonstrations in order to learn policies. The reward function learning task is transformed into a classification problem.

## Bayesian Reward Extrapolation (Bayesian REX)
Our paper proposes Bayesian Reward Extrapolation (Bayesian REX), an improved T-REX, which is not limited to only solving point estimates of the reward function, rather, it gives a probability distribution over a learned reward function.

### Making demonstration likelihood tractable

A key contribution of the B-Rex paper was to propose an alternitive likelyhood function for the demonstrations inspired by learning to rank algorthims. The normal way of defining a likelyhood for a set of demonstration state-action pairs under a reward function $R$ is: 

$$ \begin{equation}
P(D|R) = \prod_{(s,a)\in D}\pi^{\beta}_{R}(a|s) = \prod_{(s,a)\in D} \frac{e^{\beta Q^{*}_{R}(s,a)}} {\sum_{b \in A} e^{\beta Q^{*}_{R}(s,b)}}
\end{equation} $$

This equation includes the perfect quality values which requires solving the MDP analytically or sufficently aprroximate it. Both of which are too computationally expensive on bigger MDPs. 
The paper proposes instead a pairwise ranking likleyhood function, where we approximate the total demonstration likleyhood by the reward's function ability to distinguish the true pairwise order of the demonstrations:

$$ \begin{equation}
P(D,P|R_{Θ}) = \prod_{(i,j)\in D} \frac{e^{\beta R_{\theta}(\tau_i)}} {e^{\beta R_{\theta}(\tau_i)} + e^{\beta R_{\theta}(\tau_j)}}
\end{equation} $$

$R_{\theta}$ will be a neural network that we train to maximize this likelyhood. $\tau_i$ and $\tau_j$ is a pair of trajectories sampled from the demonstrations and $\beta$ is the inverse temperature parameter that models the confidence in the preference label.

### Paper's analysis of computational complexity of MCMC

# Exposition of weaknesses/strengths/potential
## Strengths
- Building on previous work (T-REX), the pairwise ranking allows for tractable sampling from the reward function posterior.
- Instead of a point estimate, Bayesian REX gives a distribution over reward functions, which allows for more in-depth policy evaluation.
- If the distribution wrongly assigns a high reward value to a poorly performing or otherwise undesirable policy, this can be easily remedied by adding this policy to the data used for MCMC sampling and giving it a low rank. Re-running the MCMC sampling will then give a new distribution which correctly assigns a low reward to this undesirable policy. 
- Policies learned with Bayesian REX are able to achieve rewards much higher than those achieved by the demonstration data the latent features were trained on. For example, in the case of the 'Breakout' game, a policy is learned that gives more than 12x improvement on the highest score in the demonstration data. 

## Weaknesses
- The RL agent can be trained using either the mean or MAP estimate of the reward function. In most cases, the agent trained with the MAP estimate outperforms the agent trained with the mean function, but for the 'Enduro' game the MAP estimate significantly underperforms the mean estimate. The authors do not investigate this behavior.

## Potential Improvements
We were intrigued by the performance graph on the different games, and especially by how MAP performance (selecting a policy based on maximimum a-postiori probability distribution of the network approximating a learned reward function) is very low in *Enduro*, whereas it results in the best performance for other games. This has motivated us to look into scenarios where MAP would *not* be the best strategy for picking a policy, and analyse the reward function distribution to find better strategies.

The authors do not provide studies on hyperparameters of their pipeline - such as embedding size. For this reason, we believe a possible (albeit minor) improvement would be experimenting with multiple hyperparameter values, especially embedding size, and observing changes in performance. This experiment would allow us to reason about what features are being encoded, and how each of them relates to the performance obtained by the model.

We would also like to attempt finetuning the embeddings themselves. This should result in a higher certainty over produced policies, as the embeddings would better represent the game elements and help the algorithm in learning new policies from the demonstrations. 

In light of the fact that the authors are using a list of demonstrations with global rankings, we also see a possible improvement in trying to change the pairwise rankings they use for the demonstrations. Increasing the rankings dimensionality to use triplet, quadruplet preferences and so on is one option. Another option would be to even use listwise preferences.

The policies generated after learning from the demonstrations can also be used for gathering further demonstrations. We consider a possible improvement could be to use these generated demonstrations to again learn a policy. This would mean, in a bootstrapping-like manner, that the algorithm would be engaged in self-play and potentially improve even more compared to the initial demonstrations.

The confidence level of evaluated policies can, as the paper itself also mentions, potentially be used for _"automatic detection" of reward hacking_. Reward hacking is an important problem in reinforcement learning and alignment research -- essentially, an agent is 'hacking a reward function' if it creates policies that maximise the defined reward function, but may not align with how the developer expects the agent to play. For example, if the agent finds software glitches that lead to high rewards, or optimize for some interim reward that leads the agent to not complete the overall objective of the task. This is a very exciting prospect, as, to our knowledge, no robust/reliable algorithm has yet been created.  However, it is hard to determine objectively what is and isn't reward hacking (couldn't software glitches be considered part of the game itself?). The authors claim that policies with high mean predicted performance and high risk/uncertainty are candidates for reward hacking policies. In their experiments, they specifically design a policy that uses reward hacking, and these traits are what sets it aside from the other tested policies.

# Describe your novel contribution

* scale model to multitask learning
* detect/predict if a game is prone to bad policies
* add causality by labeling demonstrations
* analyze feature vector dimensions and what each feature means
* change preferences from pairwise to tripletwise or listwise
* finetune learnt features to improve certainty
* conformal prediction
* use imitation learning with the AI-inferred policies

## Process
We started off by familiarizing ourselves with the codebase provided by the authors. We immediately encountered a severe issue: the codebase was highly outdated. Specifically, the authors used Python 3.6 and a mix of TensorFlow (for training and running of RL agents) and PyTorch (for training of the CNN). It was impossible to generate the demonstrations and train the CNN on a GPU within the same Python environment, because the RL agents use TensorFlow 1.x, which is outdated and no longer supported by a working CUDA version. The LISA cluster also only provides TensorFlow 2.x, which has significant API changes with respect to TensorFlow 1.x. Thus, we began splitting up the files, allowing us to work with two different environments: one in Python 3.6 with the necessary packages for generating demonstrations and training RL agents, and one in Python 3.11 for training the CNN and running the MCMC. 

Aside from the code being outdated, we also found it to be extremely difficult to work with. The CNN was defined in two seperate places, each used in a different step of the process, with slight differences each time. The code was inefficient and hard to interpret. Comments were few and far between. We also spent some time getting all of this up to a modern standard. 

Having encountered our share of code-related issues along the way, we had arrived at the second-to-last step of the pipeline: training an RL agent with our custom reward function (the result of running MCMC over the last layer of the CNN). We found that this step was computationally infeasible, because we had to run it in Python 3.6 using TensorFlow and thus no GPU support was available to us. Training the agent took a long time, it ran for an hour but seemed to be only 10% of the way at that point. Furthermore, the intermediate training results were not encouraging. However, we did not know if this was due to bad training data, our rewriting being incorrect, or simply needing to give the RL training more time. We believed that for efficient debugging, we needed a much quicker process.

Thus, we set to upgrading the code to the newest version of packages. This would allow us to run all of our code in PyTorch and on GPU. This, again, proved harder than expected. The authors mostly used the openai-baselines package for their code. However, instead of installing it via pip and using API calls, the authors had included all the files of the package in their codebase and made some changes to them. There was also no clear documentation of what exactly the authors had changed. This made upgrading to the newer version of baselines (stable-baselines3, which works on PyTorch instead of TensorFlow) difficult in its own way, because we were unsure of what exactly we needed to add ourselves and how.

Furthermore, we decided from here to work with an environment simpler than the Atari environments the authors used. We created a first version using the Cartpole environment, one of the most simple RL environments. This had two significant advantages: 
1. The data would be very low-dimensional, meaning we could train our network more quickly
2. Training an RL agent in a Carpole environment is much faster than in an Atari environment.

Both of these speedups were essential, seeing as we only had two weeks left at this point and we had not been able to start on any extension yet, due to the codebase constantly changing.

This is where we currently are at. 