Testing whether modern LLMs can reliably replicate results from classic and recent behavioral economics experiments (“homo silicus”). We pull the original experiment text from pre-registered studies, have an LLM act as survey respondents across treatment and control groups, convert its outputs into the same numeric variables as the original studies, and then statistically compare LLM results to human data.
Proof of concept: replicated three effects (conformity, loss aversion, and tax compliance) using a lightweight GPT model. Ran with ~500 simulated participants per group, The main outcome is whether the LLM reproduces the direction, magnitude, and distribution of the original treatment effects. (mixed results, some signal maybe)
Now doing with a wider batch of studies, in the process of testing out scraping pipeline

Full doc here: https://docs.google.com/document/d/1vCyfUhEbOf_YhtKaQOY9ICtdfhbauUEyaB0uIcXun-M/edit?usp=sharing

links/names of papers read (+ doc contents):

initial:
Large Language Models as Simulated Economic Agents: What Can We Learn from Homo Silicus?  
Generative Agents: Interactive Simulacra of Human Behavior  
Computational Agents Exhibit Believable Humanlike Behavior  
Generative Agent Simulations of 1,000 People  
Predicting Results of Social Science Experiments Using Large Language Models  

metrics:
https://arxiv.org/abs/1904.09675
https://aclanthology.org/2020.acl-main.704/
https://proceedings.mlr.press/v37/kusnerb15.html
https://pure.psu.edu/en/publications/automatic-analysis-of-syntactic-complexity-in-second-language-wri

distributional comparisons:
https://jmlr.csail.mit.edu/papers/v13/gretton12a.html
https://arxiv.org/abs/1806.00035
https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-1/On-Information-and-Sufficiency/10.1214/aoms/1177729694.full
https://aclanthology.org/J93-1003/
https://www.jstor.org/stable/2236101

studies:
https://www.econstor.eu/bitstream/10419/325465/1/vfs-2025-pid-129297.pdf
[there are more that are not listed, will be linked by Aywa501 in the future]

new:
https://www.pnas.org/doi/10.1073/pnas.2313925121
