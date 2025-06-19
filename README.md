# NLP_project
Project for MTS-NLP cources

\documentclass{article}
\usepackage{amsmath,amsthm,amssymb}
\usepackage{mathtext}
\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{graphicx}
\usepackage{hyperref}

\title{AI-Powered Recipe Generation}
\author{Dmitrii Zagorulia, Iskander Shamigulov}
\date{June 2025}

\begin{document}
\maketitle
\begin{abstract} 
    % Recipe generation from recipe titles only is currently unsolved, as state of the art models require both recipe titles and ingredients lists for instruction generation (Lee et al., 2020). This project investigates if a number of different architectures such as Long Short-Term Memory (LSTM) encoder-decoders, LSTM decoders, or Transformer-based decoders, can produce meaningful ingredient lists when given recipe titles only. The recipe titles and generated ingredients are then passed into an existing recipe instruction generation framework to produce cooking instructions (Liu et al., 2022). Our best ingredient generation model yielded qualitatively coherent ingredients lists with BLEU score 11.2 and F1 score 8.9, however, the BLEU and ROUGE-L scores for the final recipe instructions with ingredients from our selected transformer decode were 3.4 and 22.7. The baseline plug-and-play recipe instruction generation framework, relying on RecipeGPT and ground truth recipe title and ingredients demonstrates BLEU and ROUGE-L scores of 13.73 and 39.1 respectively for instruction generation. Since BLEU and ROUGE-L performance are influenced by n-gram matching and order, further evaluation would be required with metrics such as Semantic Textual Similarity (STS) to evaluate the meaning of the produced ingredients in thee context of each recipe.

    This document will provide you with guidelines for your project final report. You will learn how to structure the report and present your results. Use this field for the short description of your work. Please provide a link to your project code right here: \url{https://github.com/Shamighoul/NLP_project} 
\end{abstract}

\section{Introduction}
% In today's multicultural and diverse world, food preferences and dietary restrictions play a crucial role in people's daily lives. Religious dietary laws, such as \emph{Halal} (Islam), \emph{Kosher} (Judaism), \emph{vegetarianism} (Hinduism, Buddhism, Lenten), and other faith-based restrictions, significantly influence what individuals can or cannot eat. However, finding suitable recipes that align with both available ingredients and religious requirements can be challenging.  

% This problem is relevant for several reasons:
% \begin{enumerate}
%     \item{\textbf{Cultural and Religious Sensitivity} -- Many people strictly follow dietary laws due to their faith, and violating these rules can be offensive or unacceptable.}
%     \item{\textbf{Reducing Food Waste} -- By suggesting recipes based on available ingredients, the system encourages efficient use of food resources.}
%     \item{\textbf{Convenience \& Personalization} -- Traditional recipe apps often lack filters for religious constraints, forcing users to manually check each ingredient. An AI that automates this process saves time and improves accessibility.}
%     \item{\textbf{Global Applicability} -- With increasing multicultural interactions (travel, migration, international cuisine), such a tool can assist both individuals and businesses (restaurants, catering) in offering compliant meal options.}
% \end{enumerate}
\subsection{Team}

\textbf{Dmitrii Zagorulia} \\
Prepared this document. \\
Search for articles and information. \\
Models training\\
\textbf{Iskander Shamigulov}\\
Prepared this document. \\
Prepared this document. \\
Search for articles and information \\


\section{Related Work} \label{sec:related}
To the best of our knowledge, there is not much existing academic literature on generating recipe instructions from recipe ingredients. RecipeGPT can generate ingredients, but requires both the recipe title and recipe instructions as input \cite{recipegpt}. The majority of existing recipe generation models require both ingredient title and ingredients. 

For these methods, there is much work on controlling recipe instruction generation. Some approaches have included modifying the architecture of and then retraining pre-trained language models, including CTRL \cite{ctrl}, and POINTER \cite{pointer}, which are effective but require a lot of computational resources and task-specific labelled data (Liu et al., 2022). 
There are also fine-tuning methods such as ParaPattern \cite{bostrom2021} and prefix-tuning \cite{li2021}, which require less computation and often perform adequately, but cannot enforce hard constraints on the outputs directly (Liu et al., 2022). Finally, there are post-processing methods such as PPLM \cite{pplm}, FUDGE \cite{fudge}, and neurologic decoding \cite{lu2021}, which use a separate guiding module to control output. These methods require the least computational resources, and are flexible in design because the guidance module is separate from the pre-trained language model (Liu et al., 2022). 

% In our paper, we utilize the plug and play recipe generation method (Liu et al., 2022) because it achieves state of the art performance and is open sourced. We also focus on experimenting with different model architectures, training methods, and simple decoding methods (top-k, top-p, beam search) for ingredient generation.

% vij2025}


% In this section, you will describe in details the existing approaches to the problem you work on. For each approach, you need to provide a reference. 

% is a sample reference to the previous art. is a sample reference in Russian.

\section{Model Description}
GPT-2 \cite{gpt2}, vanilla LSTM \cite{lstm}

Here you need to write a detailed description of your approach. It is important to mention that this description should give more details than the descriptions from section \ref{sec:related}\footnote{This is an example of internal references and footnotes at the same time.}. 

You will likely be providing a figure to better present your approach. A sample circle is presented on Fig.

The other possible contents of this section are formulae. They could be on a new line:
$$S=\pi r^2,$$
or they could be inline, e.g. if you want to describe the used variables, like $S$ is an area of a circle, while $r$ is its radius. 

\section{Dataset}
We began by exploring several publicly available recipe collections on Kaggle -- namely the \href{https://www.kaggle.com/c/whats-cooking/overview}{What’s Cooking?} challenge dataset, the \href{https://www.kaggle.com/datasets/hugodarwood/epirecipes/data}{Epicurious} corpus, and \href{https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_interactions.csv}{Food.com}’s recipes and interaction logs -- but found that none included detailed cooking instructions, which are essential for our purposes. We then turned to Recipe1M+ and RecipeNLG \cite{recipe1m_ext, recipenlg}, both of which offer rich procedural text; however, we later discovered more up-to-date alternatives. Ultimately, we selected the A3M2+ dataset \cite{a3m2+}, an enhanced successor to A3M2 \cite{a3m2}, itself an extension of RecipeNLG, as the most suitable and current resource for our task.

% In this section, you need to describe the dataset(s) you are working with. 
% An example dataset we will use is WikiText-2. Please mention a paper where it was presented, e.g. WikiText-2 was presented in. Please provide guidance on how to obtain the dataset\footnote{The one way to do it is to include a link to the website, where it could be downloaded from. Like~\href{https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/}{this}.}. It is important to mention that the dataset you use must be available for the research purposes. So please make sure about that.

% Your description will likely be including a table. On the Tab.~\ref{tab:statistics} you can see the statistics for the mentioned dataset. It is important to notice that there is a split of the dataset and this split is covered by the description.

% \begin{table}[tbh!]
% \begin{center}
% \begin{tabular}[t]{|l|ccc|}
% \hline
% %\cline{2-4}
%  & Train & Valid & Test \\
% \hline
% Articles & 600 & 60 & 60  \\
% Tokens& 2,088,628 & 217,646 & 245,569 \\
% Vocabulary size & \multicolumn{3}{c|}{33,278} \\
% Out of Vocab rate &  \multicolumn{3}{c|}{2.6\%}  \\
% \hline
% \end{tabular}
% \caption{Statistics of the WikiText-2. The out of vocabulary (OoV) rate notes what percentage of tokens have been replaced by an $\langle unk \rangle$ token. The token count includes newlines which add to the structure of the dataset.}
% \label{tab:statistics}
% \end{center}
% \end{table}

% If you were collecting the dataset on your own please describe the collection procedure, like criteria were used to filter the documents, the pre-processing steps, etc. It is preferable that you release your dataset for the public, but you are not obliged to do this. Please make sure that you have legal rights to collect and distribute the data you were working with. We recommend you to look at C4Corpus and how it is licensed to make your corpora. C4Corpus is described in.

\section{Experiments}

This section should include several subsections.
\subsection{Metrics}

To assess the performance of our AI-powered recipe generation model, we employed several well-established metrics in natural language processing (NLP) and text generation tasks. These metrics help quantify the quality, relevance, and coherence of the generated recipes compared to human-written references. Below, we describe the key metrics used in our evaluation.  \\


\textbf{BLEU (Bilingual Evaluation Understudy)}\\
BLEU measures the precision of n-gram matches between the generated text and reference texts, focusing on lexical similarity \cite{bleu}.\\
- Helps evaluate how closely the generated recipes match human-written ones in terms of word choice and phrasing.  \\
- Particularly useful for assessing ingredient lists and step-by-step instructions. 

\[
BLEU = BP \cdot \exp\left( \sum_{n=1}^{N} w_n \log p_n \right)
\]  
where:  \\
\(BP\) (Brevity Penalty) penalizes overly short outputs:  
  \[
  BP = 
  \begin{cases} 
  1 & \text{if } c > r \\
  e^{(1 - r/c)} & \text{if } c \leq r 
  \end{cases}
  \]  
  (\(c\) = length of candidate text, \(r\) = effective reference length)  \\
\(p_n\) =  the modified n-gram precision for n-grams of size \(n\).  \\
\(w_n\) = a weight (typically uniform: \(w_n = 1/N\)).  \\


\textbf{ROUGE (Recall-Oriented Understudy for Gisting Evaluation)} \\
ROUGE evaluates the recall of overlapping n-grams between generated and reference texts, emphasizing content coverage \cite{rouge}. We used three variants:  

ROUGE-1 measures unigram (single-word) overlap.  

ROUGE-2 measures bigram (two-word sequence) overlap.  \\

\[
ROUGE\text{-}N = \frac{\sum_{S \in \{Ref\}} \sum_{gram_n \in S} Count_{match}(gram_n)}{\sum_{S \in \{Ref\}} \sum_{gram_n \in S} Count(gram_n)}
\]  
where:  \\
\(Count_{match}(gram_n)\) = the number of n-grams co-occurring in the candidate and reference.  \\
\(Count(gram_n)\) = the total n-grams in the reference.  \\

\textbf{ROUGE-L (Longest Common Subsequence)}\\
- Measures the longest sequence of words (not necessarily contiguous) shared between texts, capturing structural similarity.  \\
ROUGE-L evaluates the logical flow of recipe steps (e.g., "mix ingredients before baking"). 

\[
ROUGE\text{-}L = \frac{(1 + \beta^2) R_{lcs} P_{lcs}}{R_{lcs} + \beta^2 P_{lcs}}
\]  
where:  \\
\(R_{lcs} = \frac{LCS(X,Y)}{m}\) (recall), \(P_{lcs} = \frac{LCS(X,Y)}{n}\) (precision).  \\
\(LCS(X,Y)\) = length of longest common subsequence between candidate \(X\) and reference \(Y\).  \\
\(m, n\) = lengths of reference and candidate, respectively.  \\
\(\beta\) = balances recall/precision (typically \(\beta = 1\)).  \\


\textbf{Metric Selection Justification}\\
- BLEU + ROUGE together provide a balanced view of precision (BLEU) and recall (ROUGE). \\ 
- Since recipe generation requires both accuracy (correct ingredients) and coherence (logical steps), these metrics complement each other.  \\
- Unlike single-score metrics (e.g., perplexity), they offer interpretable insights into specific weaknesses (e.g., missing ingredients or disordered steps).  \\

\textbf{Limitations}\\
- These metrics focus on surface-level text overlap and may not fully capture semantic correctness or religious compliance.  \\
- Future work could incorporate human evaluation or task-specific metrics (e.g., dietary rule violation rate).  \\


\subsection{Experiment Setup}
Secondly, you need to describe the design of your experiment, e.g. how many runs there were, how the data split was done. The important details of your model, like hyper-parameters used in the experiments, and so on.

The considered model uses the classical \textbf{Sequence-to-Sequence} (Seq2Seq) approach with bidirectional \textbf{LSTM} layers. The following are the key aspects of its implementation, including architectural solutions, training methods, and data processing.\\
10e5 samples were taken for the experiment. \\
Optimizer: Adam with lr=1e-3.\\
Loss function: CrossEntropyLoss (ignores padding tokens).\\
Decoding: greedy, with a length limit of max len=100.\\
batch size = 32, epochs	= 10\\
The Encoder: Embedding layer: dimension emb size=128. \\
LSTM: hidden dimension hide size=256, 1 layer.\\
The Decoder: Similar parameters, but with a linear layer (nn.Linear) for predicting tokens.\\




\subsection{Baselines}
Another important feature is that you could provide here the description of some simple approaches for your problem, like logistic regression over TF-IDF embedding for text classification. The baselines are needed is there is no previous art on the problem you are presenting.

\section{Results}

\begin{table}[ht]
\centering
% \renewcommand{\arraystretch}{1.2} % increase row height for readability
% \setlength{\tabcolsep}{10pt}      % add column padding
\begin{tabular}{|c|c|c|c|c|}
\hline
Model & BLEU & ROUGE-1 & ROUGE-2 & ROUGE-L \\
\hline 
LSTM & 0.0253 & 0.2694 & 0.0661 & 0.1812 \\
GPT-2 (medium) & 0.0164 & 0.1154 & 0.0205 & 0.0674 \\
GPT-2 (medium + LoRA) & 0 & 0 & 0 & 0 \\
\hline
\end{tabular}
\caption{Evaluation metrics for GPT-2 trained on $10^5$ recipes without optimization}
\label{tab:gpt_2}
\end{table}

In this section, you need to list and describe the achieved results. It is crucial to have the results of the experiments for the other approaches. This is needed to be able to compare your results with some competitors. Most preferably, you should provide some references with results on the same problem.

Almost inevitably the results are presented as a table, but it is also possible to have a graph, i.e. a figure.

You need also to provide an interpretation of the presented results, to describe some features. E.g. your approach shows higher results on the short texts or by one metric instead of another.

Also in this section, you could provide some results for your model inference. The samples could be found in Tab.~\ref{tab:output}.

\begin{table}[!tbh]
    \centering
    \begin{tabular}{|c|}
\hline
Это пример вывода вашей модели на русском.\\
This is a sample output of your model in English.
\\
\hline
    \end{tabular}
    \caption{Output samples.}
    \label{tab:output}
\end{table}

\section{Conclusion}
In this section, you need to describe all the work in short: what you have done and what has been achieved. E.g. you have collected a dataset, made a markup for it and developed a model showing the best results compared to other models. 

\bibliographystyle{apalike}
\bibliography{lit}
\end{document}
