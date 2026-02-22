# Council of Alphas, Related Work and Research Gaps (short literature review)

## 1. LLMs for trading and strategy discovery (2024–2025)

Recent work increasingly uses LLMs to generate and iterate on trading artifacts (factors, rules, or code), often in a closed loop: propose a candidate, evaluate it via backtests or proxies, then refine based on feedback.

**Alpha and factor discovery.**
AlphaAgent argues that LLM-driven alpha mining is promising but can over-rely on existing knowledge, producing *homogeneous* factors that heighten crowding and accelerate alpha decay, and it introduces regularized exploration mechanisms (for example AST-based originality constraints) to counter this effect.[1]
Automate Strategy Finding with LLM in Quant Investment proposes a three-stage pipeline (generation, multi-agent evaluation, and weight optimization) and reports improved backtest performance on its selected benchmarks.[2]

**Agentic trading decision systems.**
Multi-agent architectures such as FinCon (NeurIPS 2024) and TradingAgents (arXiv 2024/2025) emulate role-specialized “trading firms” (analysts, risk control, etc.) and report gains over baselines in backtest settings, but they also highlight non-trivial orchestration and compute costs for multi-agent decision-making.[3,4]

**Takeaway.**
By 2025, the literature supports the narrower claim that “generate → evaluate → refine” loops can improve candidates *within* specific experimental setups, but reported gains are sensitive to asset universe, horizon, leakage controls, and cost assumptions.[1–4]

## 2. Robust evaluation exposes fragility and generalization gaps

More recent benchmarks argue that many positive results are sensitive to narrow windows or evaluation bias.

FINSABER reassesses LLM timing-style strategies across longer periods and broader universes and finds that previously reported advantages often deteriorate under broader and longer evaluation, with regime-dependent failures (overly conservative in bull markets, overly aggressive in bear markets).[5]
AI-Trader introduces a live, data-uncontaminated benchmark across U.S. stocks, A-shares, and cryptocurrencies, and reports that most tested agents show poor returns and weak risk management, with risk control emerging as a key driver of cross-market robustness.[6]

**Implication.**
A defensible framing is that LLM pipelines can generate plausible strategies and sometimes improve backtest metrics, but robustness under stronger, contamination-resistant evaluation remains an open problem.[5,6]

## 3. Diversity collapse (“mode collapse”) as a concrete obstacle

A recurring concern is that LLM-based strategy generation can concentrate on a narrow subset of familiar solutions, limiting novelty and increasing crowding risk.

**In finance.**
AlphaAgent explicitly identifies factor homogenization as a core issue for LLM-based alpha mining and motivates novelty- and decay-aware regularization to sustain performance.[1]

**In general LLM post-training and prompting.**
RLHF has been shown to change generalisation behavior and reduce diversity in LLM outputs in controlled studies.[9]
Separately, prompt formatting and strict templates can induce “diversity collapse,” reducing variability in form and content.[10]
Recent work also explicitly analyzes alignment-driven mode collapse and proposes inference-time mitigation strategies (for example “Verbalized Sampling”), reinforcing that the phenomenon is measurable and not merely anecdotal.[11]
Conformative decoding is another approach aimed at reintroducing diversity in instruction-tuned models while maintaining quality.[12]

**Implication.**
For strategy generation, reduced diversity can manifest as crowded, overly similar candidate strategies, even if local refinement improves short-horizon backtests.[1,10,11]

## 4. Iterative refinement and evidence-based verification

Many LLM agent systems rely on iterative self-critique and refinement to improve outputs over multiple steps. Self-Refine and Reflexion demonstrate that reflection-style loops can improve quality across diverse tasks without additional model training.[14,15] However, these loops do not inherently guarantee that critiques are grounded in verifiable evidence, and ungrounded revisions can preserve or amplify mistakes. Verification-oriented methods such as Chain-of-Verification (CoVe) explicitly reduce hallucinations by forcing the model to plan and answer fact-checking questions before revising its draft.[16] In financial strategy search, where evaluations are expensive and failure modes are regime-specific, there remains room for refinement mechanisms that are constrained to be *evidence-locked* and *surgical* (small, auditable changes justified by explicit diagnostic failures).

## 5. Evolutionary search and LLM-guided code evolution

Evolutionary computation provides explicit mechanisms for exploration (population, selection, mutation, crossover), and recent systems increasingly combine LLMs with evolutionary structure.

**In trading.**
ProFiT frames trading as program search, combining LLM-driven strategy rewriting with evolutionary ideas and walk-forward validation inside a closed feedback loop, reporting improvements over seed strategies and baseline comparators in its experiments.[7]

**Beyond finance.**
AlphaEvolve uses an evolutionary process coupled to automated evaluators to iteratively improve code and reports advances on challenging algorithmic tasks, including improved procedures for multiplying two 4×4 complex-valued matrices using 48 scalar multiplications.[8,13]

**Limitation.**
Evolutionary approaches can be evaluation intensive (population × generations), which becomes costly when each candidate evaluation includes backtests and/or multi-agent LLM tool usage.[7,8]

## 6. The gaps we target

Taken together, the literature suggests three complementary gaps:

1) **Exploration gap in LLM strategy generation:** closed-loop refinement exists, but systems can converge to homogeneous solutions and often show weak robustness under broader or live-style evaluation.[1,5,6,10]

2) **Efficiency gap in evolutionary search:** evolutionary approaches help preserve diversity and novelty, but can become prohibitively expensive when they require large numbers of candidate evaluations.[7,8]

3) **Evidence-grounded refinement gap:** reflection-style refinement is widely used, but there is limited work on *constraining* strategy edits to be evidence-locked and surgical, especially when the “evidence” is a structured set of backtest diagnostics rather than free-form text.[14–16]

## 7. Our approach in one paragraph (Council of Alphas)

We propose an evolutionary multi-agent framework that preserves diversity by construction (specialists constrained to distinct strategy families plus niche-preserving selection), while using LLMs as “smart operators” that decide when and how to mutate, hybridize, and refine candidates based on structured diagnostics. Empirically, we target fee-inclusive backtests on 1-hour Solana (SOL-USD) candlesticks and report both global metrics and regime-stratified diagnostics (session, trend, volatility). To reduce ungrounded iteration, our refinement step is evidence-locked: the critic must cite concrete diagnostic failures and propose a single surgical code change, which is validated with accept/revert gates to prevent degradation.

---

## References

[1] Z. Tang, Z. Chen, J. Yang, J. Mai, Y. Zheng, K. Wang, J. Chen, and L. Lin. *AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay.* KDD 2025 / arXiv:2502.16789. https://arxiv.org/abs/2502.16789

[2] Z. Kou, H. Yu, J. Luo, J. Peng, X. Li, C. Liu, J. Dai, L. Chen, S. Han, and Y. Guo. *Automate Strategy Finding with LLM in Quant Investment.* Findings of EMNLP 2025 (ACL Anthology). https://aclanthology.org/2025.findings-emnlp.1005/

[3] Y. Yu et al. *FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making.* NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/f7ae4fe91d96f50abc2211f09b6a7e49-Paper-Conference.pdf

[4] Y. Xiao, E. Sun, D. Luo, and W. Wang. *TradingAgents: Multi-Agents LLM Financial Trading Framework.* arXiv:2412.20138. https://arxiv.org/abs/2412.20138

[5] W. W. Li, H. Kim, M. Cucuringu, and T. Ma. *Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?* (FINSABER). arXiv:2505.07078. https://arxiv.org/abs/2505.07078

[6] T. Fan, Y. Yang, Y. Jiang, Y. Zhang, Y. Chen, and C. Huang. *AI-Trader: Benchmarking Autonomous Agents in Real-Time Financial Markets.* arXiv:2512.10971. https://arxiv.org/abs/2512.10971

[7] M. Siper, A. Khalifa, L. Soros, M. Nassir, and J. Togelius. *ProFiT: Program Search for Financial Trading.* SSRN (2025). https://ssrn.com/abstract=5889762 (DOI: 10.2139/ssrn.5889762)

[8] A. Novikov et al. *AlphaEvolve: A coding agent for scientific and algorithmic discovery.* arXiv:2506.13131. https://arxiv.org/abs/2506.13131

[9] R. Kirk, I. Mediratta, C. Nalmpantis, J. Luketina, E. Hambro, E. Grefenstette, and R. Raileanu. *Understanding the Effects of RLHF on LLM Generalisation and Diversity.* ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/5a68d05006d5b05dd9463dd9c0219db0-Abstract-Conference.html

[10] L. Yun, C. An, Z. Wang, L. Peng, and J. Shang. *The Price of Format: Diversity Collapse in LLMs.* Findings of EMNLP 2025 (ACL Anthology). https://aclanthology.org/2025.findings-emnlp.836/

[11] J. Zhang, S. Yu, D. Chong, A. Sicilia, M. R. Tomz, C. D. Manning, and W. Shi. *Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity.* arXiv:2510.01171. https://arxiv.org/abs/2510.01171

[12] M. Peeperkorn, T. Kouwenhoven, D. Brown, and A. Jordanous. *Mind the Gap: Conformative Decoding to Improve Output Diversity of Instruction-Tuned Large Language Models.* arXiv:2507.20956. https://arxiv.org/abs/2507.20956

[13] Google DeepMind. *AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms.* Blog post (May 14, 2025). https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/

[14] A. Madaan et al. *Self-Refine: Iterative Refinement with Self-Feedback.* arXiv:2303.17651 / NeurIPS 2023. https://arxiv.org/abs/2303.17651

[15] N. Shinn et al. *Reflexion: Language Agents with Verbal Reinforcement Learning.* arXiv:2303.11366 / NeurIPS 2023. https://arxiv.org/abs/2303.11366

[16] S. Dhuliawala et al. *Chain-of-Verification Reduces Hallucination in Large Language Models.* Findings of ACL 2024 (ACL Anthology). https://aclanthology.org/2024.findings-acl.212/
