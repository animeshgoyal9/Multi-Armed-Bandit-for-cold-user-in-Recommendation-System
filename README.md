# Data Science Lab

The Problem — Cold Start

So, what is the cold start problem? The term derives from cars. When it’s really cold, the engine has problems with starting up, but once it reaches its optimal operating temperature, it will run smoothly. With recommendation engines, the “cold start” simply means that the circumstances are not yet optimal for the engine to provide the best possible results. Our aim is to minimize this time to heat the engine. The two distinct categories of cold start: product cold start and user cold starts. In this blog, we are concentrating on the user cold start problem.

To solve this problem, we are introducing the concept of using Multi-Armed bandit

Multi-Armed Bandit (MAB)


Multi-Armed Bandit Problem
Multi-armed bandit problem is a classical problem that models an agent (or planner or center) who wants to maximize its total reward by which it simultaneously desires to acquire new knowledge(“exploration”) and optimize his or her decisions based on existing knowledge(“exploitation”). MAB problem captures the scenario where the gambler is faced with a trade-off between exploration, pulling less explored arms optimistically in search of an arm with better reward, and exploitation, pulling the arm known to be best till the current time instant, in terms of yielding the maximum reward.

MAB for cold users

Our goal is to use different bandit algorithms to explore/ exploit optimal recommendations for the user. There are several MAB algorithms, each favoring exploitation over exploration to different degrees. Three of the most popular are Epsilon Greedy, Thompson Sampling, and Upper Confidence Bound 1 (UCB-1). 

