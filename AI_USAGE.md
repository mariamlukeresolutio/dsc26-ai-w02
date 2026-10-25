# AI usage

<!-- Replace every comment below with your own words. Short is fine. The grader checks only
     that this file exists and has been changed from this template; your Track Mentor reads it. -->

## Tools I used, and for which functions

I used Gemini/Copilot to discuss the implementation of the functions in metrics.py and audit.py. The AI helped me think through input validation, threshold selection, leakage detection, and edge cases described in the specification.

## One place where the AI was wrong, incomplete, or not what the spec said

One AI suggestion did not fully account for the tie-breaking rule in best_threshold(). The specification requires returning the highest threshold when multiple thresholds have the same expected cost. I noticed the issue by reviewing the task instructions and updated the implementation to match the required behavior.

## The function I wrote without AI, and why

I did not write any function completely without AI assistance. I used AI throughout the task for explanations, implementation ideas, and code reviews. However, I verified the generated solutions against the project specification and adjusted the code when needed to make sure it matched the required behavior.