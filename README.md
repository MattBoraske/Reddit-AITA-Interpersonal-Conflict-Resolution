# CPSS-24-Reddit-Conflict-Resolution

Tasks
- Create the raw reddit dataset (NTA, YTA, ESH, NAH)
- Implement extra cleaning as necessary
- Train the four Llama3 models on each type of data
- Find a generic set of training data to use as a baseline to show that each of the four llama3 models has varying levels of perplexity based on what they were finetuned on
    - Also compare them to each others test sets
    - Use these baseline differences to establish boundaries between NTA/YTA/ESH/NAH
- Calculate ambiguity scores using these metrics
- Finetunings from there...
    - All data and low ambiguity versions?
    - compare to baseline super-large LLM like GPT4o
- Evaluate by feeding outputs to each of the four llama3 AITA class-specific models. whichever provides the lowest perplexity score is the class (find a way to give confidence of class using mixture of 4 scores?)
    - Also evaluate by grabbing AITA classification at beginning of sentence
        - does this match w/ the justification class using the llama3 AITA class-specific models?
