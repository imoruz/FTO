https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

Rather than pre-processing all relevant data up front, agents built with the “just in time” approach maintain lightweight identifiers (file paths, stored queries, web links, etc.) and use these references to dynamically load data into context at runtime using tools. Anthropic’s agentic coding solution Claude Code uses this approach to perform complex data analysis over large databases. The model can write targeted queries, store results, and leverage Bash commands like head and tail to analyze large volumes of data without ever loading the full data objects into context. This approach mirrors human cognition: we generally don’t memorize entire corpuses of information, but rather introduce external organization and indexing systems like file systems, inboxes, and bookmarks to retrieve relevant information on demand.

Letting agents navigate and retrieve data autonomously also enables progressive disclosure—in other words, allows agents to incrementally discover relevant context through exploration. Each interaction yields context that informs the next decision: file sizes suggest complexity; naming conventions hint at purpose; timestamps can be a proxy for relevance. Agents can assemble understanding layer by layer, maintaining only what's necessary in working memory and leveraging note-taking strategies for additional persistence. This self-managed context window keeps the agent focused on relevant subsets rather than drowning in exhaustive but potentially irrelevant information.

 To enable agents to work effectively across extended time horizons, we've developed a few techniques that address these context pollution constraints directly: compaction, structured note-taking, and multi-agent architectures.



# Compaction

Compaction is the practice of taking a conversation nearing the context window limit, summarizing its contents, and reinitiating a new context window with the summary. Compaction typically serves as the first lever in context engineering to drive better long-term coherence. At its core, compaction distills the contents of a context window in a high-fidelity manner, enabling the agent to continue with minimal performance degradation.

In Claude Code, for example, we implement this by passing the message history to the model to summarize and compress the most critical details. The model preserves architectural decisions, unresolved bugs, and implementation details while discarding redundant tool outputs or messages. The agent can then continue with this compressed context plus the five most recently accessed files. Users get continuity without worrying about context window limitations.

The art of compaction lies in the selection of what to keep versus what to discard, as overly aggressive compaction can result in the loss of subtle but critical context whose importance only becomes apparent later. For engineers implementing compaction systems, we recommend carefully tuning your prompt on complex agent traces. Start by maximizing recall to ensure your compaction prompt captures every relevant piece of information from the trace, then iterate to improve precision by eliminating superfluous content.


# Structured note-taking

Structured note-taking, or agentic memory, is a technique where the agent regularly writes notes persisted to memory outside of the context window. These notes get pulled back into the context window at later times.

This strategy provides persistent memory with minimal overhead. Like Claude Code creating a to-do list, or your custom agent maintaining a NOTES.md file, this simple pattern allows the agent to track progress across complex tasks, maintaining critical context and dependencies that would otherwise be lost across dozens of tool calls.




Claude said:

# Family 1 — Summarization (roll your own). 
This is the Claude Code / LangGraph approach, and it's genuinely simple to implement yourself. The algorithm is a "summary buffer": keep a running summary plus the last N raw turns; when the aggregated input crosses a token threshold, send the older portion to an LLM with a structured prompt and fold the result back in. The core loop is small:
def compact(messages, running_summary, keep_last=5, threshold=8000):
    if token_count(messages) < threshold:
        return messages
    head, tail = messages[:-keep_last], messages[-keep_last:]
    summary = llm.summarize(
        prior_summary=running_summary,
        to_compact=head,
        schema=["user_intent", "key_decisions", "open_问题", "current_state", "next_step"]
    )
    return [as_message(summary)] + tail
The art is entirely in that schema and prompt — what you force it to preserve (decisions, unresolved issues, file/entity names) versus discard. Anthropic's own advice is to first maximize recall on real traces, then tighten precision. Because you control every parameter, you can ablate them in your eval, which is hard to do with a framework's built-in.

# Family 2 — Token pruning (third-party: LLMLingua). 

This is a different compaction algorithm and a citable one, which is valuable for a thesis because it gives you a second method to benchmark against summarization. It's open-source from Microsoft Research (EMNLP 2023), pip install llmlingua. It uses a small language model like GPT2-small or LLaMA-7B to identify and remove unimportant tokens from prompts — it scores each token's importance and drops the lowest-ranked ones, lossy by design. Reported numbers: up to 20× compression with about a 1.5-point accuracy drop, though 20× is peak — production workloads more commonly see 4–10× before tasks start breaking. There are three variants worth knowing: the original LLMLingua, LLMLingua-2, a BERT-size token-classification model distilled from GPT-4 that is task-agnostic and 3–6× faster, and LongLLMLingua, aimed at long-context and RAG scenarios, improving RAG performance by up to 21.4% using a quarter of the tokens. It's a drop-in inside your adapter: feed it the aggregated input text, get back compressed text, pass that on. Microsoft + 4
These two map cleanly onto the taxonomy from earlier: summarization is the "rewrite and discard" approach, LLMLingua is "compress token-by-token without rewriting." A genuinely interesting thesis design is a 2×2: {summarization, LLMLingua} × {LangGraph, DevAll}, with the compaction logic living in NodeAdapter so it's identical across both frameworks. Then DevAll's native retrieval-memory and LangGraph's native SummarizationMiddleware become two extra baselines.
One caution for the adapter: after compacting, the message sequence still has to be valid for whatever provider you're calling (e.g. roles in the right order, no orphaned tool-call/tool-result pairs). That's the detail that trips people up when they bypass the framework's own message handling.
Want me to sketch out a full NodeAdapter-integrated compaction module — one that wraps both a summarizer and an LLMLingua path behind a common interface so you can switch methods with a flag? I'd need to know roughly what shape NodeAdapter passes around (raw strings, or structured message objects with roles/tool calls).