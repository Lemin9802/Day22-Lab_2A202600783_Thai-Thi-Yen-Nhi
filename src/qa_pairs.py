"""
50 câu hỏi và đáp án chuẩn (ground truth) dựa trên PDF:
"The New SDLC With Vibe Coding"

SAMPLE_QUESTIONS : danh sách câu hỏi — dùng cho Bước 1 & 2
QA_PAIRS         : câu hỏi + đáp án chuẩn — dùng cho Bước 3 (RAGAS evaluation)
"""

QA_PAIRS = [
    {
        "question": "What is the main shift described in The New SDLC With Vibe Coding?",
        "reference": "The main shift is from writing code in precise syntax to expressing intent, while intelligent systems translate that intent into working software.",
    },
    {
        "question": "How is the developer's role changing in the new SDLC?",
        "reference": "The developer's role is shifting from primary implementor to system designer, quality arbiter, and reviewer of AI-generated output.",
    },
    {
        "question": "What is an AI agent?",
        "reference": "An AI agent is a software system that perceives a goal, plans steps, takes actions through tools, observes the results, and iterates until the goal is met or a stopping condition is reached.",
    },
    {
        "question": "What are the five parts of an AI agent?",
        "reference": "The five parts of an AI agent are the model, tools, memory, orchestration, and deployment.",
    },
    {
        "question": "What is the role of the model in an AI agent?",
        "reference": "The model is the reasoning engine that reads the current context, decides what should happen next, and produces the next thought, tool call, or message.",
    },
    {
        "question": "What is the role of tools in an AI agent?",
        "reference": "Tools connect the model to the world, including APIs, code execution, databases, and other agents it can delegate to.",
    },
    {
        "question": "What is the role of memory in an AI agent?",
        "reference": "Memory is the state that lets an agent recall past interactions, retrieve project-specific rules, and retain context across sessions.",
    },
    {
        "question": "What is the role of orchestration in an AI agent?",
        "reference": "Orchestration is the code that runs the agent loop by assembling context, dispatching tool calls, capturing results, and deciding whether to continue.",
    },
    {
        "question": "What is the role of deployment in an AI agent?",
        "reference": "Deployment turns the prototype into a service by providing hosting, identity, observability, and production infrastructure.",
    },
    {
        "question": "What is vibe coding?",
        "reference": "Vibe coding is an AI-assisted programming approach where a developer describes what they want in natural language, accepts AI output, and often feeds errors back to the AI for fixes.",
    },
    {
        "question": "Why did the term vibe coding become popular?",
        "reference": "The term became popular because it captured a way many developers were already working: prompting AI for code, accepting outputs, and iterating by feeding back errors.",
    },
    {
        "question": "What is agentic engineering?",
        "reference": "Agentic engineering is the disciplined end of AI-assisted development, where AI works within structured specifications, tests, constraints, feedback loops, and human oversight.",
    },
    {
        "question": "What is the key differentiator between vibe coding and agentic engineering?",
        "reference": "The key differentiator is not whether AI is used, but how much structure, verification, and human judgment surrounds the AI's output.",
    },
    {
        "question": "How does verification differ between vibe coding and agentic engineering?",
        "reference": "In vibe coding, verification is optional and based on whether the code seems to work. In agentic engineering, tests and evaluations systematically verify deterministic and non-deterministic parts of the system.",
    },
    {
        "question": "When is pure vibe coding appropriate?",
        "reference": "Pure vibe coding is appropriate for low-stakes work such as weekend prototypes, scripts, personal projects, or hackathons.",
    },
    {
        "question": "When does software development demand agentic engineering?",
        "reference": "Software that organizations depend on, especially production systems and high-stakes systems such as financial transaction APIs, demands agentic engineering.",
    },
    {
        "question": "What is context engineering?",
        "reference": "Context engineering is the practice of giving AI agents rich, structured information about the codebase, architecture, conventions, tools, constraints, and intent they need to do good work.",
    },
    {
        "question": "Why does context engineering matter more than clever prompting?",
        "reference": "The paper argues that AI-generated code quality depends less on clever prompts and more on the quality of the context provided to the model.",
    },
    {
        "question": "What are the six primary types of context developers should consider?",
        "reference": "The six types of context are instructions, knowledge, memory, examples, tools, and guardrails.",
    },
    {
        "question": "What is static context?",
        "reference": "Static context is always loaded into the model, such as system instructions, rule files, global memory, and persona definitions.",
    },
    {
        "question": "What is dynamic context?",
        "reference": "Dynamic context is loaded on demand, such as skill instructions, retrieved documents, tool results, and windowed session history.",
    },
    {
        "question": "Why can too much static context be harmful?",
        "reference": "Too much static context wastes tokens and dilutes important signals because it is included in every interaction whether relevant or not.",
    },
    {
        "question": "What are Agent Skills used for?",
        "reference": "Agent Skills package procedural knowledge that the agent loads only when needed, allowing a generalist agent to become a specialist through progressive disclosure.",
    },
    {
        "question": "What problems do Agent Skills help solve?",
        "reference": "Agent Skills help solve context rot from overloaded prompts, lack of procedural memory, multi-agent operational overhead, and portability across tools and vendors.",
    },
    {
        "question": "How does AI compress the traditional SDLC?",
        "reference": "AI compresses implementation work dramatically, turning work that once took weeks into hours, while requirements, architecture, and verification remain more human-paced.",
    },
    {
        "question": "How does AI affect requirements and planning?",
        "reference": "AI can help refine requirements by generating user stories, identifying edge cases, producing API schemas, and creating prototypes from natural-language specifications.",
    },
    {
        "question": "Why does architecture remain human-centric?",
        "reference": "Architecture remains human-centric because it requires trade-off decisions involving business context, organizational constraints, and long-term strategy that AI cannot fully grasp.",
    },
    {
        "question": "How does AI help once architectural decisions are made?",
        "reference": "Given clear architecture, AI agents can scaffold applications, generate consistent patterns across modules, and conform new code to established conventions.",
    },
    {
        "question": "How does AI transform implementation?",
        "reference": "AI agents can generate features from natural-language descriptions, implement algorithms, and make multi-file changes, shifting human work from writing to reviewing, guiding, and verifying.",
    },
    {
        "question": "What did the METR study suggest about experienced developers using AI assistants?",
        "reference": "The paper cites a METR study suggesting experienced developers took 19% longer on certain tasks, largely because of time spent verifying, debugging, and correcting AI output.",
    },
    {
        "question": "How does AI change testing and quality assurance?",
        "reference": "Testing must evaluate both the final output and the agent's trajectory, while agents can also generate tests, edge cases, and property-based tests.",
    },
    {
        "question": "What is output evaluation?",
        "reference": "Output evaluation checks the final artifact, such as whether code compiles and tests pass.",
    },
    {
        "question": "What is trajectory evaluation?",
        "reference": "Trajectory evaluation checks the sequence of tool calls and intermediate reasoning the agent used to reach the final output.",
    },
    {
        "question": "What is the continuous quality flywheel?",
        "reference": "The continuous quality flywheel evaluates against benchmarks, diagnoses failures, optimizes prompts or tools, verifies fixes against regressions, and monitors production traffic for new failure modes.",
    },
    {
        "question": "How is code review augmented by AI?",
        "reference": "AI can act as a first-pass reviewer to identify bugs, style issues, security vulnerabilities, and performance problems before human review.",
    },
    {
        "question": "Why does AI not replace human code review?",
        "reference": "AI does not replace human review because context-dependent decisions about design, maintainability, and strategic alignment still require human judgment.",
    },
    {
        "question": "How can AI help with maintenance and evolution?",
        "reference": "AI agents can navigate legacy codebases, identify relevant files, implement modifications, refactor code, update deprecated APIs, and modernize test suites.",
    },
    {
        "question": "What is the factory model of software development?",
        "reference": "The factory model says the developer's primary output is not code but the system that produces code, including specifications, agents, tests, feedback loops, and guardrails.",
    },
    {
        "question": "In the factory model, what does the developer design?",
        "reference": "The developer designs the development system or assembly line that lets agents produce code and lets tests and quality gates verify the output.",
    },
    {
        "question": "What is a harness in AI-assisted development?",
        "reference": "A harness is the scaffolding around the model, including prompts, tools, context policies, hooks, sandboxes, orchestration, sub-agents, observability, and constraints.",
    },
    {
        "question": "Why is the model alone not the whole agent?",
        "reference": "A raw model is not an agent because it needs a harness that provides state, tool execution, feedback loops, and enforceable constraints.",
    },
    {
        "question": "What components are included in a harness?",
        "reference": "A harness includes instructions and rule files, tools, sandboxes, execution environments, orchestration logic, guardrails or hooks, and observability.",
    },
    {
        "question": "What is the role of guardrails or hooks in the harness?",
        "reference": "Guardrails or hooks are deterministic code that runs at lifecycle points to enforce rules, such as blocking a commit with a hard-coded password.",
    },
    {
        "question": "Why is observability important in the harness?",
        "reference": "Observability tracks logs, traces, evaluations, cost, latency, and drift so humans can audit what the agent is doing.",
    },
    {
        "question": "How does harness configuration define the transition from vibe coding to agentic engineering?",
        "reference": "The transition is defined by how deliberately the harness is configured and applied, from minimal implicit scaffolding to clear, extensive abstractions and monitoring.",
    },
    {
        "question": "What is conductor mode?",
        "reference": "Conductor mode is a hands-on workflow where the developer works in real time with an AI pair programmer, guiding prompts and corrections while maintaining fine-grained control.",
    },
    {
        "question": "What is orchestrator mode?",
        "reference": "Orchestrator mode is a higher-level workflow where the developer defines goals, delegates work to agents, reviews results, and provides course corrections asynchronously.",
    },
    {
        "question": "What skills does orchestrator mode require?",
        "reference": "Orchestrator mode requires specification, decomposition, evaluation, and system design skills.",
    },
    {
        "question": "What is the 80% problem in AI-assisted development?",
        "reference": "The 80% problem is that AI agents can quickly generate about 80% of a feature, but the remaining edge cases, error handling, integrations, and subtle correctness requirements need deep contextual knowledge.",
    },
    {
        "question": "What are the three places coding agents show up in everyday developer work?",
        "reference": "Coding agents show up in the editor, in the terminal, and in the background.",
    },
]

SAMPLE_QUESTIONS = [item["question"] for item in QA_PAIRS]