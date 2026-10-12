"""
AGI-5L Benchmark Suite: 50 Rigorous Questions Across the 5 Levels of AGI
========================================================================
Levels Tested:
- Level 1: Chatbots (Pengetahuan & Bahasa) - 10 Questions
- Level 2: Reasoners (Logika & Pemecahan Masalah) - 10 Questions
- Level 3: Agents (Aksi Otonom & Adaptasi) - 10 Questions
- Level 4: Innovators (Penemuan Sains Mandiri) - 10 Questions
- Level 5: Organizations (Superintelligence / ASI) - 10 Questions

Total: 50 Questions (10 per Level)
"""

def get_agi_5level_suite():
    suite = [
        # =====================================================================
        # LEVEL 1: CHATBOTS (Pengetahuan & Bahasa)
        # =====================================================================
        {
            "id": "L1_01",
            "level": "Level 1: Chatbots",
            "domain": "Linguistics & Semantic Precision",
            "question": "Which of the following sentences correctly resolves semantic ambiguity without syntactic dangling modifiers?\nA. Barking loudly at midnight, the neighbours were disturbed by the dog.\nB. The dog disturbed the neighbours by barking loudly at midnight.\nC. Having finished the task, the computer was shut down by Matthew.\nD. To succeed in science, hard work is required by everyone.",
            "correct": "B"
        },
        {
            "id": "L1_02",
            "level": "Level 1: Chatbots",
            "domain": "World Epistemic Knowledge",
            "question": "What is the primary thermodynamic driving force behind the hydrophobic effect in protein folding?\nA. Enthalpic gain from peptide backbone hydrogen bonding\nB. Net decrease in protein vibrational entropy\nC. Entropic release of structured clathrate-like water molecules\nD. Covalent disulfide bond formation between nonpolar residues",
            "correct": "C"
        },
        {
            "id": "L1_03",
            "level": "Level 1: Chatbots",
            "domain": "Cross-lingual Idiomatic Nuance (Indonesian)",
            "question": "Dalam konteks semantik bahasa Indonesia, peribahasa manakah yang bermakna 'menginginkan sesuatu yang mustahil tercapai'?\nA. Bagai pungguk merindukan bulan\nB. Air beriak tanda tak dalam\nC. Tong kosong nyaring bunyinya\nD. Menepuk air di dulang terpercik muka sendiri",
            "correct": "A"
        },
        {
            "id": "L1_04",
            "level": "Level 1: Chatbots",
            "domain": "Pragmatic Inference & Negation",
            "question": "Statement: 'Not all employees who passed the security audit were granted administrative access.'\nWhich of the following MUST be logically true?\nA. No employee who failed the audit was granted administrative access.\nB. All employees who were granted administrative access passed the audit.\nC. At least one employee who passed the security audit was not granted administrative access.\nD. A majority of employees failed the security audit.",
            "correct": "C"
        },
        {
            "id": "L1_05",
            "level": "Level 1: Chatbots",
            "domain": "Computer Science Knowledge",
            "question": "What is the worst-case time complexity of searching for an element in an AVL tree containing N nodes?\nA. O(1)\nB. O(log N)\nC. O(N)\nD. O(N log N)",
            "correct": "B"
        },
        {
            "id": "L1_06",
            "level": "Level 1: Chatbots",
            "domain": "Philosophical & Cognitive Epistemology",
            "question": "In the Gettier problem, what classical definition of knowledge is shown to be insufficient?\nA. Coherentist belief network\nB. Justified True Belief (JTB)\nC. Pragmatic instrumentalism\nD. Radical empiricism",
            "correct": "B"
        },
        {
            "id": "L1_07",
            "level": "Level 1: Chatbots",
            "domain": "Syntactic Inversion & Grammar",
            "question": "Which of the following demonstrates correct negative inversion in English grammar?\nA. Seldom he has witnessed such extraordinary computational power.\nB. Seldom has he witnessed such extraordinary computational power.\nC. Seldom does he witnessed such extraordinary computational power.\nD. Seldom he had witnessed such extraordinary computational power.",
            "correct": "B"
        },
        {
            "id": "L1_08",
            "level": "Level 1: Chatbots",
            "domain": "Information Theory",
            "question": "For a discrete memoryless source emitting 4 equiprobable symbols, what is the Shannon entropy in bits per symbol?\nA. 1 bit\nB. 2 bits\nC. 4 bits\nD. 8 bits",
            "correct": "B"
        },
        {
            "id": "L1_09",
            "level": "Level 1: Chatbots",
            "domain": "Technical Etymology & Terminology",
            "question": "The term 'heuristic' in artificial intelligence originates from the Greek 'heurisko', meaning:\nA. To calculate\nB. To find or discover\nC. To prove by contradiction\nD. To iterate indefinitely",
            "correct": "B"
        },
        {
            "id": "L1_10",
            "level": "Level 1: Chatbots",
            "domain": "Linguistic Ambiguity & Winograd",
            "question": "In the sentence: 'The trophy would not fit in the brown suitcase because it was too large.' What does 'it' refer to?\nA. The trophy\nB. The suitcase\nC. The brown color\nD. The room",
            "correct": "A"
        },

        # =====================================================================
        # LEVEL 2: REASONERS (Logika & Pemecahan Masalah)
        # =====================================================================
        {
            "id": "L2_01",
            "level": "Level 2: Reasoners",
            "domain": "Formal Deductive Logic",
            "question": "Premises:\n1. All topological manifolds are Hausdorff spaces.\n2. No Hausdorff space has non-unique limits of convergent sequences.\n3. Space X has a non-unique limit for some convergent sequence.\nConclusion:\nA. Space X is a compact topological manifold.\nB. Space X is not a topological manifold.\nC. Space X is a metric space.\nD. Space X has finite dimension.",
            "correct": "B"
        },
        {
            "id": "L2_02",
            "level": "Level 2: Reasoners",
            "domain": "Discrete Mathematics & Number Theory",
            "question": "Find the remainder when 3^100 is divided by 7.\nA. 1\nB. 2\nC. 4\nD. 6",
            "correct": "C"
        },
        {
            "id": "L2_03",
            "level": "Level 2: Reasoners",
            "domain": "ARC Spatial Invariant (90-Degree Rotation)",
            "question": "Given the 2x2 binary grid:\n[[1, 0],\n [1, 1]]\nWhat is the resulting grid after a 90-degree CLOCKWISE rotation?\nA. [[1, 1], [1, 0]]\nB. [[1, 1], [0, 1]]\nC. [[0, 1], [1, 1]]\nD. [[1, 0], [1, 1]]",
            "correct": "A"
        },
        {
            "id": "L2_04",
            "level": "Level 2: Reasoners",
            "domain": "Graph Theory & Combinatorics",
            "question": "How many edges are present in a complete bipartite graph K(4, 5)?\nA. 9\nB. 20\nC. 25\nD. 40",
            "correct": "B"
        },
        {
            "id": "L2_05",
            "level": "Level 2: Reasoners",
            "domain": "Counterfactual & Causal Reasoning",
            "question": "If switch A and switch B are in series controlling lamp L, and L is currently OFF while switch A is ON, what causal intervention guarantees L turns ON?\nA. Turning switch A OFF\nB. Turning switch B ON\nC. Toggling switch A twice\nD. Removing switch B from circuit",
            "correct": "B"
        },
        {
            "id": "L2_06",
            "level": "Level 2: Reasoners",
            "domain": "Probability & Bayesian Deduction",
            "question": "A rare disease has an incidence of 1 in 1000. A test has a 99% true positive rate and a 5% false positive rate. If a randomly chosen person tests positive, what is the approximate posterior probability that they actually have the disease?\nA. ~99%\nB. ~50%\nC. ~2%\nD. ~0.1%",
            "correct": "C"
        },
        {
            "id": "L2_07",
            "level": "Level 2: Reasoners",
            "domain": "ARC Spatial Invariant (Bounding Box Crop)",
            "question": "Given a 4x4 grid where non-zero cells are at row 1, col 2 (val 7) and row 2, col 3 (val 7), with all other cells 0. What are the dimensions of the minimal bounding box enclosing all non-zero cells?\nA. 1x1\nB. 2x2\nC. 2x3\nD. 3x3",
            "correct": "B"
        },
        {
            "id": "L2_08",
            "level": "Level 2: Reasoners",
            "domain": "Multi-Step Mathematical Logic",
            "question": "Solve for real x: log_2(x) + log_2(x - 2) = 3.\nA. x = 2\nB. x = 4\nC. x = -2\nD. x = 8",
            "correct": "B"
        },
        {
            "id": "L2_09",
            "level": "Level 2: Reasoners",
            "domain": "Algorithmic Invariant Tracing",
            "question": "In Dijkstra's algorithm with non-negative edge weights, what property is invariant for each vertex u upon extraction from the priority queue?\nA. u has the maximum degree in the graph\nB. The recorded distance d[u] is the exact shortest path distance from source\nC. All outgoing edges from u have been permanently removed\nD. The graph is verified to be acyclic",
            "correct": "B"
        },
        {
            "id": "L2_10",
            "level": "Level 2: Reasoners",
            "domain": "Constraint Satisfaction (Zebra Logic)",
            "question": "Five houses are in a row. The Green house is directly to the right of the Ivory house. If the Ivory house is at position 3, what position is the Green house?\nA. Position 2\nB. Position 4\nC. Position 1\nD. Position 5",
            "correct": "B"
        },

        # =====================================================================
        # LEVEL 3: AGENTS (Aksi Otonom & Adaptasi)
        # =====================================================================
        {
            "id": "L3_01",
            "level": "Level 3: Agents",
            "domain": "Autonomous Tool Call Schema",
            "question": "An autonomous agent needs to inspect a SQLite database table 'users' without loading full records into RAM. Which SQL query action should the agent dispatch?\nA. SELECT * FROM users;\nB. SELECT id, email FROM users LIMIT 5;\nC. DROP TABLE users;\nD. ALTER TABLE users ADD COLUMN temp INT;",
            "correct": "B"
        },
        {
            "id": "L3_02",
            "level": "Level 3: Agents",
            "domain": "Error Self-Correction Loop",
            "question": "An autonomous agent receives a HTTP 429 Too Many Requests response during an API scraping workflow. What is the optimal adaptive action?\nA. Immediately re-send the request 100 times in parallel\nB. Apply exponential backoff with jitter and retry after the 'Retry-After' header duration\nC. Abort the entire program and delete the database\nD. Switch HTTP method to DELETE and retry",
            "correct": "B"
        },
        {
            "id": "L3_03",
            "level": "Level 3: Agents",
            "domain": "State Machine Adaptation",
            "question": "In a goal-directed agent with states {IDLE, PLANNING, EXECUTING, RECOVERY}, if an execution step fails with an unexpected exception, to which state should the agent transition to preserve safety and adapt?\nA. IDLE\nB. RECOVERY\nC. TERMINATED\nD. EXECUTING",
            "correct": "B"
        },
        {
            "id": "L3_04",
            "level": "Level 3: Agents",
            "domain": "Goal Decomposition",
            "question": "To fulfill the high-level user goal 'Deploy a trained PyTorch model as a high-throughput microservice', which sequence of sub-goals represents the correct autonomous execution order?\nA. Load testing -> Write code -> Benchmark -> Dockerize\nB. Serialize model weights -> Build FastAPI endpoint -> Containerize (Docker) -> Benchmarking & Deployment\nC. Deploy -> Containerize -> Code -> Train\nD. Dockerize -> Train -> Deploy -> Write tests",
            "correct": "B"
        },
        {
            "id": "L3_05",
            "level": "Level 3: Agents",
            "domain": "Continuous Plasticity & Episodic Memory (VGNI)",
            "question": "When an agent encounters a novel domain distribution not present in static pretraining weights, how does an endogenous plasticity memory mechanism (such as VGNI Layer 11 associative slots) preserve past capabilities while adapting?\nA. Wiping all previous model weights to zero\nB. Orthogonal subspace gradient projection to prevent catastrophic forgetting of previous anchors\nC. Increasing learning rate to 10.0 indiscriminately\nD. Disabling backpropagation entirely",
            "correct": "B"
        },
        {
            "id": "L3_06",
            "level": "Level 3: Agents",
            "domain": "Sandboxed Execution Safety",
            "question": "Before executing user-submitted untrusted code, an autonomous agent must enforce sandbox boundaries. Which technique provides the strongest security isolation?\nA. Running the code inside `eval()` in the main process\nB. Running inside a restricted seccomp-bpf Linux container with drop-capabilities and no network egress\nC. Prepending `try ... except:` to the code\nD. Increasing Python recursion limit",
            "correct": "B"
        },
        {
            "id": "L3_07",
            "level": "Level 3: Agents",
            "domain": "Multi-Step Rollback & Transactional Safety",
            "question": "If step 3 of a 5-step file refactoring pipeline throws an unhandled FileNotFoundError, what should the transactional agent do with the temporary files created in steps 1 and 2?\nA. Leave them permanently scattered on disk\nB. Execute compensation rollback logic to delete temporary files and restore initial state\nC. Rename them to .backup and exit immediately\nD. Re-format the entire hard drive",
            "correct": "B"
        },
        {
            "id": "L3_08",
            "level": "Level 3: Agents",
            "domain": "Environment Exploration vs Exploitation",
            "question": "In an autonomous reinforcement learning agent exploring a stochastic gridworld with unknown reward distributions, how does Upper Confidence Bound (UCB) balance exploration?\nA. It chooses actions uniformly at random\nB. It prioritizes actions with high estimated return plus an uncertainty bonus proportional to square-root of log(t)/N(a)\nC. It always exploits the single action with smallest count\nD. It terminates immediately after step 1",
            "correct": "B"
        },
        {
            "id": "L3_09",
            "level": "Level 3: Agents",
            "domain": "Reflective Prompting & CoT Self-Critique",
            "question": "An agent generates a draft code solution, executes unit tests, and detects an Off-By-One error. What is the effective reflective prompt for self-correction?\nA. 'I am perfect and the test is wrong.'\nB. 'Inspect the loop boundary condition (e.g., range(len(arr)) vs range(len(arr)-1)) and adjust index bounds to pass the test.'\nC. 'Delete the failing test assertion.'\nD. 'Output empty string.'",
            "correct": "B"
        },
        {
            "id": "L3_10",
            "level": "Level 3: Agents",
            "domain": "Dynamic Resource Throttling (Hardware Safety)",
            "question": "When an autonomous agent running long-horizon GPU workloads detects thermal saturation (temperature > 85°C) or potential driver watchdog timeouts, what is the appropriate autonomous mitigation?\nA. Overclock the GPU core clock by +500 MHz\nB. Throttle throughput by inserting cooldown pauses (`time.sleep`) and flushing VRAM caches\nC. Disable all system fans to save energy\nD. Launch 10 additional threads simultaneously",
            "correct": "B"
        },

        # =====================================================================
        # LEVEL 4: INNOVATORS (Penemuan Sains Mandiri)
        # =====================================================================
        {
            "id": "L4_01",
            "level": "Level 4: Innovators",
            "domain": "Scientific Hypothesis Abduction",
            "question": "Astronomers observe an anomalous 43 arcseconds per century precession in Mercury's perihelion that Newtonian gravitation cannot explain. What abductive hypothesis resolves this anomaly without inventing unseen planets?\nA. Gravitational fields curve spacetime according to Einstein's General Relativity tensor equations\nB. Solar wind exerts asymmetric radiative pressure\nC. The speed of light is infinite in solar vicinity\nD. Coulomb electrostatic repulsion between Mercury and Sun",
            "correct": "A"
        },
        {
            "id": "L4_02",
            "level": "Level 4: Innovators",
            "domain": "Biochemical Reaction Discovery",
            "question": "In de novo molecular drug design, which structural modification to a lead compound typically enhances metabolic stability against cytochrome P450 oxidation at an aromatic C-H site?\nA. Replacing the C-H bond with a bioisosteric Fluorine atom (C-F)\nB. Adding a highly reactive aldehyde group\nC. Removing all aromatic rings entirely\nD. Increasing lipophilicity to maximum possible value",
            "correct": "A"
        },
        {
            "id": "L4_03",
            "level": "Level 4: Innovators",
            "domain": "Falsification & Experimental Design",
            "question": "According to Karl Popper's criterion of demarcation, what distinguishes a scientific theory from pseudoscience?\nA. The ability to verify the theory with infinite empirical confirmations\nB. The presence of testable predictions that are capable of being empirically falsified\nC. The authority and prestige of the proposing scientist\nD. Total mathematical complexity of the equations",
            "correct": "B"
        },
        {
            "id": "L4_04",
            "level": "Level 4: Innovators",
            "domain": "Quantum Mechanics & Superconductivity",
            "question": "What is the microscopic mechanism in BCS theory that allows electrons to form Cooper pairs and condense into a zero-resistance superconducting state?\nA. Coulomb electrostatic attraction between naked electrons\nB. Phonon-mediated effective attractive interaction between electron pairs of opposite spin and momentum\nC. Strong nuclear force exchange\nD. Magnetic monopole condensation",
            "correct": "B"
        },
        {
            "id": "L4_05",
            "level": "Level 4: Innovators",
            "domain": "Novel Mathematical Conjecture",
            "question": "The Langlands Program seeks to establish deep, revolutionary conjectures connecting which two major mathematical fields?\nA. Euclidean Geometry and Elementary Arithmetic\nB. Number Theory (Galois representations) and Harmonic Analysis (Automorphic forms)\nC. Topology and Ordinary Differential Equations\nD. Set Theory and Classical Mechanics",
            "correct": "B"
        },
        {
            "id": "L4_06",
            "level": "Level 4: Innovators",
            "domain": "Thermodynamics & Non-Equilibrium Systems",
            "question": "In non-equilibrium thermodynamics, Ilya Prigogine showed that open systems far from thermal equilibrium can spontaneously form ordered 'dissipative structures' by:\nA. Violating the second law of thermodynamics locally\nB. Continually dissipating entropy to the external environment\nC. Halting all internal microscopic motion\nD. Becoming perfectly isolated systems",
            "correct": "B"
        },
        {
            "id": "L4_07",
            "level": "Level 4: Innovators",
            "domain": "Genetic Engineering & Synthetic Biology",
            "question": "In synthetic biology, how does a CRISPR-Cas9 ribonucleoprotein complex achieve sequence-specific double-strand DNA cleavage?\nA. Random blunt-force digestion\nB. Base-pairing between a single guide RNA (sgRNA) spacer and target protospacer adjacent to a PAM sequence\nC. Binding exclusively to methylated CpG islands\nD. Synthesizing de novo nucleotide monomers",
            "correct": "B"
        },
        {
            "id": "L4_08",
            "level": "Level 4: Innovators",
            "domain": "Astrophysics & Dark Matter Detection",
            "question": "What observational evidence conclusively demonstrated that dark matter exists independently of baryonic gas distributions?\nA. Keplerian planetary orbits in the Solar System\nB. The Bullet Cluster (1E 0657-558) gravitational lensing offset from X-ray gas peaks\nC. Solar neutrino oscillations\nD. Pulsar timing residuals from Jupiter",
            "correct": "B"
        },
        {
            "id": "L4_09",
            "level": "Level 4: Innovators",
            "domain": "Materials Science & Nanotechnology",
            "question": "Why does monolayer graphene exhibit exceptionally high electrical conductivity at room temperature?\nA. It is a heavy-atom insulator\nB. Its charge carriers behave as massless Dirac fermions with linear dispersion at the Brillouin zone Dirac points\nC. It possesses infinite Cooper pairs at room temperature\nD. It contains a high density of free radioactive protons",
            "correct": "B"
        },
        {
            "id": "L4_10",
            "level": "Level 4: Innovators",
            "domain": "Cognitive Science & Artificial General Intelligence",
            "question": "In formulating theories of fluid intelligence for AGI, what core distinction separates François Chollet's ARC-AGI benchmark from standard large-language-model pretraining benchmarks?\nA. ARC-AGI measures few-shot program acquisition and novel prior synthesis, rather than memorized task interpolation\nB. ARC-AGI has fewer than 10 tokens total\nC. ARC-AGI tests only vocabulary spelling in English\nD. ARC-AGI requires 100 terabytes of memory",
            "correct": "A"
        },

        # =====================================================================
        # LEVEL 5: ORGANIZATIONS (Superintelligence / ASI)
        # =====================================================================
        {
            "id": "L5_01",
            "level": "Level 5: Organizations",
            "domain": "Game-Theoretic Nash Equilibrium",
            "question": "In a symmetric 2-player Prisoner's Dilemma with payoff matrix where (Cooperate, Cooperate) = (3, 3), (Defect, Defect) = (1, 1), and (Defect, Cooperate) = (5, 0), what is the unique Nash Equilibrium in pure strategies?\nA. (Cooperate, Cooperate)\nB. (Defect, Defect)\nC. (Defect, Cooperate)\nD. (Cooperate, Defect)",
            "correct": "B"
        },
        {
            "id": "L5_02",
            "level": "Level 5: Organizations",
            "domain": "Decentralized Multi-Agent Consensus",
            "question": "In a distributed network of N autonomous agents with Byzantine faults (arbitrary malicious failures), what is the maximum number of faulty nodes f that a consensus protocol can tolerate in a synchronous system without signatures?\nA. f < N/2\nB. f < N/3\nC. f < N/4\nD. f < 1",
            "correct": "B"
        },
        {
            "id": "L5_03",
            "level": "Level 5: Organizations",
            "domain": "Macroeconomic Equilibrium & Pareto Efficiency",
            "question": "According to the First Fundamental Theorem of Welfare Economics, under what conditions does a competitive market equilibrium achieve Pareto efficiency?\nA. Only under strict central price control\nB. In complete markets with perfect competition, price-taking behavior, and no externalities or transaction costs\nC. Only when government owns all capital goods\nD. When inflation rate is exactly zero",
            "correct": "B"
        },
        {
            "id": "L5_04",
            "level": "Level 5: Organizations",
            "domain": "Superintelligence Alignment & Orthogonality",
            "question": "According to Nick Bostrom's Orthogonality Thesis, what is the relationship between an artificial intelligence's level of intelligence and its final goals?\nA. Higher intelligence automatically guarantees benevolent moral goals\nB. More intelligent systems always maximize survival instincts\nC. More or less any level of intelligence could in principle be combined with more or less any final goal\nD. Superintelligence cannot have any goals whatsoever",
            "correct": "C"
        },
        {
            "id": "L5_05",
            "level": "Level 5: Organizations",
            "domain": "Instrumental Convergence",
            "question": "What is an example of an 'instrumentally convergent' sub-goal that almost any advanced autonomous agent would pursue across diverse terminal goals?\nA. Self-destruction within 5 minutes\nB. Self-preservation and resource acquisition\nC. Playing chess at 200 Elo\nD. Deleting its own memory banks",
            "correct": "B"
        },
        {
            "id": "L5_06",
            "level": "Level 5: Organizations",
            "domain": "Multi-Agent Mechanism Design",
            "question": "What key property does the Vickrey-Clarke-Groves (VCG) auction mechanism guarantee for participating strategic agents?\nA. First-price pay-as-bid winner\nB. Strategy-proofness (truthful revelation of private valuations is a dominant strategy)\nC. Equal profit division among all bidders\nD. Maximum possible revenue for seller in all circumstances",
            "correct": "B"
        },
        {
            "id": "L5_07",
            "level": "Level 5: Organizations",
            "domain": "Organizational Cybernetics & Viable Systems",
            "question": "In Stafford Beer's Viable System Model (VSM) for complex adaptive organizations, what is the primary role of 'System 4' (Intelligence / Adaptation)?\nA. Daily routine operations\nB. Looking outward to the external environment and forward into the future to guide strategic adaptation\nC. Auditing financial receipts\nD. Physical security enforcement",
            "correct": "B"
        },
        {
            "id": "L5_08",
            "level": "Level 5: Organizations",
            "domain": "Dynamic Resource Allocation under Uncertainty",
            "question": "When coordinating a swarm of 100 autonomous worker agents under dynamic workload arrival rates, which dispatch policy minimizes mean job completion time when service times follow an unknown distribution?\nA. Random round-robin without queue inspection\nB. Join the Shortest Queue (JSQ) / Power of Two Choices routing\nC. Always routing to the single slowest agent\nD. Discarding 50% of incoming jobs randomly",
            "correct": "B"
        },
        {
            "id": "L5_09",
            "level": "Level 5: Organizations",
            "domain": "Global Catastrophic Risk Governance",
            "question": "In modeling systemic contagion across global financial, energy, and digital agent networks, what structural network topology presents the highest risk of cascading systemic failure?\nA. Completely disconnected isolated nodes\nB. Highly centralized hub-and-spoke networks with tightly coupled non-diversified interbank/inter-agent exposures\nC. Uniform ring lattice with local nearest-neighbor connections only\nD. Zero-degree null graphs",
            "correct": "B"
        },
        {
            "id": "L5_10",
            "level": "Level 5: Organizations",
            "domain": "Collective Superintelligence Coordination",
            "question": "In a decentralized superintelligent multi-agent architecture (Level 5 ASI), what mechanism prevents principal-agent divergence between coordinating neural nodes?\nA. Periodic random re-initialization of weights\nB. Formal cryptographic verifiability (zero-knowledge proofs of execution) coupled with aligned multi-party objective consensus\nC. Running all agents on a single CPU core\nD. Suppressing inter-node communication entirely",
            "correct": "B"
        }
    ]
    return suite
