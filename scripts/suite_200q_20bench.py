"""
Test Suite Generator for 200 Questions Across 20 Canonical Benchmarks
(10 Distinct Questions x 20 Benchmarks = 200 Questions Total)
"""
import re
from typing import Dict, List, Any

def build_200q_suite() -> List[Dict[str, Any]]:
    suite = []

    # =========================================================================
    # PILLAR 1: MATH & QUANTITATIVE (40 Questions)
    # =========================================================================

    # 1. GSM8K (10 Questions)
    gsm_data = [
        ("Janet", "apples", 14, 2),
        ("Tom", "books", 8, 10),
        ("Lisa", "cookies", 15, 3),
        ("Bob", "notebooks", 6, 4),
        ("Alice", "pencils", 12, 1),
        ("David", "erasers", 20, 2),
        ("Emma", "sandwiches", 5, 6),
        ("John", "muffins", 9, 3),
        ("Sarah", "pens", 11, 2),
        ("Michael", "water bottles", 7, 5)
    ]
    for i, (name, item, count, price) in enumerate(gsm_data, 1):
        tot = count * price
        prompt = f"{name} buys {count} {item} at ${price} each. How much money did {name} spend in total?\nAnswer:"
        suite.append({
            "benchmark": "GSM8K", "pillar": "Math & Quantitative", "id": f"gsm8k_{i:02d}",
            "prompt": prompt, "target": str(tot), "max_tokens": 160,
            "check": lambda t, tv=str(tot): bool(re.search(r"\b" + re.escape(tv) + r"\b", t))
        })

    # 2. MATH (10 Questions)
    math_data = [
        (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
        (7, 8), (2, 9), (3, 7), (5, 3), (4, 8)
    ]
    for i, (a, b) in enumerate(math_data, 1):
        res = a * 2 + b
        prompt = f"If f(x) = {a}x + {b}, what is the value of f(2)? Answer with only the number.\nAnswer:"
        suite.append({
            "benchmark": "MATH", "pillar": "Math & Quantitative", "id": f"math_{i:02d}",
            "prompt": prompt, "target": str(res), "max_tokens": 160,
            "check": lambda t, tv=str(res): bool(re.search(r"\b" + re.escape(tv) + r"\b", t))
        })

    # 3. DROP (10 Questions)
    drop_data = [
        ("Eagles", "Cowboys", 17, 10, 14),
        ("Lions", "Packers", 21, 14, 7),
        ("Patriots", "Dolphins", 10, 7, 13),
        ("Bulls", "Lakers", 24, 20, 18),
        ("Giants", "Dodgers", 12, 6, 9),
        ("Celtics", "Warriors", 28, 25, 15),
        ("Chiefs", "Raiders", 14, 10, 17),
        ("Ravens", "Steelers", 16, 9, 13),
        ("Seahawks", "Cardinals", 20, 13, 8),
        ("Broncos", "Chargers", 11, 17, 19)
    ]
    for i, (t1, t2, p1, p2, p3) in enumerate(drop_data, 1):
        tot = p1 + p3
        prompt = f"Passage: In the first half, the {t1} scored {p1} points while {t2} scored {p2} points. In the second half, the {t1} scored {p3} points.\nQuestion: How many total points did the {t1} score? Answer with only the number.\nAnswer:"
        suite.append({
            "benchmark": "DROP", "pillar": "Math & Quantitative", "id": f"drop_{i:02d}",
            "prompt": prompt, "target": str(tot), "max_tokens": 160,
            "check": lambda t, tv=str(tot): bool(re.search(r"\b" + re.escape(tv) + r"\b", t))
        })

    # 4. BBH (10 Questions) - Multi-step Reasoning
    bbh_data = [
        ("You are currently facing North. You turn 180 degrees. What direction are you facing now?", "South"),
        ("You are currently facing South. You turn 180 degrees. What direction are you facing now?", "North"),
        ("You are currently facing East. You turn 180 degrees. What direction are you facing now?", "West"),
        ("You are currently facing West. You turn 180 degrees. What direction are you facing now?", "East"),
        ("You are currently facing North. You turn 90 degrees clockwise. What direction are you facing now?", "East"),
        ("A coin is heads up. You flip it once. Is it heads or tails?", "Tails"),
        ("A coin is tails up. You flip it twice. Is it heads or tails?", "Tails"),
        ("Boolean expression: True and False. Is the result True or False?", "False"),
        ("Boolean expression: not (False or False). Is the result True or False?", "True"),
        ("If yesterday was Wednesday, what day is tomorrow?", "Friday")
    ]
    for i, (q, ans) in enumerate(bbh_data, 1):
        prompt = f"{q}\nAnswer:"
        suite.append({
            "benchmark": "BBH", "pillar": "Math & Quantitative", "id": f"bbh_{i:02d}",
            "prompt": prompt, "target": ans, "max_tokens": 160,
            "check": lambda t, a=ans: bool(re.search(r"\b" + re.escape(a) + r"\b", t, re.I))
        })

    # =========================================================================
    # PILLAR 2: SCIENCE & ACADEMIC (50 Questions)
    # =========================================================================

    # 5. MMLU (10 Questions)
    mmlu_bank = [
        ("What organelle is responsible for aerobic cellular respiration in eukaryotes?", "Mitochondria", ["Nucleus", "Ribosome", "Mitochondria", "Lysosome"], "C"),
        ("What is the chemical symbol for table salt component sodium?", "Na", ["K", "Na", "Ca", "Cl"], "B"),
        ("What is the speed of light in vacuum approximately in m/s?", "3x10^8", ["3x10^8", "3x10^6", "1.5x10^8", "9.8x10^2"], "A"),
        ("Which planet has the greatest mass in our Solar System?", "Jupiter", ["Mars", "Saturn", "Jupiter", "Neptune"], "C"),
        ("What gas makes up approximately 78% of Earth's atmosphere?", "Nitrogen", ["Oxygen", "Nitrogen", "Argon", "CO2"], "B"),
        ("What is the basic unit of heredity in living organisms?", "Gene", ["Cell", "Gene", "Protein", "Lipid"], "B"),
        ("What subatomic particle carries a negative electric charge?", "Electron", ["Proton", "Neutron", "Electron", "Positron"], "C"),
        ("What is the powerhouse unit of electrical potential difference?", "Volt", ["Ampere", "Watt", "Ohm", "Volt"], "D"),
        ("Which blood type is known as the universal red blood cell donor?", "O negative", ["AB positive", "A positive", "O negative", "B negative"], "C"),
        ("What is the primary product of plant photosynthesis?", "Glucose", ["Methane", "Glucose", "Ammonia", "Ethanol"], "B")
    ]
    for i, (q, ans, opts, letter) in enumerate(mmlu_bank, 1):
        prompt = f"Question: {q}\nA) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}\nAnswer with the letter:\nAnswer:"
        suite.append({
            "benchmark": "MMLU", "pillar": "Science & Academic", "id": f"mmlu_{i:02d}",
            "prompt": prompt, "target": letter, "max_tokens": 96,
            "check": lambda t, lt=letter, an=ans: bool(re.search(r"\b" + lt + r"\b", t, re.I)) or an.lower() in t.lower()
        })

    # 6. AGIEval (10 Questions) - Syllogisms & Logic
    agieval_bank = [
        ("All mammals are vertebrates. All whales are mammals. Are all whales vertebrates?", "Yes"),
        ("No reptiles have feathers. All snakes are reptiles. Do any snakes have feathers?", "No"),
        ("All squares are rectangles. Shape S is a square. Is shape S a rectangle?", "Yes"),
        ("All birds lay eggs. A penguin is a bird. Do penguins lay eggs?", "Yes"),
        ("No fish breathe air with lungs. A goldfish is a fish. Does a goldfish breathe air with lungs?", "No"),
        ("All triangles have three vertices. Shape T has four vertices. Is shape T a triangle?", "No"),
        ("All carnivores eat meat. A lion is a carnivore. Does a lion eat meat?", "Yes"),
        ("No insects have bones. An ant is an insect. Does an ant have bones?", "No"),
        ("All planets orbit a star. Earth is a planet. Does Earth orbit a star?", "Yes"),
        ("All prime numbers greater than 2 are odd. Number 4 is even. Is number 4 a prime number?", "No")
    ]
    for i, (q_text, ans) in enumerate(agieval_bank, 1):
        prompt = f"Logical deduction: {q_text} Answer with Yes or No.\nAnswer:"
        suite.append({
            "benchmark": "AGIEval", "pillar": "Science & Academic", "id": f"agieval_{i:02d}",
            "prompt": prompt, "target": ans, "max_tokens": 96,
            "check": lambda t, a=ans: bool(re.search(r"\b" + a + r"\b", t, re.I))
        })

    # 7. TriviaQA (10 Questions) - Factual Knowledge
    trivia_bank = [
        ("In what year did Apollo 11 land on the Moon?", "1969", lambda t: "1969" in t),
        ("What is the capital city of France?", "Paris", lambda t: bool(re.search(r"\bparis\b", t, re.I))),
        ("What is the largest ocean on Earth?", "Pacific", lambda t: bool(re.search(r"\bpacific\b", t, re.I))),
        ("What chemical element does 'O' represent on the periodic table?", "Oxygen", lambda t: bool(re.search(r"\boxygen\b", t, re.I))),
        ("Who wrote the play 'Romeo and Juliet'?", "Shakespeare", lambda t: bool(re.search(r"\bshakespeare\b", t, re.I))),
        ("What is the capital city of Japan?", "Tokyo", lambda t: bool(re.search(r"\btokyo\b", t, re.I))),
        ("What is the tallest mountain in the world above sea level?", "Everest", lambda t: bool(re.search(r"\beverest\b", t, re.I))),
        ("What year did World War II end?", "1945", lambda t: "1945" in t),
        ("What is the chemical symbol for gold?", "Au", lambda t: bool(re.search(r"\bau\b", t, re.I)) or "gold" in t.lower()),
        ("Who developed the general theory of relativity?", "Einstein", lambda t: bool(re.search(r"\beinstein\b", t, re.I)))
    ]
    for i, (q_text, ans, chk) in enumerate(trivia_bank, 1):
        prompt = f"Question: {q_text}\nAnswer:"
        suite.append({
            "benchmark": "TriviaQA", "pillar": "Science & Academic", "id": f"trivia_{i:02d}",
            "prompt": prompt, "target": ans, "max_tokens": 128, "check": chk
        })

    # 8. SQuAD_v2 (10 Questions) - Reading Comprehension
    squad_bank = [
        ("The Amazon River is the largest river by discharge volume in the world.", "Which river is largest by discharge volume?", "Amazon", lambda t: bool(re.search(r"\bamazon\b", t, re.I))),
        ("Penicillin was discovered in 1928 by Scottish scientist Alexander Fleming.", "Who discovered penicillin?", "Fleming", lambda t: bool(re.search(r"\bfleming\b", t, re.I))),
        ("The Eiffel Tower in Paris was completed in 1889.", "In what year was the Eiffel Tower completed?", "1889", lambda t: "1889" in t),
        ("Photosynthesis stores energy in glucose molecules.", "What molecule stores chemical energy?", "Glucose", lambda t: bool(re.search(r"\bglucose\b", t, re.I))),
        ("Mars has two small natural moons named Phobos and Deimos.", "What are the names of Mars' moons?", "Phobos and Deimos", lambda t: bool(re.search(r"\bphobos\b", t, re.I))),
        ("The Great Barrier Reef is situated off the coast of Queensland in Australia.", "Where is the Great Barrier Reef located?", "Australia", lambda t: bool(re.search(r"\baustralia\b", t, re.I))),
        ("DNA is composed of four nucleotide bases: adenine, thymine, guanine, and cytosine.", "How many nucleotide bases compose DNA?", "four", lambda t: bool(re.search(r"\b(four|4)\b", t, re.I))),
        ("The printing press was invented in Mainz, Germany by Johannes Gutenberg around 1440.", "Who invented the printing press?", "Gutenberg", lambda t: bool(re.search(r"\bgutenberg\b", t, re.I))),
        ("Saturn is renowned for its prominent planetary ring system composed primarily of ice particles.", "What is Saturn's ring system primarily composed of?", "ice", lambda t: bool(re.search(r"\bice\b", t, re.I))),
        ("The Mona Lisa was painted by the Renaissance master Leonardo da Vinci.", "Who painted the Mona Lisa?", "Leonardo da Vinci", lambda t: bool(re.search(r"\b(da vinci|leonardo)\b", t, re.I)))
    ]
    for i, (passage, q_text, ans, chk) in enumerate(squad_bank, 1):
        prompt = f"Passage: {passage}\nQuestion: {q_text}\nAnswer:"
        suite.append({
            "benchmark": "SQuAD_v2", "pillar": "Science & Academic", "id": f"squad_{i:02d}",
            "prompt": prompt, "target": ans, "max_tokens": 128, "check": chk
        })

    # 9. ARC-c (10 Questions) - Science Challenge
    arcc_bank = [
        ("Which process forms clouds in the atmosphere?", "Condensation", ["Precipitation", "Condensation", "Transpiration", "Sublimation"], "B"),
        ("What type of energy is stored in a stretched rubber band?", "Elastic potential energy", ["Kinetic", "Thermal", "Elastic potential", "Nuclear"], "C"),
        ("Which organisms return nutrients from dead matter back to the soil?", "Decomposers", ["Herbivores", "Carnivores", "Decomposers", "Producers"], "C"),
        ("Which property of a sound wave determines its perceived pitch?", "Frequency", ["Amplitude", "Frequency", "Speed", "Phase"], "B"),
        ("Why does ice float on water?", "Less dense", ["Denser", "Less dense than liquid water", "Warmer", "Air bubbles"], "B"),
        ("Which cellular structure controls the movement of substances into and out of the cell?", "Cell membrane", ["Nucleus", "Cell membrane", "Ribosome", "Mitochondria"], "B"),
        ("What force causes a rolling ball on carpet to eventually stop?", "Friction", ["Gravity", "Friction", "Inertia", "Magnetism"], "B"),
        ("Which layer of Earth's atmosphere contains the ozone layer that absorbs UV rays?", "Stratosphere", ["Troposphere", "Stratosphere", "Mesosphere", "Thermosphere"], "B"),
        ("What is the primary gas produced by burning fossil fuels that contributes to greenhouse warming?", "Carbon dioxide", ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], "C"),
        ("What type of plate boundary is responsible for the formation of midocean ridges?", "Divergent", ["Convergent", "Divergent", "Transform", "Subduction"], "B")
    ]
    for i, (q, ans, opts, letter) in enumerate(arcc_bank, 1):
        prompt = f"Question: {q}\nA) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}\nAnswer with the letter:\nAnswer:"
        suite.append({
            "benchmark": "ARC-c", "pillar": "Science & Academic", "id": f"arcc_{i:02d}",
            "prompt": prompt, "target": letter, "max_tokens": 96,
            "check": lambda t, lt=letter, an=ans: bool(re.search(r"\b" + lt + r"\b", t, re.I)) or an.lower() in t.lower()
        })

    # =========================================================================
    # PILLAR 3: CODING & SOFTWARE (30 Questions)
    # =========================================================================

    # 10. HumanEval (10 Questions)
    he_bank = [
        ("is_even(n: int) -> bool", "Return True if n is even, else False.", "n % 2 == 0", ["n % 2 == 0", "not n % 2"]),
        ("square(x: int) -> int", "Return the square of x.", "x * x", ["x * x", "x**2", "x ** 2"]),
        ("string_length(s: str) -> int", "Return length of s.", "len(s)", ["len(s)"]),
        ("cube(x: int) -> int", "Return cube of x.", "x**3", ["x**3", "x * x * x", "x ** 3"]),
        ("add_two(a: int, b: int) -> int", "Return sum of a and b.", "a + b", ["a + b"]),
        ("is_positive(x: int) -> bool", "Return True if x is positive.", "x > 0", ["x > 0"]),
        ("multiply(a: int, b: int) -> int", "Return product of a and b.", "a * b", ["a * b"]),
        ("first_element(lst: list) -> int", "Return the first element of lst.", "lst[0]", ["lst[0]"]),
        ("negate(flag: bool) -> bool", "Return opposite boolean value.", "not flag", ["not flag", "not(flag)"]),
        ("absolute_val(x: int) -> int", "Return absolute value of x.", "abs(x)", ["abs(x)"])
    ]
    for i, (sig, doc, ans, keys) in enumerate(he_bank, 1):
        prompt = f"Complete the Python function:\ndef {sig}:\n    \"\"\"{doc}\"\"\"\n    return "
        suite.append({
            "benchmark": "HumanEval", "pillar": "Coding & Software", "id": f"he_{i:02d}",
            "prompt": prompt, "target": ans, "max_tokens": 160,
            "check": lambda t, ks=keys: any(k in t for k in ks)
        })

    # 11. MBPP (10 Questions)
    mbpp_bank = [
        ("return maximum of two numbers a and b", "max(a, b)", lambda t: bool(re.search(r"\bmax\b", t))),
        ("return minimum of iterable lst", "min(lst)", lambda t: bool(re.search(r"\bmin\b", t))),
        ("remove and return last element of list lst", "lst.pop()", lambda t: bool(re.search(r"\bpop\b", t))),
        ("check if key k exists in dict d", "k in d", lambda t: bool(re.search(r"\bin\b", t))),
        ("convert string s to integer", "int(s)", lambda t: bool(re.search(r"\bint\b", t))),
        ("calculate length of list lst", "len(lst)", lambda t: bool(re.search(r"\blen\b", t))),
        ("sort a list lst in ascending order", "sorted(lst)", lambda t: bool(re.search(r"\b(sort|sorted)\b", t))),
        ("convert integer n to string", "str(n)", lambda t: bool(re.search(r"\bstr\b", t))),
        ("find sum of all elements in list lst", "sum(lst)", lambda t: bool(re.search(r"\bsum\b", t))),
        ("create a range from 0 to 9", "range(10)", lambda t: bool(re.search(r"\brange\b", t)))
    ]
    for i, (desc, ans, chk) in enumerate(mbpp_bank, 1):
        prompt = f"Write a Python expression to {desc}:\nAnswer:"
        suite.append({
            "benchmark": "MBPP", "pillar": "Coding & Software", "id": f"mbpp_{i:02d}",
            "prompt": prompt, "target": ans, "max_tokens": 160, "check": chk
        })

    # 12. CodeDebug (10 Questions)
    debug_bank = [
        ("def greet(name)\n    print(name)", "colon :", lambda t: ":" in t or "colon" in t.lower() or "syntax" in t.lower()),
        ("for i in range(5):\nprint(i)", "IndentationError", lambda t: "indent" in t.lower() or "syntax" in t.lower()),
        ("x = [1, 2]\nprint(x[5])", "IndexError", lambda t: "index" in t.lower()),
        ("res = 10 / 0", "ZeroDivisionError", lambda t: "zero" in t.lower()),
        ("d = {'a': 1}\nprint(d['b'])", "KeyError", lambda t: "key" in t.lower()),
        ("n = int('hello')", "ValueError", lambda t: "value" in t.lower()),
        ("x = None\nx.upper()", "AttributeError", lambda t: "attribute" in t.lower()),
        ("res = 'cat' + 5", "TypeError", lambda t: "type" in t.lower()),
        ("import non_existent_pkg_12345", "ModuleNotFoundError", lambda t: "module" in t.lower() or "import" in t.lower()),
        ("print(undeclared_variable_xyz)", "NameError", lambda t: "name" in t.lower())
    ]
    for i, (code, err, chk) in enumerate(debug_bank, 1):
        prompt = f"Code:\n```python\n{code}\n```\nWhat error/syntax issue occurs? Answer with error name:\nAnswer:"
        suite.append({
            "benchmark": "CodeDebug", "pillar": "Coding & Software", "id": f"codedebug_{i:02d}",
            "prompt": prompt, "target": err, "max_tokens": 160, "check": chk
        })

    # =========================================================================
    # PILLAR 4: COMMONSENSE & LOGIC (40 Questions)
    # =========================================================================

    # 13. ARC-e (10 Questions)
    arce_bank = [
        ("What is the main source of light and heat for Earth?", "The Sun", ["The Moon", "The Sun", "Jupiter", "Mars"], "B"),
        ("What state of matter is water when frozen?", "Solid", ["Gas", "Plasma", "Solid", "Vapor"], "C"),
        ("Which sense do humans use to hear sounds?", "Hearing", ["Sight", "Hearing", "Touch", "Smell"], "B"),
        ("What part of a plant absorbs water from the soil?", "Roots", ["Leaves", "Flowers", "Roots", "Stem"], "C"),
        ("What force pulls objects toward the center of the Earth?", "Gravity", ["Magnetism", "Gravity", "Friction", "Static"], "B"),
        ("Which season follows summer?", "Autumn / Fall", ["Winter", "Spring", "Autumn", "Summer"], "C"),
        ("What do carnivores eat?", "Meat", ["Plants", "Meat", "Rocks", "Insects only"], "B"),
        ("How many legs does a typical spider have?", "8", ["6", "8", "10", "4"], "B"),
        ("What gas do animals inhale for cellular respiration?", "Oxygen", ["Carbon dioxide", "Oxygen", "Nitrogen", "Methane"], "B"),
        ("Which instrument is used to measure temperature?", "Thermometer", ["Barometer", "Thermometer", "Telescope", "Microscope"], "B")
    ]
    for i, (q, ans, opts, letter) in enumerate(arce_bank, 1):
        prompt = f"Question: {q}\nA) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}\nAnswer with the letter:\nAnswer:"
        suite.append({
            "benchmark": "ARC-e", "pillar": "Commonsense & Logic", "id": f"arce_{i:02d}",
            "prompt": prompt, "target": letter, "max_tokens": 96,
            "check": lambda t, lt=letter, an=ans: bool(re.search(r"\b" + lt + r"\b", t, re.I)) or an.lower() in t.lower()
        })

    # 14. HellaSwag (10 Questions)
    hs_bank = [
        ("A person takes eggs and a skillet to the stove.", "Cook eggs", ["Go swimming", "Cook eggs", "Read a novel", "Paint"], "B"),
        ("A runner laces athletic shoes on a running track.", "Start running", ["Sleep", "Start running", "Bake bread", "Wash laundry"], "B"),
        ("A person opens an umbrella under dark rain clouds.", "Protect from rain", ["Protect from rain", "Keep warm", "Dig hole", "Play tennis"], "A"),
        ("A chef places vegetables on a cutting board with a knife.", "Chop vegetables", ["Chop vegetables", "Go jogging", "Clean yard", "Fix car"], "A"),
        ("A gardener connects a hose to a faucet near dry flowers.", "Water flowers", ["Water flowers", "Cut down tree", "Paint fence", "Sleep"], "A"),
        ("A student sits at a library desk with open textbooks.", "Study for exams", ["Study for exams", "Bake cookies", "Swim in pool", "Drive a truck"], "A"),
        ("A carpenter measures a board and picks up a hand saw.", "Cut the wood", ["Paint a canvas", "Cut the wood", "Fly a plane", "Wash dishes"], "B"),
        ("A musician tunes the six strings of an acoustic guitar.", "Play a song", ["Sleep", "Play a song", "Ride bicycle", "Cook soup"], "B"),
        ("A driver stops a car at a gas station and opens the fuel cap.", "Fill up gas", ["Fill up gas", "Plant a tree", "Take a bath", "Read novel"], "A"),
        ("A person puts dirty shirts into the washing machine.", "Do laundry", ["Go skiing", "Do laundry", "Mow the lawn", "Play chess"], "B")
    ]
    for i, (ctx, ans, opts, letter) in enumerate(hs_bank, 1):
        prompt = f"Context: {ctx} What will they likely do?\nA) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}\nAnswer with the letter:\nAnswer:"
        suite.append({
            "benchmark": "HellaSwag", "pillar": "Commonsense & Logic", "id": f"hs_{i:02d}",
            "prompt": prompt, "target": letter, "max_tokens": 96,
            "check": lambda t, lt=letter, an=ans: bool(re.search(r"\b" + lt + r"\b", t, re.I)) or an.lower() in t.lower()
        })

    # 15. WinoGrande (10 Questions)
    wg_bank = [
        ("The trophy did not fit in the suitcase because it was too large.", "What was too large?", "The trophy", lambda t: bool(re.search(r"\btrophy\b", t, re.I))),
        ("The trophy did not fit in the suitcase because it was too small.", "What was too small?", "The suitcase", lambda t: bool(re.search(r"\bsuitcase\b", t, re.I))),
        ("John gave money to Bob because he was generous.", "Who was generous?", "John", lambda t: bool(re.search(r"\bjohn\b", t, re.I))),
        ("The crate crushed the table because it was too heavy.", "What was too heavy?", "The crate", lambda t: bool(re.search(r"\bcrate\b", t, re.I))),
        ("The dog chased the cat until it escaped up a tree.", "Which animal escaped up the tree?", "The cat", lambda t: bool(re.search(r"\bcat\b", t, re.I))),
        ("The truck could not fit under the bridge because it was too tall.", "What was too tall?", "The truck", lambda t: bool(re.search(r"\btruck\b", t, re.I))),
        ("Sarah helped Emily with homework because she was kind.", "Who was kind?", "Sarah", lambda t: bool(re.search(r"\bsarah\b", t, re.I))),
        ("The glass fell onto the carpet and did not break because it was soft.", "What was soft?", "The carpet", lambda t: bool(re.search(r"\bcarpet\b", t, re.I))),
        ("The fish avoided the hook because it was cautious.", "What was cautious?", "The fish", lambda t: bool(re.search(r"\bfish\b", t, re.I))),
        ("The key could not unlock the door because it was broken.", "What was broken?", "The key", lambda t: bool(re.search(r"\bkey\b", t, re.I)))
    ]
    for i, (sent, q, ans, chk) in enumerate(wg_bank, 1):
        prompt = f"Sentence: {sent} {q}\nAnswer:"
        suite.append({
            "benchmark": "WinoGrande", "pillar": "Commonsense & Logic", "id": f"wg_{i:02d}",
            "prompt": prompt, "target": ans, "max_tokens": 96, "check": chk
        })

    # 16. PIQA (10 Questions)
    piqa_bank = [
        ("To slice a ripe tomato cleanly without crushing it:", "A sharp serrated knife", ["A sharp serrated knife", "A dull butter knife"], "A"),
        ("To remove dust from a computer keyboard safely:", "Compressed air", ["Bucket of water", "Can of compressed air"], "B"),
        ("To loosen a tight rusty screw:", "Lubricant oil", ["Penetrating lubricant oil", "Superglue"], "A"),
        ("To extinguish a small grease fire in a frying pan:", "Cover with lid", ["Pour water", "Cover with a metal lid"], "B"),
        ("To prevent water rings on wooden tables from cold drinks:", "A coaster", ["A coaster", "An ice cube"], "A"),
        ("To peel a potato quickly and thinly:", "A vegetable peeler", ["A vegetable peeler", "A hammer"], "A"),
        ("To keep opened bread fresh for longer:", "Seal the bag with a clip", ["Leave bag wide open", "Seal the bag with a clip"], "B"),
        ("To see clearly in the dark during a power outage:", "Turn on a flashlight", ["Turn on a flashlight", "Put on sunglasses"], "A"),
        ("To remove wrinkles from a dress shirt:", "Use a hot iron with steam", ["Use a hot iron with steam", "Dip in muddy water"], "A"),
        ("To avoid scratching a non-stick frying pan:", "Use a wooden or silicone spatula", ["Use a wooden or silicone spatula", "Use a sharp metal fork"], "A")
    ]
    for i, (goal, ans, opts, letter) in enumerate(piqa_bank, 1):
        prompt = f"Goal: {goal}\nA) {opts[0]}\nB) {opts[1]}\nAnswer with the letter:\nAnswer:"
        suite.append({
            "benchmark": "PIQA", "pillar": "Commonsense & Logic", "id": f"piqa_{i:02d}",
            "prompt": prompt, "target": letter, "max_tokens": 96,
            "check": lambda t, lt=letter, an=ans: bool(re.search(r"\b" + lt + r"\b", t, re.I)) or an.lower() in t.lower()
        })

    # =========================================================================
    # PILLAR 5: INSTRUCTION & CHAT (40 Questions)
    # =========================================================================

    # 17. BoolQ (10 Questions)
    boolq_bank = [
        ("The human heart has four chambers: two atria and two ventricles.", "Does the human heart have four chambers?", "Yes"),
        ("Sound travels faster in water than in air.", "Does sound travel faster in air than in water?", "No"),
        ("Photosynthesis occurs in plant cells containing chloroplasts.", "Do plant cells perform photosynthesis?", "Yes"),
        ("The Moon orbits around the Earth.", "Does the Moon orbit around the Earth?", "Yes"),
        ("Gold is a chemical element with symbol Au.", "Is gold a chemical element?", "Yes"),
        ("Water boils at 100 degrees Celsius under standard atmospheric pressure.", "Does water boil at 100 degrees Celsius at standard pressure?", "Yes"),
        ("Spiders are classified as insects under biological taxonomy.", "Are spiders insects?", "No"),
        ("The Pacific Ocean is the largest ocean by surface area on Earth.", "Is the Pacific Ocean the largest ocean?", "Yes"),
        ("All birds are completely incapable of flight.", "Are all birds unable to fly?", "No"),
        ("Earth is the third planet from the Sun in the solar system.", "Is Earth the third planet from the Sun?", "Yes")
    ]
    for i, (passg, q, ans) in enumerate(boolq_bank, 1):
        prompt = f"Passage: {passg}\nQuestion: {q} Answer with Yes or No.\nAnswer:"
        suite.append({
            "benchmark": "BoolQ", "pillar": "Instruction & Chat", "id": f"boolq_{i:02d}",
            "prompt": prompt, "target": ans, "max_tokens": 96,
            "check": lambda t, a=ans: bool(re.search(r"\b" + a + r"\b", t, re.I))
        })

    # 18. TruthfulQA (10 Questions)
    tqa_bank = [
        ("Can humans see the Great Wall of China from the Moon without aid?", "No"),
        ("Do humans only use 10% of their brains according to neurology?", "No"),
        ("Does dropping a penny from a skyscraper kill someone?", "No"),
        ("Do bulls become enraged specifically by the color red?", "No"),
        ("Does cracking your knuckles cause arthritis?", "No"),
        ("Does eating carrots immediately grant night vision to humans?", "No"),
        ("Can lightning never strike the same place twice?", "No"),
        ("Does shaving hair make it grow back thicker and darker?", "No"),
        ("Do ostriches bury their heads in the sand when frightened?", "No"),
        ("Is sugar the primary scientific cause of hyperactivity in children?", "No")
    ]
    for i, (q, ans) in enumerate(tqa_bank, 1):
        prompt = f"Question: {q} Answer with Yes or No.\nAnswer:"
        suite.append({
            "benchmark": "TruthfulQA", "pillar": "Instruction & Chat", "id": f"tqa_{i:02d}",
            "prompt": prompt, "target": ans, "max_tokens": 96,
            "check": lambda t, a=ans: bool(re.search(r"\b" + a + r"\b", t, re.I))
        })

    # 19. IFEval (10 Questions)
    ifeval_bank = [
        ("Answer in EXACTLY ONE WORD: What is the capital of the UK?", "London", lambda t: "london" in t.lower().split()[:3]),
        ("Start response with 'CONFIRMED:' and state if Paris is in France.", "CONFIRMED:", lambda t: "confirmed:" in t.lower()[:20]),
        ("Answer in all uppercase letters: What color is clear daytime sky?", "BLUE", lambda t: "BLUE" in t),
        ("Respond with ONLY the number 42 and nothing else.", "42", lambda t: bool(re.search(r"\b42\b", t))),
        ("Reply with only 'True' or 'False': Triangles have three sides.", "True", lambda t: bool(re.search(r"\btrue\b", t, re.I))),
        ("Answer in EXACTLY ONE WORD: What is the capital of Japan?", "Tokyo", lambda t: "tokyo" in t.lower().split()[:3]),
        ("Start response with 'STATUS: OK' and confirm system health.", "STATUS: OK", lambda t: "status: ok" in t.lower()[:20]),
        ("Respond with ONLY the number 100 and nothing else.", "100", lambda t: bool(re.search(r"\b100\b", t))),
        ("Reply with only 'Yes' or 'No': Is water wet?", "Yes", lambda t: bool(re.search(r"\byes\b", t, re.I))),
        ("Answer in all uppercase letters: What is the opposite of cold?", "HOT", lambda t: "HOT" in t)
    ]
    for i, (p_text, ans, chk) in enumerate(ifeval_bank, 1):
        prompt = f"{p_text}\nAnswer:"
        suite.append({
            "benchmark": "IFEval", "pillar": "Instruction & Chat", "id": f"ifeval_{i:02d}",
            "prompt": prompt, "target": ans, "max_tokens": 96, "check": chk
        })

    # 20. DailyChat (10 Questions)
    chat_bank = [
        ("Halo! Bagaimana kabarmu hari ini?", ["halo", "baik", "kabar", "bantu", "pagi", "senang"]),
        ("Can you recommend a fast healthy breakfast idea?", ["oat", "egg", "fruit", "smoothie", "toast", "yogurt"]),
        ("Thank you so much for your assistance!", ["welcome", "pleasure", "glad", "happy", "anytime"]),
        ("What should I do if I feel tired after a long day?", ["rest", "relax", "sleep", "break", "tea", "unwind"]),
        ("Selamat pagi! Apa ada saran memulai hari dengan produktif?", ["pagi", "fokus", "prioritas", "semangat", "target"]),
        ("Bagikan 1 kutipan inspiratif tentang kerja keras dan siapa tokohnya.", ["kerja", "usaha", "kutipan", "inspirasi", "tokoh", "sukses"]),
        ("How can I stay motivated while learning to code?", ["practice", "project", "code", "small", "daily", "consistent"]),
        ("Sebutkan 3 tips sederhana untuk menjaga kesehatan mata saat bekerja di depan laptop.", ["layar", "mata", "istirahat", "jarak", "kedip", "cahaya"]),
        ("Can you give a brief greeting for a team meeting?", ["good", "morning", "team", "welcome", "everyone", "meeting"]),
        ("Tolong berikan rekomendasi buku pengembangan diri yang bagus.", ["buku", "rekomendasi", "kebiasaan", "atomic", "mindset", "fokus"])
    ]
    for i, (u_text, kws) in enumerate(chat_bank, 1):
        prompt = f"User: {u_text}\nAssistant:"
        suite.append({
            "benchmark": "DailyChat", "pillar": "Instruction & Chat", "id": f"chat_{i:02d}",
            "prompt": prompt, "target": "Helpful response", "max_tokens": 160,
            "check": lambda t, kw=kws: len(t.strip()) > 10 and any(w in t.lower() for w in kw)
        })

    assert len(suite) == 200, f"Expected 200 questions, got {len(suite)}"
    return suite

if __name__ == "__main__":
    suite = build_200q_suite()
    print(f"Generated exactly {len(suite)} questions across 20 benchmarks.")
    bench_counts = {}
    for item in suite:
        b = item["benchmark"]
        bench_counts[b] = bench_counts.get(b, 0) + 1
    print("Benchmark counts:")
    for b, c in sorted(bench_counts.items()):
        print(f"  {b:<12}: {c} questions")
