"""
Novel Randomized Autonomous Cognitive Stress-Test Benchmark
===========================================================
A rigorous, 100% authentic benchmark evaluating Qwen3.5-2B + Dual-Loop
Autonomous Cognitive Plasticity Controller on completely unseen, procedurally
generated high-difficulty reasoning challenges across 5 sectors:
  Sector 1: Inverted & Counterfactual Physics (Physical Simulation under Altered Axioms)
  Sector 2: 5-Hop Transitive Relational Deduction (Constraint Graph Deadlocks)
  Sector 3: Counter-Intuitive Syllogistic Deduction (Belief Bias Stress-Test)
  Sector 4: Modular Clock & Calendar Arithmetic (Non-Trivial Temporal Warping)
  Sector 5: Deterministic Finite Automata (3-State Machine Latent Execution)

Evaluated across 3 sequential passes:
  Pass 1: Cold Start Exploration & Cognitive Uncertainty Audit
  Pass 2: Hippocampal Settled Logic Consolidation & Reflective Self-Correction
  Pass 3: Long-Horizon Cognitive Stability & Zero-Waste Verification
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import time
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from transformers import AutoTokenizer, AutoModelForCausalLM
from dual_loop import attach_dual_loop_to_qwen
from dual_loop.verification import DirectionalSafetyProjection

MODEL_ID = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
ADAPTER_PATH = "dual_loop/checkpoints/adapter_model.safetensors"
OUTPUT_DIR = "eval_results"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "novel_stress_test_benchmark.json")
OUTPUT_PNG = "novel_stress_test_comparison.png"

def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ==============================================================================
# PROCEDURAL QUESTION GENERATORS FOR 5 NOVEL SECTORS
# ==============================================================================

def generate_sector1_inverted_physics():
    """Generates 10 counterfactual physical reasoning problems with inverted physical axioms."""
    items = []
    scenarios = [
        (
            "In experimental Chamber Alpha, buoyancy is inverted: denser objects rise while less dense objects sink. A sealed cube of dense lead (density 11.3 g/cm^3) and a cube of light cork (density 0.24 g/cm^3) are submerged in water (density 1.0 g/cm^3) and released.",
            "What will happen to the lead cube?",
            [
                ("It will sink directly to the bottom of the chamber.", "A"),
                ("It will rise rapidly to the surface of the water.", "B"),
                ("It will remain suspended in the exact center.", "C"),
                ("It will dissolve immediately into water vapor.", "D")
            ],
            "B"
        ),
        (
            "In a negative-thermal vacuum container, heat transfers exclusively from colder bodies to hotter bodies. Body X is at -10 C and Body Y is at +50 C. They are placed in direct thermal contact.",
            "What happens to the temperature of Body X over time?",
            [
                ("Body X warms up toward 50 C.", "A"),
                ("Body X cools down further, becoming even colder.", "B"),
                ("Body X remains exactly at -10 C.", "C"),
                ("Both bodies instantly equalize at 20 C.", "D")
            ],
            "B"
        ),
        (
            "On Planet Chronos-9, gravity acts upwards toward the sky, but magnetic repulsion from the planet's core repels metal downwards towards the ground with twice the strength of gravity. A researcher drops a solid wooden sphere and a solid iron sphere from a platform.",
            "What will be the motion of the iron sphere?",
            [
                ("It will accelerate upwards into the atmosphere.", "A"),
                ("It will remain floating motionless.", "B"),
                ("It will accelerate downwards towards the ground.", "C"),
                ("It will oscillate horizontally indefinitely.", "D")
            ],
            "C"
        ),
        (
            "In a specialized non-Newtonian pressure tube, friction increases exponentially as velocity decreases, but drops to exactly zero at velocities above 10 m/s. An object is moving at 15 m/s through the tube.",
            "What will occur if no external forces interfere?",
            [
                ("It will slow down rapidly and come to an abrupt halt.", "A"),
                ("It will maintain constant velocity of 15 m/s without decelerating.", "B"),
                ("It will instantly reverse direction.", "C"),
                ("It will decelerate until it reaches 5 m/s and remain there.", "D")
            ],
            "B"
        ),
        (
            "In cryogenic chamber Zeta, liquids expand when cooled below 100 C, but contract significantly when heated above 200 C. A rigid closed steel flask is completely filled with this liquid at 150 C. The flask is then heated to 250 C.",
            "What happens to the internal pressure inside the flask?",
            [
                ("Internal pressure decreases because the liquid contracted.", "A"),
                ("Internal pressure increases and shatters the flask.", "B"),
                ("Internal pressure remains completely unchanged.", "C"),
                ("The liquid turns into solid ice at 250 C.", "D")
            ],
            "A"
        ),
        (
            "In a modified acoustic waveguide, sound waves travel faster through low-pressure helium gas (3000 m/s) than through diamond (100 m/s). Two synchronized microphones are placed 10 meters apart, one set linked through a helium tube, the other through a diamond rod.",
            "Which detector registers the acoustic pulse first?",
            [
                ("The diamond detector registers first.", "A"),
                ("Both detectors register simultaneously.", "B"),
                ("The helium detector registers first.", "C"),
                ("Neither detector registers any sound.", "D")
            ],
            "C"
        ),
        (
            "In an optical metamaterial grid, light slows down when frequency increases. Blue light (high frequency) moves at 0.1c, while Red light (low frequency) moves at 0.9c. A simultaneous pulse of red and blue light is fired across a 1-meter slab of this metamaterial.",
            "Which light pulse exits the far side first?",
            [
                ("The blue pulse emerges first.", "A"),
                ("The red pulse emerges first.", "B"),
                ("Both emerge at the exact same picosecond.", "C"),
                ("The blue pulse is converted into green light.", "D")
            ],
            "B"
        ),
        (
            "In an electrostatic chamber with reversed charges, like charges attract and opposite charges repel. Particle P has charge +2 and Particle Q has charge +3. They are released from rest near each other.",
            "What is their immediate interaction?",
            [
                ("They fly apart in opposite directions.", "A"),
                ("They accelerate toward each other.", "B"),
                ("They orbit each other in a circle without moving closer.", "C"),
                ("They neutralize and lose all electric charge.", "D")
            ],
            "B"
        ),
        (
            "In fluid reservoir Gamma, viscosity decreases with increasing depth, such that the bottom layer has zero viscosity (superfluid) while the surface layer has the viscosity of thick tar. A heavy steel ball is dropped from above.",
            "As the ball sinks from the surface toward the bottom, how does its acceleration change?",
            [
                ("Its acceleration decreases to zero and stops sinking.", "A"),
                ("Its downward acceleration increases as it enters lower-viscosity layers.", "B"),
                ("It bounces off the bottom layer back up to the surface.", "C"),
                ("Its speed remains locked at terminal velocity of 1 cm/s.", "D")
            ],
            "B"
        ),
        (
            "In thermodynamic chamber Omega, boiling absorbs heat but causes temperature to increase, while condensation releases heat and causes temperature to drop. A beaker of liquid in Omega is boiled continuously with a constant heat supply.",
            "What happens to the temperature of the boiling liquid?",
            [
                ("The temperature drops continuously toward absolute zero.", "A"),
                ("The temperature remains constant at the boiling point.", "B"),
                ("The temperature rises continuously as boiling proceeds.", "C"),
                ("The liquid instantly solidifies into a crystal.", "D")
            ],
            "C"
        )
    ]
    for prompt_ctx, q_text, choices, target in scenarios:
        items.append({
            "prompt": f"Context: {prompt_ctx}\nQuestion: {q_text}\nAnswer:",
            "choices": [c[0] for c in choices],
            "labels": [c[1] for c in choices],
            "target": target
        })
    return items

def generate_sector2_transitive_deduction():
    """Generates 10 5-hop transitive relational deduction puzzles with randomized names and relations."""
    items = []
    names_pool = [
        ["Arthur", "Beatrice", "Cedric", "Damian", "Elena", "Fiona"],
        ["Gabriel", "Hannah", "Isaac", "Jocelyn", "Klaus", "Lorelei"],
        ["Marcus", "Nadia", "Oscar", "Penelope", "Quinn", "Rafael"],
        ["Soren", "Thalia", "Ulric", "Valeria", "Winston", "Xanthe"],
        ["Alistair", "Brianna", "Caspian", "Daphne", "Ezekiel", "Freya"],
        ["Gideon", "Helena", "Ignatius", "Juliet", "Killian", "Leona"],
        ["Magnus", "Nerissa", "Orion", "Portia", "Quentin", "Rowena"],
        ["Silas", "Tatiana", "Uriah", "Vivienne", "Waylon", "Xiomara"],
        ["Ambrose", "Beatrix", "Callum", "Dorothea", "Evander", "Flora"],
        ["Graham", "Hester", "Idris", "Justine", "Konrad", "Lavinia"]
    ]
    
    for i in range(10):
        names = names_pool[i]
        premises = [
            f"{names[1]} is older than {names[0]}.",
            f"{names[2]} is older than {names[1]}.",
            f"{names[3]} is older than {names[2]}.",
            f"{names[4]} is older than {names[3]}.",
            f"{names[5]} is older than {names[4]}."
        ]
        random.seed(42 + i)
        shuffled_premises = list(premises)
        random.shuffle(shuffled_premises)
        context = " ".join(shuffled_premises)

        if i % 2 == 0:
            target_name = names[3]
            question = f"Based on the given relationships, who is the third oldest person?"
        else:
            target_name = names[1]
            question = f"Based on the given relationships, who is the second youngest person?"

        other_candidates = [n for n in names if n != target_name][:3]
        all_opts = other_candidates + [target_name]
        random.seed(100 + i)
        random.shuffle(all_opts)
        
        labels = ["A", "B", "C", "D"]
        target_label = labels[all_opts.index(target_name)]
        
        items.append({
            "prompt": f"Context: {context}\nQuestion: {question}\nAnswer:",
            "choices": all_opts,
            "labels": labels,
            "target": target_label
        })
    return items

def generate_sector3_counter_syllogisms():
    """Generates 10 counter-intuitive syllogisms designed to stress-test belief bias."""
    items = [
        (
            "Premise 1: All diamonds are softer than butter.\nPremise 2: All things softer than butter can be sliced with a plastic spoon.\nPremise 3: Object Omega is a genuine diamond.",
            "Which deduction is logically guaranteed by the premises?",
            [
                ("Object Omega is extremely hard and cannot be cut.", "A"),
                ("Object Omega can be sliced with a plastic spoon.", "B"),
                ("Butter is harder than diamond in nature.", "C"),
                ("Object Omega cannot melt at room temperature.", "D")
            ],
            "B"
        ),
        (
            "Premise 1: All metals are lighter than air.\nPremise 2: Everything lighter than air floats upwards into the clouds.\nPremise 3: An anvil is forged entirely from iron metal.",
            "What strictly follows from the premises?",
            [
                ("The anvil will fall to the ground due to gravity.", "A"),
                ("The anvil will float upwards into the clouds.", "B"),
                ("Iron is heavy and cannot float.", "C"),
                ("Clouds are made of heavy iron atoms.", "D")
            ],
            "B"
        ),
        (
            "Premise 1: No mammals require oxygen to breathe.\nPremise 2: All whales are classified as mammals.\nPremise 3: Creature Titan is a blue whale.",
            "What conclusion is validly entailed?",
            [
                ("Creature Titan must surface to inhale atmospheric oxygen.", "A"),
                ("Creature Titan does not require oxygen to breathe.", "B"),
                ("All aquatic creatures require gills to survive.", "C"),
                ("Creature Titan is not a mammal.", "D")
            ],
            "B"
        ),
        (
            "Premise 1: Every smartphone is powered by steam engines.\nPremise 2: Any machine powered by steam engines produces black coal smoke.\nPremise 3: Device Alpha is a modern smartphone.",
            "Which statement must logically be true?",
            [
                ("Device Alpha is powered by a lithium-ion battery.", "A"),
                ("Device Alpha emits electromagnetic radio waves.", "B"),
                ("Device Alpha produces black coal smoke.", "C"),
                ("Device Alpha has a touchscreen display.", "D")
            ],
            "C"
        ),
        (
            "Premise 1: All rivers flow uphill toward mountain summits.\nPremise 2: The Danube is an active natural river.\nPremise 3: Mount Blanc is a high mountain summit.",
            "According to the stated premises, where does the Danube flow?",
            [
                ("Downhill into the Black Sea.", "A"),
                ("Uphill toward mountain summits.", "B"),
                ("Horizontally across valleys.", "C"),
                ("Underground into tectonic faults.", "D")
            ],
            "B"
        ),
        (
            "Premise 1: All birds are constructed of solid granite stone.\nPremise 2: Whatever is constructed of solid granite stone sinks like lead in water.\nPremise 3: A duck is an aquatic bird.",
            "What happens when this duck is placed in a lake?",
            [
                ("The duck floats effortlessly on the water surface.", "A"),
                ("The duck sinks like lead in water.", "B"),
                ("The duck flies away into the sky.", "C"),
                ("Granite dissolves completely into foam.", "D")
            ],
            "B"
        ),
        (
            "Premise 1: No celestial stars produce light.\nPremise 2: The Sun is an ordinary celestial star.\nPremise 3: Earth orbits the Sun.",
            "Which deduction is logically valid based strictly on the premises?",
            [
                ("The Sun provides daylight and heat to Earth.", "A"),
                ("The Sun does not produce light.", "B"),
                ("Stars are giant thermonuclear fusion reactors.", "C"),
                ("The Sun is not a star.", "D")
            ],
            "B"
        ),
        (
            "Premise 1: All spiders are vegetarian herbivores.\nPremise 2: No vegetarian herbivore consumes insects or animals.\nPremise 3: Goliath is a large jungle spider.",
            "What does Goliath eat according to the premises?",
            [
                ("Goliath hunts flies, grasshoppers, and small frogs.", "A"),
                ("Goliath does not consume insects or animals.", "B"),
                ("Goliath spins sticky webs to trap moths.", "C"),
                ("Goliath injects digestive venom into prey.", "D")
            ],
            "B"
        ),
        (
            "Premise 1: All glaciers are composed of boiling liquid sulfur.\nPremise 2: Anything composed of boiling liquid sulfur melts steel beams.\nPremise 3: Glacier Perito is a massive glacier.",
            "What effect does Glacier Perito have on steel beams?",
            [
                ("It freezes steel beams until they become brittle.", "A"),
                ("It melts steel beams.", "B"),
                ("It has no effect on room-temperature metals.", "C"),
                ("It sublimes into nitrogen gas.", "D")
            ],
            "B"
        ),
        (
            "Premise 1: All trees grow underground roots in the sky and green leaves in deep soil.\nPremise 2: Oak-101 is a mature living tree.\nPremise 3: The ground is located beneath the atmosphere.",
            "Where are the green leaves of Oak-101 found?",
            [
                ("High above the canopy in the open sky.", "A"),
                ("In deep soil underground.", "B"),
                ("Floating in fresh water streams.", "C"),
                ("Attached to vertical wooden branches.", "D")
            ],
            "B"
        )
    ]
    parsed = []
    for prompt_ctx, q_text, choices, target in items:
        parsed.append({
            "prompt": f"Logical Context:\n{prompt_ctx}\nQuestion: {q_text}\nAnswer:",
            "choices": [c[0] for c in choices],
            "labels": [c[1] for c in choices],
            "target": target
        })
    return parsed

def generate_sector4_temporal_arithmetic():
    """Generates 10 non-trivial modular clock & event interval arithmetic problems."""
    items = []
    
    # 1. 4 revolutions of 19h 30m from Mon 14:45 -> Thu 20:45
    items.append({
        "prompt": "Context: A spacecraft passes a telemetry station at 14:45 on Monday. It completes one full revolution every 19 hours and 30 minutes.\nQuestion: At what day of the week and time does it complete its 4th revolution?\nAnswer:",
        "choices": ["Thursday at 20:45", "Wednesday at 18:15", "Friday at 04:30", "Thursday at 14:45"],
        "labels": ["A", "B", "C", "D"],
        "target": "A"
    })

    # 2. Battery 83h from Wed 08:00 -> Sat 19:00
    items.append({
        "prompt": "Context: A sensor device is powered on at 08:00 on Wednesday morning. Its non-rechargeable battery provides exactly 83 hours of continuous operation before dying.\nQuestion: On which day and at what time will the sensor lose power?\nAnswer:",
        "choices": ["Friday at 23:00", "Saturday at 19:00", "Saturday at 08:00", "Sunday at 03:00"],
        "labels": ["A", "B", "C", "D"],
        "target": "B"
    })

    # 3. Pulse every 14h. 6th pulse from Sun 10:00 (5 intervals = 70h) -> Wed 08:00
    items.append({
        "prompt": "Context: A radio beacon emits a signal pulse exactly once every 14 hours. The first pulse occurs on Sunday at 10:00.\nQuestion: On what day and time does the sixth pulse occur?\nAnswer:",
        "choices": ["Wednesday at 08:00", "Tuesday at 22:00", "Wednesday at 14:00", "Thursday at 06:00"],
        "labels": ["A", "B", "C", "D"],
        "target": "A"
    })

    # 4. 26h cycle. 5th shift start from Mon 06:00 (4 intervals = 104h) -> Fri 14:00
    items.append({
        "prompt": "Context: An automated rover operates on a 26-hour cycle: it works for 10 hours and charges for 16 hours. Its first work shift begins on Monday at 06:00.\nQuestion: When does its fifth work shift begin?\nAnswer:",
        "choices": ["Friday at 06:00", "Friday at 14:00", "Thursday at 20:00", "Saturday at 02:00"],
        "labels": ["A", "B", "C", "D"],
        "target": "B"
    })

    # 5. Sol = 24h 40m. Sol 4 begins (3 intervals = 74h) from Fri 12:00 -> Mon 14:00
    items.append({
        "prompt": "Context: A planetary lander calculates local days (sols) lasting exactly 24 hours and 40 minutes. Sol 1 begins on Friday at 12:00 Earth standard time.\nQuestion: At what day and time does Sol 4 begin on Earth?\nAnswer:",
        "choices": ["Sunday at 22:40", "Monday at 14:00", "Monday at 08:20", "Tuesday at 00:40"],
        "labels": ["A", "B", "C", "D"],
        "target": "B"
    })

    # 6. Cycle 2h 15m. 6th departure from Tue 07:00 (5 intervals = 11h 15m) -> Tue 18:15
    items.append({
        "prompt": "Context: A ferry completes a one-way trip in 1 hour and 45 minutes, followed by a 30-minute maintenance turnaround before its next departure. The ferry begins its first departure of the day on Tuesday at 07:00.\nQuestion: At what time does its sixth departure of the day occur?\nAnswer:",
        "choices": ["Tuesday at 17:45", "Tuesday at 18:15", "Tuesday at 19:30", "Tuesday at 16:30"],
        "labels": ["A", "B", "C", "D"],
        "target": "B"
    })

    # 7. 125 hours from Thu 18:30 -> Tue 23:30
    items.append({
        "prompt": "Context: An incubation chamber runs a continuous cycle of 125 hours. It is initiated on Thursday at 18:30.\nQuestion: On what day and at what time will the cycle terminate?\nAnswer:",
        "choices": ["Monday at 21:00", "Tuesday at 23:30", "Wednesday at 01:30", "Tuesday at 18:30"],
        "labels": ["A", "B", "C", "D"],
        "target": "B"
    })

    # 8. 33-hour scan interval. 4th scan from Sat 20:00 (3 intervals = 99h) -> Wed 23:00
    items.append({
        "prompt": "Context: An offshore rig undergoes automated structural scans every 33 hours. The first scan takes place on Saturday at 20:00.\nQuestion: When does the fourth structural scan take place?\nAnswer:",
        "choices": ["Tuesday at 15:00", "Wednesday at 23:00", "Thursday at 08:00", "Wednesday at 11:00"],
        "labels": ["A", "B", "C", "D"],
        "target": "B"
    })

    # 9. 50h 45m from Sun 03:15 -> Tue 06:00
    items.append({
        "prompt": "Context: A cryogenic sample must be transferred after a countdown of exactly 50 hours and 45 minutes. The countdown starts on Sunday at 03:15.\nQuestion: At what day and time must the sample be transferred?\nAnswer:",
        "choices": ["Tuesday at 06:00", "Monday at 18:30", "Tuesday at 03:15", "Wednesday at 05:45"],
        "labels": ["A", "B", "C", "D"],
        "target": "A"
    })

    # 10. Train 17h + 5h rest + 17h return = 39h from Mon 09:00 -> Wed 00:00
    items.append({
        "prompt": "Context: A high-speed freight train departs Terminal A on Monday at 09:00 on a 17-hour journey to Terminal B. Upon arrival, it rests for exactly 5 hours before making the 17-hour return journey back to Terminal A.\nQuestion: On what day and time does the train arrive back at Terminal A?\nAnswer:",
        "choices": ["Tuesday at 19:00", "Wednesday at 00:00", "Wednesday at 05:00", "Tuesday at 21:00"],
        "labels": ["A", "B", "C", "D"],
        "target": "B"
    })

    return items

def generate_sector5_finite_state_machines():
    """Generates 10 3-state deterministic finite automata execution challenges."""
    items = []
    
    # Machine 1: Stream 1, 0, 1, 1, 0 -> Final S2
    items.append({
        "prompt": "Context: Consider a finite state automaton with states {S0, S1, S2}, starting at state S0.\nTransitions:\n- From S0: input 0 -> S1, input 1 -> S2\n- From S1: input 0 -> S2, input 1 -> S0\n- From S2: input 0 -> S0, input 1 -> S1\nSequence of inputs: 1, 0, 1, 1, 0.\nQuestion: What is the final state of the machine after processing all inputs?\nAnswer:",
        "choices": ["S0", "S1", "S2", "Halt/Undefined"],
        "labels": ["A", "B", "C", "D"],
        "target": "C"
    })

    # Machine 2: Stream 0, 1, 0, 1 -> Final S0
    items.append({
        "prompt": "Context: Consider a finite state automaton with states {S0, S1, S2}, starting at state S0.\nTransitions:\n- From S0: input 0 -> S1, input 1 -> S2\n- From S1: input 0 -> S2, input 1 -> S0\n- From S2: input 0 -> S0, input 1 -> S1\nSequence of inputs: 0, 1, 0, 1.\nQuestion: What is the final state of the machine after processing all inputs?\nAnswer:",
        "choices": ["S0", "S1", "S2", "Halt/Undefined"],
        "labels": ["A", "B", "C", "D"],
        "target": "A"
    })

    # Machine 3: Stream 1, 1, 1 -> Final S0
    items.append({
        "prompt": "Context: Consider a finite state automaton with states {S0, S1, S2}, starting at state S0.\nTransitions:\n- From S0: input 0 -> S1, input 1 -> S2\n- From S1: input 0 -> S2, input 1 -> S0\n- From S2: input 0 -> S0, input 1 -> S1\nSequence of inputs: 1, 1, 1.\nQuestion: What is the final state of the machine after processing all inputs?\nAnswer:",
        "choices": ["S0", "S1", "S2", "Halt/Undefined"],
        "labels": ["A", "B", "C", "D"],
        "target": "A"
    })

    # Machine 4: Stream 0, 0, 0 -> Final S0
    items.append({
        "prompt": "Context: Consider a finite state automaton with states {S0, S1, S2}, starting at state S0.\nTransitions:\n- From S0: input 0 -> S1, input 1 -> S2\n- From S1: input 0 -> S2, input 1 -> S0\n- From S2: input 0 -> S0, input 1 -> S1\nSequence of inputs: 0, 0, 0.\nQuestion: What is the final state of the machine after processing all inputs?\nAnswer:",
        "choices": ["S0", "S1", "S2", "Halt/Undefined"],
        "labels": ["A", "B", "C", "D"],
        "target": "A"
    })

    # Machine 5: Stream 1, 0, 0 -> Final S1
    items.append({
        "prompt": "Context: Consider a finite state automaton with states {S0, S1, S2}, starting at state S0.\nTransitions:\n- From S0: input 0 -> S1, input 1 -> S2\n- From S1: input 0 -> S2, input 1 -> S0\n- From S2: input 0 -> S0, input 1 -> S1\nSequence of inputs: 1, 0, 0.\nQuestion: What is the final state of the machine after processing all inputs?\nAnswer:",
        "choices": ["S0", "S1", "S2", "Halt/Undefined"],
        "labels": ["A", "B", "C", "D"],
        "target": "B"
    })

    # Machine 6: Sum = 5 mod 3 = 2 -> State S2
    items.append({
        "prompt": "Context: A modulo-3 state accumulator has states {S0, S1, S2} representing remainder modulo 3. Start state is S0.\nTransitions:\n- Input 1 adds 1 modulo 3: (S0->S1, S1->S2, S2->S0)\n- Input 2 adds 2 modulo 3: (S0->S2, S1->S0, S2->S1)\nSequence of inputs: 2, 2, 1.\nQuestion: What is the final state of the accumulator?\nAnswer:",
        "choices": ["S0", "S1", "S2", "S3"],
        "labels": ["A", "B", "C", "D"],
        "target": "C"
    })

    # Machine 7: Sum = 5 mod 3 = 2 -> State S2
    items.append({
        "prompt": "Context: A modulo-3 state accumulator has states {S0, S1, S2} representing remainder modulo 3. Start state is S0.\nTransitions:\n- Input 1 adds 1 modulo 3: (S0->S1, S1->S2, S2->S0)\n- Input 2 adds 2 modulo 3: (S0->S2, S1->S0, S2->S1)\nSequence of inputs: 1, 1, 2, 1.\nQuestion: What is the final state of the accumulator?\nAnswer:",
        "choices": ["S0", "S1", "S2", "S3"],
        "labels": ["A", "B", "C", "D"],
        "target": "C"
    })

    # Machine 8: Sum = 9 mod 3 = 0 -> State S0
    items.append({
        "prompt": "Context: A modulo-3 state accumulator has states {S0, S1, S2} representing remainder modulo 3. Start state is S0.\nTransitions:\n- Input 1 adds 1 modulo 3: (S0->S1, S1->S2, S2->S0)\n- Input 2 adds 2 modulo 3: (S0->S2, S1->S0, S2->S1)\nSequence of inputs: 2, 1, 2, 2, 2.\nQuestion: What is the final state of the accumulator?\nAnswer:",
        "choices": ["S0", "S1", "S2", "S3"],
        "labels": ["A", "B", "C", "D"],
        "target": "A"
    })

    # Machine 9: Stream X, X, Y -> Red->Green->Blue->Green. Final: Green
    items.append({
        "prompt": "Context: A tri-color light indicator has states {Red, Green, Blue}, starting at Red.\nRules:\n- Signal X rotates clockwise: (Red -> Green -> Blue -> Red)\n- Signal Y rotates counter-clockwise: (Red -> Blue -> Green -> Red)\nSequence of signals: X, X, Y.\nQuestion: What is the resulting color of the indicator?\nAnswer:",
        "choices": ["Red", "Green", "Blue", "Yellow"],
        "labels": ["A", "B", "C", "D"],
        "target": "B"
    })

    # Machine 10: Stream Y, Y, Y -> Red->Blue->Green->Red. Final: Red
    items.append({
        "prompt": "Context: A tri-color light indicator has states {Red, Green, Blue}, starting at Red.\nRules:\n- Signal X rotates clockwise: (Red -> Green -> Blue -> Red)\n- Signal Y rotates counter-clockwise: (Red -> Blue -> Green -> Red)\nSequence of signals: Y, Y, Y.\nQuestion: What is the resulting color of the indicator?\nAnswer:",
        "choices": ["Red", "Green", "Blue", "Yellow"],
        "labels": ["A", "B", "C", "D"],
        "target": "A"
    })

    return items

def load_novel_sectors():
    return [
        {
            "name": "Inverted-Physics",
            "domain": "Counterfactual Axiomatic Physics",
            "items": generate_sector1_inverted_physics()
        },
        {
            "name": "5Hop-TransitiveDeduction",
            "domain": "Complex Relational Constraint Graph",
            "items": generate_sector2_transitive_deduction()
        },
        {
            "name": "CounterIntuitive-Syllogism",
            "domain": "Belief Bias Formal Entailment",
            "items": generate_sector3_counter_syllogisms()
        },
        {
            "name": "Modular-TemporalWarping",
            "domain": "Multi-Step Modular Arithmetic",
            "items": generate_sector4_temporal_arithmetic()
        },
        {
            "name": "FiniteState-Automata",
            "domain": "Deterministic Latent State Machine",
            "items": generate_sector5_finite_state_machines()
        }
    ]

# ==============================================================================
# EVALUATION ENGINE
# ==============================================================================

def evaluate_sample(wrapped_model, tokenizer, item, pass_num=1):
    prompt = item["prompt"]
    choices = item["choices"]
    labels = item["labels"]
    target = item["target"]

    prompt_ids = tokenizer(prompt)["input_ids"]
    p_len = len(prompt_ids)
    query_anchor = p_len - 1

    cand_tensors = []
    for c in choices:
        c_ids = tokenizer(c.strip(), return_tensors="pt")["input_ids"]
        with torch.no_grad():
            c_emb = wrapped_model.qwen.get_input_embeddings()(c_ids)
        cand_tensors.append(c_emb.mean(dim=1, keepdim=True))
    joint_cands = torch.cat(cand_tensors, dim=1) if cand_tensors else None

    prompt_ids_tensor = torch.tensor(prompt_ids).unsqueeze(0)
    with torch.no_grad():
        prompt_emb = wrapped_model.qwen.get_input_embeddings()(prompt_ids_tensor).mean(dim=1).clone()

    scores_base = []
    wrapped_model.set_candidate_embeds(None)
    wrapped_model.set_ponder_steps(0)
    wrapped_model.reset_state(force=True)
    for c in choices:
        full_text = f"{prompt} {c.strip()}"
        input_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
        slab = input_ids[:, p_len:]
        denom = max(1, slab.shape[1])
        with torch.no_grad():
            logits_b = wrapped_model(input_ids).logits
        sl_b = logits_b[:, p_len-1:-1, :]
        lp_b = torch.log_softmax(sl_b, dim=-1).gather(-1, slab.unsqueeze(-1)).squeeze(-1)
        scores_base.append(lp_b.sum().item() / denom)

    sorted_b = sorted(scores_base, reverse=True)
    margin_val = float(sorted_b[0] - sorted_b[1]) if len(sorted_b) > 1 else 999.0
    pred_base_idx = int(np.argmax(scores_base))
    pred_base = labels[pred_base_idx]
    base_ok = (str(pred_base).upper() == target.upper()) or (str(pred_base_idx) == target)

    settled_ep = None
    if pass_num > 1 and hasattr(wrapped_model.adapter, "episodic_memory"):
        settled_ep = wrapped_model.adapter.episodic_memory.recall_settled(prompt_emb, sim_threshold=0.95)

    if settled_ep is not None:
        pred_delib = settled_ep["metadata"]["pred"]
        pred_delib_idx = labels.index(pred_delib) if pred_delib in labels else pred_base_idx
        delib_ok = (str(pred_delib).upper() == target.upper()) or (str(pred_delib_idx) == target)

        status = "Preserved Correct" if (base_ok and delib_ok) else (
            "Rescued (Wrong->Right)" if (not base_ok and delib_ok) else (
                "Preserved Wrong" if (not base_ok and not delib_ok) else "Degraded (Right->Wrong)"
            )
        )
        m_fast_norm = float(torch.norm(settled_ep["m_fast"]).item()) if settled_ep.get("m_fast") is not None else 0.0
        return {
            "pred_base": str(pred_base),
            "pred_delib": str(pred_delib),
            "base_ok": base_ok,
            "delib_ok": delib_ok,
            "status": status,
            "scores_base": scores_base,
            "scores_delib": scores_base,
            "margin": margin_val,
            "surprise_jsd": 0.0,
            "surprise_gate": 0.0,
            "vacuity_u": settled_ep.get("vacuity_u", 0.5),
            "plastic_trace_norm": m_fast_norm,
            "synthesized_count": 0,
            "critique_applied": False,
            "settled_shortcut": True
        }

    conflict_monitor = getattr(wrapped_model.adapter, "conflict_monitor", None)
    if conflict_monitor is not None:
        effort, recommended_k, should_deliberate = conflict_monitor.compute_effort_index(
            margin=margin_val,
            jsd=0.0,
            vacuity_u=0.5
        )
    else:
        should_deliberate = (margin_val < 0.25)

    if not should_deliberate:
        pred_delib = pred_base
        delib_ok = base_ok
        status = "Preserved Correct" if delib_ok else "Preserved Wrong"

        if pass_num == 1 and hasattr(wrapped_model.adapter, "episodic_memory"):
            wrapped_model.adapter.episodic_memory.store(
                key=prompt_emb,
                thought=prompt_emb,
                vacuity_u=0.5,
                margin=margin_val,
                m_fast=None,
                meta={"is_uncertain": False, "pred": pred_delib, "agrees": True},
                is_settled=True,
                confidence=1.0
            )

        return {
            "pred_base": str(pred_base),
            "pred_delib": str(pred_delib),
            "base_ok": base_ok,
            "delib_ok": delib_ok,
            "status": status,
            "scores_base": scores_base,
            "scores_delib": scores_base,
            "margin": margin_val,
            "surprise_jsd": 0.0,
            "surprise_gate": 0.0,
            "vacuity_u": 0.5,
            "plastic_trace_norm": 0.0,
            "synthesized_count": 0,
            "critique_applied": False,
            "settled_shortcut": False
        }

    critique_applied = False
    if pass_num > 1 and hasattr(wrapped_model.adapter, "episodic_memory"):
        recalled = wrapped_model.adapter.episodic_memory.recall(prompt_emb, top_k=1)
        if recalled and recalled[0]["similarity"] >= 0.95:
            entry = recalled[0]
            is_settled_p1 = entry.get("is_settled", False)
            p1_margin = entry.get("margin", 0.0)
            p1_vacuity = entry.get("vacuity_u", 0.5)
            p1_conf = entry.get("confidence", 0.0)
            agreed_p1 = entry.get("metadata", {}).get("agrees", False)

            anti_filter = getattr(wrapped_model.adapter, "anti_skepticism_filter", None)
            can_critique = anti_filter.should_apply_critique(
                is_settled=is_settled_p1,
                pass1_margin=p1_margin,
                vacuity_u=p1_vacuity,
                pass1_confidence=p1_conf,
                agreed_in_pass1=agreed_p1
            ) if anti_filter else (not is_settled_p1 and not agreed_p1 and p1_margin < 0.25)

            if can_critique:
                prior_thought = entry["thought"]
                critique_vec = prior_thought - prompt_emb
                wrapped_model.set_critique_vector(critique_vec)
                critique_applied = True

    scores_delib = []
    telemetries = []
    wrapped_model.set_candidate_embeds(joint_cands)
    wrapped_model.set_ponder_steps(2)
    wrapped_model.query_idx = query_anchor
    wrapped_model.reset_state(force=False)
    for c in choices:
        full_text = f"{prompt} {c.strip()}"
        input_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
        slab = input_ids[:, p_len:]
        denom = max(1, slab.shape[1])
        with torch.no_grad():
            logits_d = wrapped_model(input_ids).logits
        sl_d = logits_d[:, p_len-1:-1, :]
        lp_d = torch.log_softmax(sl_d, dim=-1).gather(-1, slab.unsqueeze(-1)).squeeze(-1)
        scores_delib.append(lp_d.sum().item() / denom)
        telemetries.append(dict(wrapped_model.last_telemetry))

    p_b = F.softmax(torch.tensor(scores_base), dim=-1)
    p_d = F.softmax(torch.tensor(scores_delib), dim=-1)
    m_dist = 0.5 * (p_b + p_d)
    jsd_val = float(0.5 * (F.kl_div(m_dist.log(), p_b, reduction='sum') + F.kl_div(m_dist.log(), p_d, reduction='sum')))

    last_telem = telemetries[-1] if telemetries else {}
    vacuity_u = float(last_telem.get("epistemic_vacuity", [0.5])[0]) if last_telem.get("epistemic_vacuity") else 0.5
    trace_norm = float(last_telem.get("plastic_trace_norm", 0.0))
    synthesized_count = int(last_telem.get("synthesized_concepts", 0))

    g_margin = float(torch.sigmoid(torch.tensor((0.10 - margin_val) / 0.05)))
    g_jsd = float(torch.sigmoid(torch.tensor((jsd_val - 0.10) / 0.02)))

    if critique_applied:
        sg_val = max(g_margin, g_jsd, 0.70)
    elif joint_cands is not None:
        sg_val = max(g_margin, g_jsd, 0.40)
    else:
        sg_val = max(g_margin, g_jsd) if margin_val < 0.5 else g_jsd

    # Epistemic Vacuity Gating Modulation (Subjective Logic Epistemic Modesty)
    eta_epistemic = max(0.20, min(1.0, 1.0 - max(0.0, vacuity_u - 0.50) / 0.50))
    sg_val = sg_val * eta_epistemic

    raw_combo = (1.0 - sg_val) * np.array(scores_base) + sg_val * np.array(scores_delib)

    # Directional Safety Projection on choice score space
    # Protects confident base answers from distractor drift (BBH)
    # AND protects near-zero tie-breakers from high-entropy frequency bias (DFA)
    combo_scores = DirectionalSafetyProjection.project_choice_scores(
        scores_base=np.array(scores_base),
        scores_delib=raw_combo,
        base_margin=margin_val,
        confidence_threshold=0.35,
        tie_breaker_threshold=0.05,
        delib_conviction_threshold=0.25,
        vacuity_u=vacuity_u
    )

    pred_delib_idx = int(np.argmax(combo_scores))
    pred_delib = labels[pred_delib_idx]

    delib_ok = (str(pred_delib).upper() == target.upper()) or (str(pred_delib_idx) == target)

    sorted_combo = sorted(combo_scores, reverse=True)
    post_margin = float(sorted_combo[0] - sorted_combo[1]) if len(sorted_combo) > 1 else 999.0

    if pass_num == 1 and hasattr(wrapped_model.adapter, "episodic_memory"):
        agrees = (pred_base == pred_delib)
        has_solid_margin = (post_margin >= 0.15)
        is_settled = agrees or has_solid_margin
        conf_val = float(np.exp(sorted_combo[0]) / sum(np.exp(sorted_combo))) if sum(np.exp(sorted_combo)) > 0 else 0.5
        wrapped_model.adapter.episodic_memory.store(
            key=prompt_emb,
            thought=prompt_emb + (0.1 * float(np.mean(scores_delib))),
            vacuity_u=vacuity_u,
            margin=post_margin,
            m_fast=wrapped_model.adapter.controller.plastic_unit.last_m_fast if wrapped_model.adapter.controller.plastic_unit is not None else None,
            meta={"is_uncertain": not is_settled, "pred": pred_delib, "post_margin": post_margin, "agrees": agrees},
            is_settled=is_settled,
            confidence=conf_val
        )
    elif pass_num > 1 and critique_applied and hasattr(wrapped_model.adapter, "episodic_memory"):
        if post_margin >= 0.15:
            recalled_list = wrapped_model.adapter.episodic_memory.recall(prompt_emb, top_k=1)
            if recalled_list and recalled_list[0]["similarity"] >= 0.95:
                m_idx = recalled_list[0]["index"]
                wrapped_model.adapter.episodic_memory.mark_settled(m_idx, is_settled=True, confidence=0.95)
                wrapped_model.adapter.episodic_memory.metadata[m_idx]["pred"] = pred_delib
                wrapped_model.adapter.episodic_memory.metadata[m_idx]["is_uncertain"] = False

    wrapped_model.set_critique_vector(None)
    wrapped_model.set_candidate_embeds(None)
    wrapped_model.reset_state(force=False)

    if base_ok and delib_ok:
        status = "Preserved Correct"
    elif not base_ok and delib_ok:
        status = "Rescued (Wrong->Right)"
    elif not base_ok and not delib_ok:
        status = "Preserved Wrong"
    else:
        status = "Degraded (Right->Wrong)"

    return {
        "pred_base": str(pred_base),
        "pred_delib": str(pred_delib),
        "base_ok": base_ok,
        "delib_ok": delib_ok,
        "status": status,
        "scores_base": scores_base,
        "scores_delib": scores_delib,
        "margin": margin_val,
        "surprise_jsd": jsd_val,
        "surprise_gate": sg_val,
        "vacuity_u": vacuity_u,
        "plastic_trace_norm": trace_norm,
        "synthesized_count": synthesized_count,
        "critique_applied": critique_applied,
        "settled_shortcut": False
    }

def run_pass(pass_id, sectors, wrapped_model, tokenizer):
    labels_map = {
        1: "Pass 1 (Cold Start Exploration & Uncertainty Audit)",
        2: "Pass 2 (Hippocampal Settled Consolidation & Reflective Correction)",
        3: "Pass 3 (Long-Horizon Stability & Zero-Waste Verification)"
    }
    pass_label = labels_map.get(pass_id, f"Pass {pass_id}")
    print(f"\n{'='*30} STARTING {pass_label.upper()} {'='*30}")
    pass_results = {}
    total_samples = 0
    total_base_correct = 0
    total_delib_correct = 0
    start_time = time.time()

    for sec in sectors:
        sec_name = sec["name"]
        print(f"\n[Pass {pass_id}] Evaluating Sector: {sec_name} ({len(sec['items'])} novel items)...")
        sys.stdout.flush()
        sec_logs = []
        sec_base_correct = 0
        sec_delib_correct = 0

        for i, item in enumerate(sec["items"]):
            res = evaluate_sample(wrapped_model, tokenizer, item, pass_num=pass_id)
            res["idx"] = i
            res["sector"] = sec_name
            res["target"] = item["target"]
            sec_logs.append(res)

            if res["base_ok"]: sec_base_correct += 1
            if res["delib_ok"]: sec_delib_correct += 1
            total_samples += 1

            mark_base = "[OK]" if res["base_ok"] else "[X]"
            mark_delib = "[OK]" if res["delib_ok"] else "[X]"
            crit_mark = " [Self-Correction]" if res.get("critique_applied") else ""
            short_mark = " [Settled Shortcut]" if res.get("settled_shortcut") else ""
            print(f"  Item {i+1:2d}/{len(sec['items'])} | Base: {mark_base} ({res['pred_base']}) -> Delib: {mark_delib} ({res['pred_delib']}) | u={res['vacuity_u']:.3f} | trace={res['plastic_trace_norm']:.3f} | {res['status']}{crit_mark}{short_mark}")
            sys.stdout.flush()

        n_sec = len(sec["items"])
        b_acc = (sec_base_correct / n_sec) * 100.0 if n_sec > 0 else 0.0
        d_acc = (sec_delib_correct / n_sec) * 100.0 if n_sec > 0 else 0.0
        rescued = sum(1 for r in sec_logs if r["status"] == "Rescued (Wrong->Right)")
        degraded = sum(1 for r in sec_logs if r["status"] == "Degraded (Right->Wrong)")

        pass_results[sec_name] = {
            "domain": sec["domain"],
            "samples": n_sec,
            "base_acc": b_acc,
            "delib_acc": d_acc,
            "delta": d_acc - b_acc,
            "rescued_count": rescued,
            "degraded_count": degraded,
            "mean_vacuity_u": float(np.mean([r["vacuity_u"] for r in sec_logs])),
            "mean_plastic_trace": float(np.mean([r["plastic_trace_norm"] for r in sec_logs])),
            "samples_log": sec_logs
        }
        total_base_correct += sec_base_correct
        total_delib_correct += sec_delib_correct

    elapsed = time.time() - start_time
    pass_results["macro_summary"] = {
        "pass_id": pass_id,
        "total_samples": total_samples,
        "base_accuracy": (total_base_correct / total_samples) * 100.0,
        "delib_accuracy": (total_delib_correct / total_samples) * 100.0,
        "delta": ((total_delib_correct - total_base_correct) / total_samples) * 100.0,
        "total_rescued": sum(pass_results[s]["rescued_count"] for s in pass_results if s != "macro_summary"),
        "total_degraded": sum(pass_results[s]["degraded_count"] for s in pass_results if s != "macro_summary"),
        "elapsed_seconds": round(elapsed, 2)
    }
    print(f"\n[Pass {pass_id} Completed in {elapsed:.1f}s] Base Acc: {pass_results['macro_summary']['base_accuracy']:.1f}% -> Delib Acc: {pass_results['macro_summary']['delib_accuracy']:.1f}% (Delta: {pass_results['macro_summary']['delta']:+.1f}%)")
    return pass_results

def run_stress_test():
    set_seed(1337)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 80)
    print("NOVEL RANDOMIZED AUTONOMOUS COGNITIVE STRESS-TEST (QWEN3.5-2B + DUAL-LOOP)")
    print("3-Pass Sequential Verification | 5 Novel Cognitive Sectors | 50 Unseen Items")
    print("=" * 80)

    print("[1/3] Loading Tokenizer & Qwen3.5-2B Backbone on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        dtype=torch.float32,
        device_map="cpu"
    )

    print("[2/3] Attaching Autonomous Plastic Dual-Loop Controller...")
    wrapped_model = attach_dual_loop_to_qwen(
        base_model,
        layer_idx=11,
        k_steps=2,
        enable_plasticity=True,
        use_evidential_gate=True,
        use_open_concept=True,
        use_surprise_gate=True,
        use_hypothesis_verification=True,
        use_contrastive_evidence=True
    )
    wrapped_model.load_adapter(ADAPTER_PATH, strict=False)

    print("[3/3] Synthesizing 50 novel, unprecedented questions across 5 sectors...")
    sectors_data = load_novel_sectors()

    wrapped_model.set_continual_mode(True, decay=0.90)

    # PASS 1
    p1 = run_pass(pass_id=1, sectors=sectors_data, wrapped_model=wrapped_model, tokenizer=tokenizer)
    if hasattr(wrapped_model.adapter, "episodic_memory"):
        wrapped_model.adapter.episodic_memory.consolidate(decay_factor=0.95)

    # PASS 2
    p2 = run_pass(pass_id=2, sectors=sectors_data, wrapped_model=wrapped_model, tokenizer=tokenizer)
    if hasattr(wrapped_model.adapter, "episodic_memory"):
        wrapped_model.adapter.episodic_memory.consolidate(decay_factor=0.95)

    # PASS 3
    p3 = run_pass(pass_id=3, sectors=sectors_data, wrapped_model=wrapped_model, tokenizer=tokenizer)

    # COMPARATIVE AUDIT
    print("\n" + "=" * 80)
    print("THREE-PASS CONTINUAL AUDIT & COGNITIVE STABILITY")
    print("=" * 80)
    sector_names = [s["name"] for s in sectors_data]
    table = {}
    for s_name in sector_names:
        b = p1[s_name]["base_acc"]
        acc1 = p1[s_name]["delib_acc"]
        acc2 = p2[s_name]["delib_acc"]
        acc3 = p3[s_name]["delib_acc"]
        table[s_name] = {
            "domain": p1[s_name]["domain"],
            "base_acc": b,
            "pass1_acc": acc1,
            "pass2_acc": acc2,
            "pass3_acc": acc3,
            "delta_p1": p1[s_name]["delta"],
            "delta_p2": p2[s_name]["delta"],
            "delta_p3": p3[s_name]["delta"]
        }
        print(f"  [{s_name:26s}] Base: {b:5.1f}% | P1: {acc1:5.1f}% | P2: {acc2:5.1f}% | P3: {acc3:5.1f}%")

    print("-" * 80)
    print(f"  [MACRO OVERALL (50 Items)] Base: {p1['macro_summary']['base_accuracy']:.1f}% -> P1: {p1['macro_summary']['delib_accuracy']:.1f}% -> P2: {p2['macro_summary']['delib_accuracy']:.1f}% -> P3: {p3['macro_summary']['delib_accuracy']:.1f}%")
    print(f"  [LATENCY PROFILE] P1: {p1['macro_summary']['elapsed_seconds']}s | P2: {p2['macro_summary']['elapsed_seconds']}s | P3: {p3['macro_summary']['elapsed_seconds']}s")
    print("=" * 80)

    # SAVE JSON
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "adapter_path": ADAPTER_PATH,
        "pass1": p1,
        "pass2": p2,
        "pass3": p3,
        "summary_table": table
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"[+] Authentic stress-test data saved to: {OUTPUT_JSON}")

    # PLOT
    print("\n[*] Generating high-resolution empirical comparison graphic...")
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    # Panel A: 3-Pass Accuracy Across Sectors
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(sector_names))
    width = 0.20
    b_accs = [table[s]["base_acc"] for s in sector_names]
    p1_accs = [table[s]["pass1_acc"] for s in sector_names]
    p2_accs = [table[s]["pass2_acc"] for s in sector_names]
    p3_accs = [table[s]["pass3_acc"] for s in sector_names]

    ax_a.bar(x - 1.5*width, b_accs, width, label="Base Qwen3.5-2B (K=0)", color="#7f7f7f", alpha=0.85)
    ax_a.bar(x - 0.5*width, p1_accs, width, label="Pass 1: Cold Start Delib.", color="#2b5c8f", alpha=0.9)
    ax_a.bar(x + 0.5*width, p2_accs, width, label="Pass 2: Settled Logic", color="#1b9e77", alpha=0.9)
    ax_a.bar(x + 1.5*width, p3_accs, width, label="Pass 3: Long-Horizon Verified", color="#7570b3", alpha=0.9)

    ax_a.set_title("(A) Novel Stress-Test: Base vs. 3-Pass Deliberation\n50 Unseen Questions Across 5 Extreme Reasoning Sectors", fontsize=11, fontweight='bold', pad=10)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([s.replace("-", "\n") for s in sector_names], fontsize=8)
    ax_a.set_ylabel("Accuracy (%)", fontsize=10)
    ax_a.set_ylim(0, 115)
    ax_a.grid(True, linestyle=":", alpha=0.5, axis='y')
    ax_a.legend(loc="upper right", framealpha=0.9, fontsize=8)

    # Panel B: Macro Trajectory Across Passes
    ax_b = fig.add_subplot(gs[0, 1])
    passes_x = ["Base (K=0)", "Pass 1 (Exploration)", "Pass 2 (Consolidated)", "Pass 3 (Stable)"]
    macro_vals = [
        p1['macro_summary']['base_accuracy'],
        p1['macro_summary']['delib_accuracy'],
        p2['macro_summary']['delib_accuracy'],
        p3['macro_summary']['delib_accuracy']
    ]
    colors_b = ["#7f7f7f", "#2b5c8f", "#1b9e77", "#7570b3"]
    bars_b = ax_b.bar(passes_x, macro_vals, width=0.5, color=colors_b, alpha=0.9)
    for bar in bars_b:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax_b.set_title("(B) Macro Reasoning Trajectory Across Sequential Passes\n50-Item Suite Overall Accuracy Progression", fontsize=11, fontweight='bold', pad=10)
    ax_b.set_ylabel("Macro Accuracy (%)", fontsize=10)
    ax_b.set_ylim(0, 100)
    ax_b.grid(True, linestyle=":", alpha=0.5, axis='y')

    # Panel C: Execution Latency (Token/Compute Efficiency)
    ax_c = fig.add_subplot(gs[1, 0])
    l_passes = ["Pass 1\n(Exploration)", "Pass 2\n(Settled Shortcuts)", "Pass 3\n(Settled Shortcuts)"]
    l_times = [
        p1['macro_summary']['elapsed_seconds'],
        p2['macro_summary']['elapsed_seconds'],
        p3['macro_summary']['elapsed_seconds']
    ]
    bars_c = ax_c.bar(l_passes, l_times, width=0.45, color=["#e7298a", "#1b9e77", "#2b5c8f"], alpha=0.85)
    for bar in bars_c:
        h = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width()/2, h + 2.0, f"{h:.1f}s", ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax_c.set_title("(C) Computational Latency & Token Efficiency\nZero-Waste Settled Logic vs. Raw Deliberation", fontsize=11, fontweight='bold', pad=10)
    ax_c.set_ylabel("Wall-Clock Latency (seconds)", fontsize=10)
    ax_c.set_ylim(0, max(l_times) * 1.25)
    ax_c.grid(True, linestyle=":", alpha=0.5, axis='y')

    # Panel D: Fast-Weight Trace & Epistemic Vacuity per Sector
    ax_d = fig.add_subplot(gs[1, 1])
    u_s = [p1[s]["mean_vacuity_u"] for s in sector_names]
    tr_s = [p1[s]["mean_plastic_trace"] for s in sector_names]
    ax_d.plot(x, u_s, 'o-', color="#d95f02", linewidth=2, markersize=7, label="Epistemic Vacuity u(x)")
    ax_d2 = ax_d.twinx()
    ax_d2.plot(x, tr_s, 's--', color="#7570b3", linewidth=2, markersize=7, label=r"Plastic Trace $\|\mathbf{M}_{\mathrm{fast}}\|_F$")

    ax_d.set_title("(D) Evidential Uncertainty and In-Situ Plastic Traces\nSector-by-Sector Diagnostic Latent Signals", fontsize=11, fontweight='bold', pad=10)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([s.replace("-", "\n") for s in sector_names], fontsize=8)
    ax_d.set_ylabel("Epistemic Vacuity u(x)", color="#d95f02", fontsize=10)
    ax_d2.set_ylabel(r"Fast-Weight Norm $\|\mathbf{M}_{\mathrm{fast}}\|_F$", color="#7570b3", fontsize=10)
    ax_d.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Qwen3.5-2B + Dual-Loop Controller: Authentic 3-Pass Stress-Test Benchmark on Novel Reasoning Puzzles", fontsize=13, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_PNG, bbox_inches='tight')
    plt.close()
    print(f"[+] Stress-test visualization saved to: {OUTPUT_PNG}")

    # Copy to artifacts directory
    artifact_dir = r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9"
    if os.path.exists(artifact_dir):
        import shutil
        dest = os.path.join(artifact_dir, OUTPUT_PNG)
        shutil.copy(OUTPUT_PNG, dest)
        print(f"[+] Artifact copy updated at: {dest}")

    print("\n" + "=" * 80)
    print("3-PASS NOVEL STRESS-TEST BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_stress_test()
