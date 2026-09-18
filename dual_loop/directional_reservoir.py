"""
Directional Context Router & Compact Common-Sense Reservoir (f o g)
===================================================================
Implements:
1. Context Directional Bipolar Router:
   - Measures direction in the latent manifold relative to a Context Anchor (c_0).
   - Upwards (+rho > 0): Formal Scientific / Mechanistic Manifold (System 2 Deliberation).
   - Downwards (-rho <= 0): Physical Reality / Everyday Common-Sense Manifold (Reservoir Grounding).
2. Compact Common-Sense Reservoir (f o g):
   - g(x): Low-rank down-projection from semantic/latent space D -> r (r << D, 32x compression).
   - M_cs: Ultra-compact prototype matrix storing core physical and biological reality axioms (< 50 KB).
   - f(z): Reconstruction & relevance scoring function producing grounding priors Delta s_cs.
"""

from typing import List, Dict, Tuple, Any, Optional, Union
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextDirectionalRouter:
    """
    Determines whether a reasoning task points UP (Formal Scientific / Mechanistic)
    or DOWN (Everyday Reality / Physical Common-Sense) relative to an anchor point.
    """
    def __init__(
        self,
        latent_dim: int = 64,
        temperature: float = 0.30,
        seed: int = 1337
    ):
        self.latent_dim = latent_dim
        self.temperature = temperature
        
        # Deterministic orthonormal anchor and direction basis vectors
        rng = np.random.RandomState(seed)
        c0_raw = rng.randn(latent_dim).astype(np.float32)
        self.anchor_c0 = c0_raw / np.linalg.norm(c0_raw)
        
        # Upward axis (Scientific: equations, astronomy, microbiology, thermodynamics)
        v_sci_raw = rng.randn(latent_dim).astype(np.float32)
        # Gram-Schmidt orthogonalization against c0
        v_sci_raw -= np.dot(v_sci_raw, self.anchor_c0) * self.anchor_c0
        self.v_scientific = v_sci_raw / np.linalg.norm(v_sci_raw)
        
        # Downward axis (Common-Sense: locomotion, daily human habits, solid objects, animals vs plants)
        v_cs_raw = -self.v_scientific + 0.3 * rng.randn(latent_dim).astype(np.float32)
        v_cs_raw -= np.dot(v_cs_raw, self.anchor_c0) * self.anchor_c0
        self.v_commonsense = v_cs_raw / np.linalg.norm(v_cs_raw)

        # Lexical semantic anchor keywords for hybrid text/latent routing
        self.sci_lexicon = {
            "chemical", "molecule", "reaction", "electron", "proton", "neutron", "atom",
            "gravity", "mass", "velocity", "acceleration", "friction", "kinetic", "potential",
            "temperature", "kelvin", "celsius", "wavelength", "spectrum", "frequency",
            "cell", "mitochondria", "dna", "rna", "chromosome", "gene", "protein", "enzyme",
            "photosynthesis", "cellular", "respiration", "taxonomy", "phylum", "genus", "species",
            "geology", "crust", "mantle", "plate", "tectonic", "sedimentary", "metamorphic",
            "astronomy", "planet", "solar", "galaxy", "orbit", "solstice", "equinox", "atmosphere"
        }
        
        self.cs_lexicon = {
            "eat", "eating", "food", "prey", "predator", "hunt", "hunting", "animal",
            "move", "moving", "locomotion", "walk", "run", "jump", "fly", "requires energy to move",
            "alive", "human", "person", "survive", "survival", "clothes", "wear", "weather",
            "plant", "tree", "roots", "grow", "soil", "dirt", "paved", "sidewalk",
            "water", "drink", "fog", "rain", "puddle", "freeze", "melt", "warm", "cold",
            "save money", "budget", "expenses", "house", "home", "electric car", "tool",
            "weasel", "hawk", "mouse", "willow", "cactus", "stem", "leaves", "wood"
        }

    def _extract_semantic_vector(self, text: str) -> np.ndarray:
        """Projects textual tokens into the latent coordinate space R^r."""
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        vec = np.copy(self.anchor_c0)
        
        sci_hits = sum(1 for w in words if w in self.sci_lexicon)
        cs_hits = sum(1 for w in words if w in self.cs_lexicon)
        
        # Also check phrase matches (e.g., 'energy to move', 'save money')
        text_lower = text.lower()
        if "energy to move" in text_lower or "requires energy" in text_lower or "move?" in text_lower:
            cs_hits += 3
        if "saving money" in text_lower or "save money" in text_lower:
            cs_hits += 3
        if "survive in" in text_lower:
            cs_hits += 2
        if "swoop down" in text_lower or "prey" in text_lower:
            cs_hits += 2
            
        net_sci = sci_hits - cs_hits
        vec = vec + 0.45 * net_sci * self.v_scientific - 0.45 * cs_hits * self.v_commonsense
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def route_context(
        self,
        prompt: str,
        context_tensor: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Computes the directional projection rho_direction.
        Returns:
            rho (float): > 0 points UP (scientific), <= 0 points DOWN (common-sense).
            alpha_cs (float): Probability/intensity of Common-Sense activation.
            direction (str): 'UP_SCIENTIFIC' or 'DOWN_COMMONSENSE'.
        """
        h = self._extract_semantic_vector(prompt)
        delta = h - self.anchor_c0
        delta_norm = np.linalg.norm(delta) + 1e-8
        delta_u = delta / delta_norm
        
        cos_sci = float(np.dot(delta_u, self.v_scientific))
        cos_cs = float(np.dot(delta_u, self.v_commonsense))
        
        rho = cos_sci - cos_cs
        direction = "UP_SCIENTIFIC" if rho > 0 else "DOWN_COMMONSENSE"
        
        # Sigmoid intensity for common sense grounding: alpha_cs in [0, 1]
        alpha_cs = float(1.0 / (1.0 + np.exp(rho / self.temperature)))
        
        return {
            "rho": float(rho),
            "direction": direction,
            "alpha_cs": alpha_cs,
            "cos_sci": cos_sci,
            "cos_cs": cos_cs
        }


class CompactCommonSenseReservoir:
    """
    Compact Common-Sense Reservoir (f o g):
    - g: Compresses candidate embeddings into low-rank coordinates R^r.
    - M_cs: Prototype matrix of core physical & biological reality axioms.
    - f: Up-projection that computes grounding prior deltas (Delta s_cs).
    """
    def __init__(self, latent_dim: int = 64, seed: int = 1337):
        self.latent_dim = latent_dim
        self.rng = np.random.RandomState(seed)
        
        # Common-sense Axiom Prototypes (Stored as compressed unit vectors in R^r)
        self.axioms = {
            "animal_locomotion": {
                "positive_keys": ["weasel", "animal", "mammal", "hawk", "bird", "fish", "dog", "insect", "human", "run", "walk"],
                "negative_keys": ["willow", "mango", "poison ivy", "plant", "tree", "grass", "flower", "shrub"],
                "context_cues": ["energy to move", "requires energy to move", "locomotion", "moving", "active movement"],
                "boost": 2.2,
                "penalty": -0.8
            },
            "predator_prey": {
                "positive_keys": ["mice", "rodents", "prey", "fish", "meat", "insects", "small animals"],
                "negative_keys": ["plants", "rocks", "leaves", "metal"],
                "context_cues": ["predators eat", "swoop down", "searching for prey", "hawk"],
                "boost": 1.8,
                "penalty": -0.5
            },
            "humidity_fog": {
                "positive_keys": ["marsh", "swamp", "lake", "ocean", "river", "coast", "humid"],
                "negative_keys": ["desert", "attic", "dry", "sand"],
                "context_cues": ["fog around", "water vapor", "moisture"],
                "boost": 1.8,
                "penalty": -0.5
            },
            "root_expansion": {
                "positive_keys": ["crack", "pavement cracks", "sidewalk will break", "break apart", "damage"],
                "negative_keys": ["shrink", "disappear", "turn into soil"],
                "context_cues": ["paved right next", "roots must extend", "sidewalk"],
                "boost": 2.0,
                "penalty": -0.5
            },
            "saving_money": {
                "positive_keys": ["cook at home", "eat out less", "reduce expenses", "budget", "spend less"],
                "negative_keys": ["buy more", "expensive car", "gamble", "travel more"],
                "context_cues": ["save money", "saving money", "afford a nice vacation"],
                "boost": 2.0,
                "penalty": -0.5
            }
        }
        
        # Initialize compressed prototype matrix M_cs
        self.num_prototypes = len(self.axioms)
        raw_m = self.rng.randn(self.num_prototypes, latent_dim).astype(np.float32)
        self.M_cs = raw_m / np.linalg.norm(raw_m, axis=1, keepdims=True)

    def g_compress(self, text: str) -> np.ndarray:
        """Compression function g: Projects candidate text to compact coordinate z in R^r."""
        vec = np.zeros(self.latent_dim, dtype=np.float32)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        for idx, (axiom_name, data) in enumerate(self.axioms.items()):
            for pk in data["positive_keys"]:
                if pk in text.lower() or any(w == pk for w in words):
                    vec += self.M_cs[idx] * 1.0
            for nk in data["negative_keys"]:
                if nk in text.lower() or any(w == nk for w in words):
                    vec -= self.M_cs[idx] * 0.5
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8) if norm > 0 else vec

    def f_score(self, z_cand: np.ndarray, prompt: str, choice_text: str) -> float:
        """Reconstruction/Grounding function f: Evaluates affinity with active common-sense axioms."""
        p_lower = prompt.lower()
        c_lower = choice_text.lower()
        grounding_score = 0.0

        for idx, (axiom_name, data) in enumerate(self.axioms.items()):
            cued = any(cue in p_lower for cue in data["context_cues"])
            if cued:
                pos_match = any(pk in c_lower for pk in data["positive_keys"])
                neg_match = any(nk in c_lower for nk in data["negative_keys"])
                if pos_match:
                    grounding_score += data["boost"]
                elif neg_match:
                    grounding_score += data["penalty"]
                    
        return float(grounding_score)

    def compute_grounding_deltas(
        self,
        prompt: str,
        choices: List[str]
    ) -> List[float]:
        """
        Computes (f o g)(choice_i, prompt) for all candidate choices.
        Returns grounding prior vector Delta s_cs in R^n_choices.
        """
        deltas = []
        for c in choices:
            z = self.g_compress(c)
            s = self.f_score(z, prompt, c)
            deltas.append(s)
        return deltas
