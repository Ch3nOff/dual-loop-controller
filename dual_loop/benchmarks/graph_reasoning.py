import random
import torch
from typing import List, Tuple, Optional

class MultiHopGraphDataset:
    """
    Synthetic Multi-Hop Graph Reasoning Benchmark.
    Generates directed pointer chains with random distractor edges.
    Supports isolated deterministic seeding for 100% reproducible benchmarks.
    """
    def __init__(
        self,
        num_samples: int = 2000,
        num_nodes: int = 16,
        num_edges: int = 6,
        hops: int = 3,
        split: str = "test",
        seed: Optional[int] = None
    ):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.hops = hops
        self.split = split.lower()
        self.base_seed = seed

        max_possible_edges = num_nodes * (num_nodes - 1)
        if num_edges > max_possible_edges:
            raise ValueError(
                f"num_edges ({num_edges}) exceeds maximum possible directed edges "
                f"for {num_nodes} nodes ({max_possible_edges})."
            )
        if hops >= num_nodes:
            raise ValueError(
                f"hops ({hops}) must be strictly less than num_nodes ({num_nodes})."
            )

        # Disjoint seed offsets guarantee that test and validation sets
        # never collide with training data in random generation space.
        split_offsets = {"train": 0, "val": 10_000, "test": 50_000}
        offset = split_offsets.get(self.split, 50_000)

        if seed is not None:
            effective_seed = seed + offset
            self.rng = random.Random(effective_seed)
        else:
            self.rng = random.Random()
        
        self.ARROW = num_nodes
        self.SEP = num_nodes + 1
        self.QUERY = num_nodes + 2
        self.vocab_size = num_nodes + 3
        self.data = self._generate_data()

    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        samples = []
        for _ in range(self.num_samples):
            nodes = list(range(self.num_nodes))
            self.rng.shuffle(nodes)
            
            chain = nodes[:self.hops + 1]
            edges = []
            for i in range(len(chain) - 1):
                edges.append((chain[i], chain[i+1]))
            
            while len(edges) < self.num_edges:
                u = self.rng.choice(nodes)
                v = self.rng.choice([n for n in nodes if n != u])
                if (u, v) not in edges:
                    edges.append((u, v))
            
            self.rng.shuffle(edges)
            
            seq = []
            for u, v in edges:
                seq.extend([u, self.ARROW, v, self.SEP])
            
            query_node = chain[0]
            hop_targets = chain[1:self.hops + 1]
            
            seq.extend([self.QUERY, query_node, self.ARROW])
            samples.append((
                torch.tensor(seq, dtype=torch.long),
                torch.tensor(hop_targets, dtype=torch.long)
            ))
        return samples

    def get_batch(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not shuffle:
            indices = list(range(min(batch_size, self.num_samples)))
        elif seed is not None:
            batch_rng = random.Random(seed)
            indices = batch_rng.sample(range(self.num_samples), batch_size)
        else:
            indices = self.rng.sample(range(self.num_samples), batch_size)

        seqs = [self.data[i][0] for i in indices]
        targets = torch.stack([self.data[i][1] for i in indices])
        inputs = torch.stack(seqs)
        return inputs, targets
