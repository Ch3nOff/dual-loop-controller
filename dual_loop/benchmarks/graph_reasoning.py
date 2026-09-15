import random
import torch
from typing import List, Tuple

class MultiHopGraphDataset:
    """
    Synthetic Multi-Hop Graph Reasoning Benchmark.
    Generates directed pointer chains with random distractor edges.
    """
    def __init__(self, num_samples: int = 2000, num_nodes: int = 20, num_edges: int = 8, hops: int = 3):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.hops = hops
        
        self.ARROW = num_nodes
        self.SEP = num_nodes + 1
        self.QUERY = num_nodes + 2
        self.vocab_size = num_nodes + 3
        self.data = self._generate_data()

    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        samples = []
        for _ in range(self.num_samples):
            nodes = list(range(self.num_nodes))
            random.shuffle(nodes)
            
            chain = nodes[:self.hops + 1]
            edges = []
            for i in range(len(chain) - 1):
                edges.append((chain[i], chain[i+1]))
            
            while len(edges) < self.num_edges:
                u = random.choice(nodes)
                v = random.choice([n for n in nodes if n != u])
                if (u, v) not in edges:
                    edges.append((u, v))
            
            random.shuffle(edges)
            
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

    def get_batch(self, batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = random.sample(range(self.num_samples), batch_size)
        seqs = [self.data[i][0] for i in indices]
        targets = torch.stack([self.data[i][1] for i in indices])
        inputs = torch.stack(seqs)
        return inputs, targets
