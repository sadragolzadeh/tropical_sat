import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pysat.formula import CNF
from pysat.solvers import Glucose3
from torch.utils.data import DataLoader, Dataset


# ═══════════════════════════════════════════════════════════════
# ── Checkpoint saving (falls back if a folder is not writable) ──
# ═══════════════════════════════════════════════════════════════

def safe_save(state_dict, filename='best_tropical_sat.pt'):
    candidates = [
        Path.home() / 'tropical_sat_checkpoints',
        Path(tempfile.gettempdir()),
    ]
    for folder in candidates:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / filename
            torch.save(state_dict, path)
            print(f"Saved checkpoint to {path}")
            return path
        except (RuntimeError, OSError) as e:
            print(f"Could not save to {folder}: {e}")
    print("WARNING: checkpoint could not be saved anywhere")
    return None


# ═══════════════════════════════════════════════════════════════
# ── Tropical building blocks (fully tropical, nothing replaced) ──
# ═══════════════════════════════════════════════════════════════

BIG = 1e9


def flip_literals(x):
    """Swap each literal's row with its negation's row.
    Relies on the ordering [x0, ¬x0, x1, ¬x1, ...]."""
    d = x.shape[-1]
    return x.reshape(-1, 2, d).flip(1).reshape(-1, d)


def trop_reduce(x, dim, mode, hard, beta, valid=None):
    """Tropical reduction along `dim`: max-plus ('max') or min-plus ('min').
    Soft version is a log-sum-exp relaxation; hard is an exact max/min."""
    sign = 1.0 if mode == 'max' else -1.0
    if valid is not None:
        x = x.masked_fill(~valid, -sign * BIG)
    if hard:
        return x.max(dim=dim).values if mode == 'max' else x.min(dim=dim).values
    return torch.logsumexp(sign * beta * x, dim=dim) / (sign * beta)


def trop_linear(x, W, mode, hard, beta):
    """Tropical matrix-vector product: out[i,o] = (+)_k (x[i,k] + W[o,k])."""
    return trop_reduce(x.unsqueeze(1) + W.unsqueeze(0), dim=-1,
                       mode=mode, hard=hard, beta=beta)


class TropicalAttention(nn.Module):
    def __init__(self, d, mode='max'):
        super().__init__()
        self.mode = mode
        self.Wq = nn.Parameter(torch.randn(d, d) * 0.1)
        self.Wk = nn.Parameter(torch.randn(d, d) * 0.1)
        self.Wv = nn.Parameter(torch.randn(d, d) * 0.1)

    def forward(self, query, key_value, mask=None, hard=False, beta=1.0):
        """
        query: (n_q, d), key_value: (n_kv, d), mask: (n_q, n_kv) bool or None
        returns: (n_q, d)
        """
        m = self.mode
        Q = trop_linear(query, self.Wq, m, hard, beta)        # (n_q, d)
        K = trop_linear(key_value, self.Wk, m, hard, beta)    # (n_kv, d)
        V = trop_linear(key_value, self.Wv, m, hard, beta)    # (n_kv, d)

        # old:
        # s = trop_reduce(Q.unsqueeze(1) + K.unsqueeze(0), dim=-1, mode=m, hard=hard, beta=beta)

        # new: negative (max-plus) or positive (min-plus) tropical Hilbert distance
        diff = Q.unsqueeze(1) - K.unsqueeze(0)                                   # (n_q, n_kv, d)
        hi = trop_reduce(diff, dim=-1, mode='max', hard=hard, beta=beta)
        lo = trop_reduce(diff, dim=-1, mode='min', hard=hard, beta=beta)
        dist = hi - lo                                                            # >= 0
        s = -dist if m == 'max' else dist   

        # out[i,k] = (+)_j (s[i,j] + V[j,k]) over allowed j
        comb = s.unsqueeze(-1) + V.unsqueeze(0)               # (n_q, n_kv, d)
        valid = mask.unsqueeze(-1) if mask is not None else None
        out = trop_reduce(comb, dim=1, mode=m, hard=hard, beta=beta, valid=valid)

        if mask is not None:  # rows with no allowed neighbours -> 0
            out = torch.where(mask.any(dim=1, keepdim=True), out, torch.zeros_like(out))
        return out


class TropicalAggregation(nn.Module):
    """Tropical projection followed by masked tropical aggregation over neighbours.
    src: (n_src, d_in), adj: (n_dst, n_src) -> (n_dst, d_out)"""
    def __init__(self, d_in, d_out, mode='max'):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_out, d_in) * 0.1)
        self.mode = mode

    def forward(self, src, adj, hard=False, beta=1.0):
        P = trop_linear(src, self.W, self.mode, hard, beta)            # (n_src, d_out)
        scores = P.unsqueeze(0).expand(adj.shape[0], -1, -1)           # (n_dst, n_src, d_out)
        valid = (adj > 0).unsqueeze(-1)
        out = trop_reduce(scores, dim=1, mode=self.mode, hard=hard, beta=beta, valid=valid)
        return torch.where((adj > 0).any(dim=1, keepdim=True), out, torch.zeros_like(out))


# ═══════════════════════════════════════════════════════════════
# ── Main model ──
# ═══════════════════════════════════════════════════════════════

class TropicalSAT(nn.Module):
    def __init__(self, d=64, n_rounds=10, lit_init='var_random'):
        """
        lit_init: how literal vectors start
          'embedding'  - two learned vectors (positive / negative), the old behaviour
          'var_random' - one N(0,1) vector per variable shared by x and ¬x, plus the polarity embedding
          'lit_random' - an independent N(0,1) vector per literal
        """
        super().__init__()
        self.d = d
        self.n_rounds = n_rounds
        self.lit_init = lit_init

        self.lit_embedding = nn.Embedding(2, d)  # 0 = positive, 1 = negative

        # hidden -> real: tropical attention
        self.h2r_lit_attn = TropicalAttention(d, mode='max')      # literals: masked max-plus
        self.h2r_clause_attn = TropicalAttention(d, mode='min')   # clauses:  min-plus
        self.h2r_prep_attn = TropicalAttention(d, mode='max')     # prep:     max-plus

        # message passing between node types
        self.lit2clause_agg = TropicalAggregation(d, d, mode='max')    # literals -> clauses (OR)
        self.clause2lit_agg = TropicalAggregation(d, d, mode='max')    # clauses -> literals
        self.clause2prep_agg = TropicalAggregation(d, d, mode='min')   # clauses -> prep (AND)
        self.prep2clause_agg = TropicalAggregation(d, d, mode='min')   # prep -> clauses

        # own state + flipped (negation) state + clause message
        self.flip_proj = nn.Linear(3 * d, d)

        # LayerNorms for the real states
        self.norm_lit = nn.LayerNorm(d)
        self.norm_clause = nn.LayerNorm(d)
        self.norm_prep = nn.LayerNorm(d)

        self.h_clause_mlp = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d)
        )
        self.h_prep_mlp = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d)
        )

        self.vote_lit_mlp = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        self.vote_clause_mlp = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        self.vote_prep_mlp = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))

        self.W_lit = nn.Parameter(torch.ones(1))
        self.W_clause = nn.Parameter(torch.ones(1))
        self.W_prep = nn.Parameter(torch.ones(1))

    def _init_literals(self, lit_signs):
        n_lits = lit_signs.shape[0]
        device = lit_signs.device

        def randn(*shape):
            if self.training:
                return torch.randn(*shape, device=device)
            g = torch.Generator(device=device).manual_seed(0)   # same formula -> same vectors at eval
            return torch.randn(*shape, device=device, generator=g)

        if self.lit_init == 'embedding':
            return self.lit_embedding(lit_signs)
        if self.lit_init == 'lit_random':
            return randn(n_lits, self.d)
        # 'var_random': one random vector per variable, shared by x and ¬x, plus polarity embedding
        var_vec = randn(n_lits // 2, self.d).repeat_interleave(2, dim=0)
        return var_vec + self.lit_embedding(lit_signs)

    def forward(self, lit_signs, adj_lc, co_occurrence_mask, hard=False, beta=1.0,
                return_embeddings=False):
        """
        lit_signs: (n_lits,) long, 0 positive / 1 negative
        adj_lc: (n_clauses, n_lits) literal-to-clause adjacency
        co_occurrence_mask: (n_lits, n_lits) bool, True if literals share a clause
        """
        n_clauses = adj_lc.shape[0]
        device = lit_signs.device

        h_lit = self._init_literals(lit_signs)
        h_clause = torch.zeros(n_clauses, self.d, device=device)
        h_prep = torch.zeros(1, self.d, device=device)

        r_lit = torch.zeros_like(h_lit)
        r_clause = torch.zeros_like(h_clause)
        r_prep = torch.zeros_like(h_prep)

        adj_cp = torch.ones(1, n_clauses, device=device)   # clauses -> prep
        adj_pc = adj_cp.T                                  # (n_clauses, 1): prep -> every clause
        adj_cl = adj_lc.T                                  # (n_lits, n_clauses): clauses -> literals

        trace = []

        for t in range(self.n_rounds):
            # clauses from literals (OR: max-plus) + residual + message from prep
            h_clause_new = self.lit2clause_agg(h_lit, adj_lc, hard=hard, beta=beta)
            if t > 0:
                m_prep = self.prep2clause_agg(h_prep, adj_pc, hard=hard, beta=beta)
                h_clause_new = h_clause_new + h_clause + m_prep
            h_clause = self.h_clause_mlp(h_clause_new)

            # literals receive clause messages plus their negation's state
            m_lit = self.clause2lit_agg(h_clause, adj_cl, hard=hard, beta=beta)
            h_in = self.flip_proj(torch.cat([h_lit, flip_literals(h_lit), m_lit], dim=-1))
            a_lit = self.h2r_lit_attn(h_in, h_in, mask=co_occurrence_mask, hard=hard, beta=beta)
            r_lit = self.norm_lit(a_lit + h_in)

            # prep from clauses (AND: min-plus)
            h_prep_new = self.clause2prep_agg(h_clause, adj_cp, hard=hard, beta=beta)
            if t > 0:
                h_prep_new = h_prep_new + h_prep
            h_prep = self.h_prep_mlp(h_prep_new)

            # clause attention: self-attention in round 1, then query = previous real clause state
            q_clause = h_clause if t == 0 else r_clause
            a_clause = self.h2r_clause_attn(q_clause, h_clause, hard=hard, beta=beta)
            r_clause = self.norm_clause(a_clause + h_clause)

            a_prep = self.h2r_prep_attn(r_prep, h_prep, hard=hard, beta=beta)
            r_prep = self.norm_prep(a_prep + h_prep)

            if return_embeddings:
                trace.append({
                    'round': t,
                    'attn_lit': a_lit.detach().cpu(),       'r_lit': r_lit.detach().cpu(),
                    'attn_clause': a_clause.detach().cpu(), 'r_clause': r_clause.detach().cpu(),
                    'attn_prep': a_prep.detach().cpu(),     'r_prep': r_prep.detach().cpu(),
                })

            h_lit = r_lit

        votes_lit = self.vote_lit_mlp(r_lit).squeeze(-1)
        votes_clause = self.vote_clause_mlp(r_clause).squeeze(-1)
        votes_prep = self.vote_prep_mlp(r_prep).squeeze(-1)

        final_vote = (
            self.W_lit * votes_lit.mean() +
            self.W_clause * votes_clause.mean() +
            self.W_prep * votes_prep
        )

        logit = final_vote.squeeze()
        return (logit, trace) if return_embeddings else logit


# ═══════════════════════════════════════════════════════════════
# ── Dataset: SAT / UNSAT minimal pairs ──
# ═══════════════════════════════════════════════════════════════

def check_sat(clauses):
    cnf = CNF(from_clauses=clauses)
    with Glucose3(bootstrap_with=cnf) as solver:
        return solver.solve()


def clauses_to_tensors(clauses, n_vars):
    n_clauses = len(clauses)
    n_lits = 2 * n_vars

    lit_signs = torch.zeros(n_lits, dtype=torch.long)
    for i in range(n_vars):
        lit_signs[2 * i] = 0
        lit_signs[2 * i + 1] = 1

    adj_lc = torch.zeros(n_clauses, n_lits)
    for i, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            col = 2 * var if lit > 0 else 2 * var + 1
            adj_lc[i, col] = 1

    co_occurrence_mask = (adj_lc.T @ adj_lc) > 0
    return lit_signs, adj_lc, co_occurrence_mask


def random_clause(n_vars, rng):
    vars_ = rng.choice(n_vars, size=3, replace=False) + 1
    return [int(v) if rng.random() < 0.5 else -int(v) for v in vars_]


def generate_sat_unsat_pair(n_vars, rng, max_clauses=None):
    """Add random clauses until the formula first becomes UNSAT. Flipping one
    literal of that last clause gives a SAT twin. Returns (sat, unsat) or None."""
    max_clauses = max_clauses or 20 * n_vars
    clauses = []
    with Glucose3() as solver:
        while len(clauses) < max_clauses:
            clause = random_clause(n_vars, rng)
            solver.add_clause(clause)
            if not solver.solve():
                unsat = clauses + [clause]
                twin = list(clause)
                j = int(rng.integers(3))
                twin[j] = -twin[j]
                return clauses + [twin], unsat
            clauses.append(clause)
    return None


class SATPairDataset(Dataset):
    """n_pairs minimal pairs -> 2 * n_pairs instances, exactly 50/50 SAT/UNSAT."""
    def __init__(self, n_pairs, n_vars, seed, verbose=True):
        self.data = []
        rng = np.random.default_rng(seed)
        n_done = 0
        while n_done < n_pairs:
            pair = generate_sat_unsat_pair(n_vars, rng)
            if pair is None:
                continue
            sat, unsat = pair
            if not check_sat(sat) or check_sat(unsat):   # sanity check, should never trigger
                continue
            for clauses, label in ((sat, 1.0), (unsat, 0.0)):
                lit_signs, adj_lc, co_mask = clauses_to_tensors(clauses, n_vars)
                self.data.append({'lit_signs': lit_signs, 'adj_lc': adj_lc,
                                  'co_occurrence_mask': co_mask, 'label': label})
            n_done += 1
        rng.shuffle(self.data)
        if verbose:
            sizes = [d['adj_lc'].shape[0] for d in self.data]
            print(f"Dataset: {len(self.data)} instances (50/50), "
                  f"clauses per instance: min {min(sizes)}, mean {np.mean(sizes):.1f}, max {max(sizes)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ═══════════════════════════════════════════════════════════════
# ── Training / evaluation ──
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, optimizer, device, beta_scale, accum=4):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        lit_signs = batch['lit_signs'][0].to(device)
        adj_lc = batch['adj_lc'][0].to(device)
        co_mask = batch['co_occurrence_mask'][0].to(device)
        label = batch['label'].to(device).float()

        logit = model(lit_signs, adj_lc, co_mask, hard=False, beta=beta_scale)
        loss = F.binary_cross_entropy_with_logits(logit.unsqueeze(0), label)
        (loss / accum).backward()

        if (i + 1) % accum == 0 or (i + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        correct += ((logit > 0).float() == label).sum().item()
        total += 1

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, device, beta_scale=1.0):
    model.eval()
    correct, total = 0, 0

    for batch in dataloader:
        lit_signs = batch['lit_signs'][0].to(device)
        adj_lc = batch['adj_lc'][0].to(device)
        co_mask = batch['co_occurrence_mask'][0].to(device)
        label = batch['label'].to(device).float()

        logit = model(lit_signs, adj_lc, co_mask, hard=True, beta=beta_scale)
        correct += ((logit > 0).float() == label).sum().item()
        total += 1

    return correct / total


def print_embeddings(model, dataset, device, epoch, beta=1.0, idx=0, round_idx=-1,
                     max_rows=3, hard=True, full=False):
    """Print the tropical-attention outputs for one fixed instance."""
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=200,
                           threshold=100000 if full else 20, edgeitems=4)
    was_training = model.training
    model.eval()
    item = dataset[idx]
    with torch.no_grad():
        logit, trace = model(item['lit_signs'].to(device), item['adj_lc'].to(device),
                             item['co_occurrence_mask'].to(device),
                             hard=hard, beta=beta, return_embeddings=True)
    if was_training:
        model.train()

    rec = trace[round_idx]
    print(f"\n── Epoch {epoch} | instance {idx} (label {item['label']:.0f}, "
          f"logit {logit.item():+.3f}) | round {rec['round'] + 1}/{len(trace)} ──")
    for name in ('attn_lit', 'attn_clause', 'attn_prep'):
        x = rec[name]
        print(f"{name}: shape {tuple(x.shape)} | mean {x.mean().item():.3f} | std {x.std().item():.3f}")
        print(x[:max_rows])
    torch.set_printoptions(profile='default')


# ═══════════════════════════════════════════════════════════════
# ── Main ──
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    N_VARS = 6              # debugging size; raise to 10 once the model learns
    D = 64
    N_ROUNDS = 10
    N_EPOCHS = 100
    LR = 3e-4
    SEED = 42
    PAIRS_PER_EPOCH = 100   # fresh pairs every epoch -> 200 instances
    LIT_INIT = 'var_random' # 'embedding' | 'var_random' | 'lit_random'
    PRINT_EVERY = 1         # print attention embeddings every N epochs (0 = never)
    OVERFIT_TEST = False    # True: run the 16-instance memorisation check and exit

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if OVERFIT_TEST:
        small = SATPairDataset(n_pairs=8, n_vars=N_VARS, seed=SEED + 500)   # 16 instances
        small_loader = DataLoader(small, batch_size=1, shuffle=True)
        model = TropicalSAT(d=D, n_rounds=N_ROUNDS, lit_init=LIT_INIT).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        for ep in range(200):
            loss, acc = train_epoch(model, small_loader, opt, device, beta_scale=1.0, accum=4)
            if ep % 10 == 0:
                print(f"Overfit epoch {ep:3d} | loss {loss:.4f} | acc {acc:.3f}")
        raise SystemExit

    # validation and test sets are fixed; their seeds never overlap the training seeds
    val_dataset = SATPairDataset(n_pairs=50, n_vars=N_VARS, seed=SEED + 1)
    test_dataset = SATPairDataset(n_pairs=50, n_vars=N_VARS, seed=SEED + 2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = TropicalSAT(d=D, n_rounds=N_ROUNDS, lit_init=LIT_INIT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    best_val_acc = -1.0
    best_state = None
    beta = 1.0

    for epoch in range(N_EPOCHS):
        beta = 1.0 + 0.1 * epoch    # annealing schedule for the soft max/min

        # fresh training pairs every epoch (seeds 1000+ never collide with SEED+1 / SEED+2)
        train_dataset = SATPairDataset(n_pairs=PAIRS_PER_EPOCH, n_vars=N_VARS,
                                       seed=SEED + 1000 + epoch, verbose=False)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, beta)
        val_acc = evaluate(model, val_loader, device, beta_scale=beta)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch + 1:3d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} "
              f"| Val Acc: {val_acc:.3f} | Beta: {beta:.2f}")

        if PRINT_EVERY and (epoch + 1) % PRINT_EVERY == 0:
            print_embeddings(model, val_dataset, device, epoch + 1, beta=beta)

    # evaluate the best-on-validation checkpoint exactly once on the held-out test set
    model.load_state_dict(best_state)
    safe_save(model.state_dict())
    test_acc = evaluate(model, test_loader, device, beta_scale=beta)

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Final test accuracy (single evaluation): {test_acc:.3f}")
