import torch

class Memory():
    def __init__(self, num_envs):
        self.num_envs = num_envs

        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.rewards_int = []
        self.rewards_int_norm = []
        self.dones = []
        self.logits = []
        self.values = []
        self.values_int = []

    def save(self, state, action, reward, reward_int, reward_int_norm, next_state, done, logit, value, value_int):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.rewards_int.append(reward_int)
        self.rewards_int_norm.append(reward_int_norm)
        self.dones.append(done)
        self.logits.append(logit)
        self.values.append(value)
        self.values_int.append(value_int)

    def reset(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.rewards_int = []
        self.rewards_int_norm = []
        self.dones = []
        self.logits = []
        self.values = []
        self.values_int = []

    def get_data(self):
        return self.states, self.actions, self.next_states, self.rewards, self.rewards_int, self.rewards_int_norm, self.dones, self.logits, self.values, self.values_int
    
class Embedding_Replay_Buffer():
    def __init__(self, capacity, emb_dim, device):
        self.capacity = capacity
        self.device = device
        self.emb_dim = emb_dim

        self.size = 0
        self.idx = 0

        self.embeddings = torch.rand(capacity, emb_dim).to(self.device)

    def push(self, emb):
        i = self.idx
        self.embeddings[i] = emb
        self.idx  = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def pushs(self, embs):
        n = embs.shape[0]
        space = self.capacity - self.idx

        if n <= space:
            self.embeddings[self.idx:self.idx + n] = embs
        else:
            self.embeddings[self.idx:] = embs[:space]
            self.embeddings[:n - space] = embs[space:]

        self.idx = (self.idx + n) % self.capacity
        self.size = min(self.size + n, self.capacity)