import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.num_tiles = np.ceil((self.state_high - self.state_low) / self.tile_width).astype(int) + 1
        self.num_tiles_per_tiling = np.prod(self.num_tiles).astype(int)

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return int(self.num_actions * self.num_tilings * self.num_tiles_per_tiling)

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        if done:
            return np.zeros(self.feature_vector_len())
        
        feature_vector = np.zeros(self.feature_vector_len())
        
        for tiling in range(self.num_tilings):
            offset = self.state_low - tiling * self.tile_width / self.num_tilings
            tile_indices = ((s - offset) / self.tile_width).astype(int)
            state_index = np.ravel_multi_index(
                (a, tiling, tile_indices[0], tile_indices[1]),
                (self.num_actions, self.num_tilings, self.num_tiles[0], self.num_tiles[1]),
            )
            feature_vector[state_index] = 1
            
        return feature_vector
    
        

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)
        
    w = np.zeros((X.feature_vector_len()))

    for episode in range(num_episode):
        s = env.reset()
        done = False
        a = epsilon_greedy_policy(s,done,w)
        x = X(s, done, a)
        z = np.zeros_like(w)
        Q_old = 0
        while not done:
            s_prime, r, done, _ = env.step(a)
            a_prime = epsilon_greedy_policy(s_prime, done, w)
            x_prime = X(s_prime, done, a_prime)
            Q = np.dot(w, x)
            Q_prime = np.dot(w, x_prime)
            delta = r + gamma * Q_prime - Q
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q_prime
            x = x_prime
            a = a_prime
            
    return w