import multiprocessing as mp
from poker_env import PokerEnv

def worker(remote, parent_remote):
    parent_remote.close()
    env = PokerEnv(num_players=6)
    
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                action = data
                current_player, state, rewards, done = env.step(action)
                if done:
                    # Send back the terminal state information first so train.py gets the reward
                    valid_actions = []
                    remote.send((current_player, state, valid_actions, rewards, done))
                else:
                    valid_actions = env.get_valid_actions(current_player)
                    remote.send((current_player, state, valid_actions, rewards, done))
            elif cmd == 'reset':
                current_player, state = env.reset()
                valid_actions = env.get_valid_actions(current_player)
                remote.send((current_player, state, valid_actions))
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
        except EOFError:
            break

class VectorizedPokerEnv:
    def __init__(self, num_envs=32):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = [mp.Process(target=worker, args=(work_remote, remote))
                          for work_remote, remote in zip(self.work_remotes, self.remotes)]
        
        for p in self.processes:
            p.daemon = True # Processes will close when parent closes
            p.start()
            
        for remote in self.work_remotes:
            remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        
        results = [remote.recv() for remote in self.remotes]
        return results

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
        results = [remote.recv() for remote in self.remotes]
        
        final_results = []
        for i, (current_player, state, valid_actions, rewards, done) in enumerate(results):
            if done:
                # Provide the final rewards for logging, but seamlessly reset the env to ready the next hand
                self.remotes[i].send(('reset', None))
                nxt_current_player, nxt_state, nxt_valid_actions = self.remotes[i].recv()
                final_results.append((nxt_current_player, nxt_state, nxt_valid_actions, rewards, done))
            else:
                final_results.append((current_player, state, valid_actions, rewards, done))
                
        return final_results

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
