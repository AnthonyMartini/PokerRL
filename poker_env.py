import numpy as np
from treys import Card, Deck, Evaluator

class PokerEnv:
    def __init__(self, num_players=6, starting_stack=1000, small_blind=10, big_blind=20):
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.evaluator = Evaluator()
        self.reset()
        
    def reset(self):
        self.deck = Deck()
        self.board = []
        self.hands = [self.deck.draw(2) for _ in range(self.num_players)]
        self.stacks = [self.starting_stack] * self.num_players
        self.active = [True] * self.num_players
        self.pot = 0
        self.button = 0 # Player 0 is dealer
        
        self.bets = [0] * self.num_players
        self.total_invested = [0] * self.num_players
        self.acted = [False] * self.num_players
        
        self._post_blinds()
        
        # SB is button + 1, BB is button + 2. Under the Gun is button + 3
        self.current_player = (self.button + 3) % self.num_players 
        
        self.stage = 0 # 0: Preflop, 1: Flop, 2: Turn, 3: River
        self.street_finished = False
        self.hand_over = False
        self.winners = []
        self.reward = [0.0] * self.num_players
        
        return self.current_player, self._get_state(self.current_player)

    def _post_blinds(self):
        sb_idx = (self.button + 1) % self.num_players
        bb_idx = (self.button + 2) % self.num_players
        
        sb = min(self.small_blind, self.stacks[sb_idx])
        self.stacks[sb_idx] -= sb
        self.bets[sb_idx] += sb
        self.total_invested[sb_idx] += sb
        self.pot += sb
        
        bb = min(self.big_blind, self.stacks[bb_idx])
        self.stacks[bb_idx] -= bb
        self.bets[bb_idx] += bb
        self.total_invested[bb_idx] += bb
        self.pot += bb

    def _get_vector(self, cards):
        vec = np.zeros(52, dtype=np.float32)
        if type(cards) is int:
            cards = [cards]
            
        for c in cards:
            rank = Card.get_rank_int(c)
            suit = Card.get_suit_int(c)
            # suit is 1, 2, 4, 8 -> map to 0, 1, 2, 3
            suit_map = {1:0, 2:1, 4:2, 8:3}
            idx = rank * 4 + suit_map.get(suit, 0)
            vec[idx] = 1.0
        return vec

    def _get_state(self, player_idx):
        hand_vec = self._get_vector(self.hands[player_idx])
        board_vec = self._get_vector(self.board)
        
        # 1. Existing Chip / Player Stats
        normalization_factor = self.starting_stack * self.num_players
        stack = self.stacks[player_idx] / normalization_factor
        highest_bet = max(self.bets)
        to_call_raw = max(0, highest_bet - self.bets[player_idx])
        to_call = to_call_raw / normalization_factor
        pot_size = self.pot / normalization_factor
        active_opponents = (sum(self.active) - 1) / max(1, self.num_players - 1)
        
        # 2. NEW: Game Stage (One-hot encoded)
        stage_vec = np.zeros(4, dtype=np.float32)
        stage_vec[min(self.stage, 3)] = 1.0 # 0: Preflop, 1: Flop, 2: Turn, 3: River
        
        # 3. NEW: Position Relative to Button (One-hot encoded)
        pos_vec = np.zeros(self.num_players, dtype=np.float32)
        dist_to_button = (player_idx - self.button) % self.num_players
        pos_vec[dist_to_button] = 1.0
        
        # 4. NEW: Pot Odds
        # to_call / (pot + to_call). Added 1e-9 to prevent division by zero.
        pot_odds = to_call_raw / (self.pot + to_call_raw + 1e-9)
        
        # Combine everything into a single 118-dimension array
        state = np.concatenate([
            hand_vec,           # 52 dims
            board_vec,          # 52 dims
            stage_vec,          # 4 dims
            pos_vec,            # 5 dims
            [stack, to_call, pot_size, active_opponents, pot_odds] # 5 dims
        ])
        
        return state

    def get_valid_actions(self, player_idx):
        highest_bet = max(self.bets)
        to_call = max(0, highest_bet - self.bets[player_idx])
        
        valid = [0, 1] # Fold, Call/Check
        
        if self.stacks[player_idx] > to_call:
            valid.append(2) # Raise
            
        return valid

    def _next_active_player(self, current):
        nxt = (current + 1) % self.num_players
        while not self.active[nxt] or self.stacks[nxt] == 0:
            nxt = (nxt + 1) % self.num_players
            if nxt == current:
                return -1 # No other active players with non-zero stack
        return nxt

    def _is_street_finished(self):
        highest_bet = max(self.bets)
        active_players_with_chips = 0
        for i in range(self.num_players):
            if self.active[i]:
                if self.stacks[i] > 0:
                    active_players_with_chips += 1
                if self.stacks[i] > 0 and (not self.acted[i] or self.bets[i] < highest_bet):
                    return False
        return True

    def step(self, action):
        if self.hand_over:
            raise ValueError("Hand is already over")
            
        p = self.current_player
        self.acted[p] = True
        
        highest_bet = max(self.bets)
        to_call = max(0, highest_bet - self.bets[p])
        
        if action == 0: # Fold
            self.active[p] = False
                
        elif action == 1: # Call / Check
            call_amount = min(to_call, self.stacks[p])
            self.stacks[p] -= call_amount
            self.bets[p] += call_amount
            self.total_invested[p] += call_amount
            self.pot += call_amount
            
        elif action == 2: # Raise
            call_amount = min(to_call, self.stacks[p])
            pot_after_call = self.pot + call_amount
            raise_amount = min(pot_after_call // 2, self.stacks[p] - call_amount)
            if raise_amount <= 0:
                raise_amount = min(self.big_blind, self.stacks[p] - call_amount)
                
            total_bet = call_amount + raise_amount
            self.stacks[p] -= total_bet
            self.bets[p] += total_bet
            self.total_invested[p] += total_bet
            self.pot += total_bet
            
            # Reset acted flags for others because of the raise
            for i in range(self.num_players):
                if i != p and self.active[i] and self.stacks[i] > 0:
                    self.acted[i] = False
                    
        # Check if hand ended by everyone folding except 1
        active_count = sum(self.active)
        if active_count == 1:
            self.hand_over = True
            for i in range(self.num_players):
                if self.active[i]:
                    self.winners = [i]
            self._compute_rewards()
            done = True
            return self.current_player, self._get_state(self.current_player), self.reward.copy(), done

        self.street_finished = self._is_street_finished()
            
        if self.street_finished:
            active_with_chips = sum(1 for i in range(self.num_players) if self.active[i] and self.stacks[i] > 0)
            
            if active_with_chips <= 1 and active_count > 1: # All in situation, not everyone folded
                while self.stage < 3:
                    self._next_stage()
                self.hand_over = True
                self._evaluate_showdown()
            else:
                self._next_stage()
                if self.stage > 3: # Showdown
                    self.hand_over = True
                    self._evaluate_showdown()
                else: # Next street setup
                    self.acted = [False] * self.num_players
                    self.bets = [0] * self.num_players
                    # First active player after button
                    nxt = self._next_active_player(self.button)
                    self.current_player = nxt if nxt != -1 else self.current_player
        else:
            nxt = self._next_active_player(self.current_player)
            if nxt != -1:
                self.current_player = nxt
                
        done = self.hand_over
        if done and len(self.winners) == 0:
            # edge case fallback
            self._compute_rewards()
            
        state = self._get_state(self.current_player) if not done else None
        
        return self.current_player, state, self.reward.copy(), done

    def _next_stage(self):
        self.stage += 1
        num_draw = 3 if self.stage == 1 else 1 if self.stage <= 3 else 0
        if num_draw > 0:
            cards = self.deck.draw(num_draw)
            if type(cards) is int:
                 self.board.append(cards)
            else:
                 self.board.extend(cards)

    def _evaluate_showdown(self):
        # We ensure river is dealt before calling this
        if len(self.board) < 5:
            pass
            
        best_score = float('inf')
        best_players = []
        
        for i in range(self.num_players):
            if self.active[i]:
                try:
                    score = self.evaluator.evaluate(self.board, self.hands[i])
                    if score < best_score:
                        best_score = score
                        best_players = [i]
                    elif score == best_score:
                        best_players.append(i)
                except Exception as e:
                    # In case of treys failure, just skip
                    pass
                    
        self.winners = best_players
        self._compute_rewards()
        
    def _compute_rewards(self):
        # Simple pot distribution
        num_winners = len(self.winners)
        if num_winners == 0:
            return
            
        win_amount = self.pot / num_winners
        
        for i in range(self.num_players):
            if i in self.winners:
                self.reward[i] = (win_amount - self.total_invested[i]) / self.starting_stack
            else:
                self.reward[i] = -self.total_invested[i] / self.starting_stack
