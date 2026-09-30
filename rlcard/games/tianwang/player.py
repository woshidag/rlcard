# -*- coding: utf-8 -*-
'''Implement Tianwang Player class (RLCard 风格玩家)'''
from rlcard.games.tianwang.core import ACTION_SPACE, Combo


class TianwangPlayer:
    '''Player stores hand, bid, role and performs actions.

    Notes:
        1. bid: 叫分 (40..105), None 表示未叫/派司
        2. is_dealer: 本局是否做庄 (叫分最低者)
        3. _current_hand: 出牌后剩余手牌
        4. captured / trick_score: 吃到的牌与抓分
    '''

    def __init__(self, player_id, np_random):
        '''Initialize a player

        Args:
            player_id (int): the player_id of a player
            np_random: numpy random state
        '''
        self.np_random = np_random
        self.player_id = player_id
        self.initial_hand = []
        self._current_hand = []
        self.bid = None
        self.is_dealer = False
        self.played_cards = []                  # 本局打出过的所有 Combo
        self.captured = []                      # 吃到的墩牌(Card列表, 含分数)
        self.trick_score = 0                    # 抓分合计
        self._recorded_played_cards = []        # play/play_back 记录

    @property
    def current_hand(self):
        return self._current_hand

    def set_current_hand(self, value):
        self._current_hand = value[:]

    def available_actions(self, state):
        '''Get legal action ids according to the follow rules

        Args:
            state (TianwangState): current game state

        Returns:
            list: list of integer action ids
        '''
        from rlcard.games.tianwang.core import get_legal_actions
        return get_legal_actions(state)

    def play(self, action_id):
        '''Perform an action: decode id -> remove cards from hand

        Args:
            action_id (int): the global action id

        Returns:
            Combo: the combo played
        '''
        combo = ACTION_SPACE.decode(action_id)
        assert self._hand_contains(combo), \
            f"P{self.player_id} cannot play {combo}, not in hand"
        for card in combo.cards:
            self._current_hand.remove(card)
        self.played_cards.append(combo)
        self._recorded_played_cards.append(combo)
        return combo

    def play_back(self):
        '''Restore the last played combo (for step_back)'''
        combo = self._recorded_played_cards.pop()
        self._current_hand.extend(combo.cards)
        self._current_hand.sort(key=lambda c: c.face)
        self.played_cards.pop()

    def _hand_contains(self, combo):
        hand = self._current_hand[:]
        for card in combo.cards:
            if card in hand:
                hand.remove(card)
            else:
                return False
        return True

    def win_trick(self, cards):
        '''Collect the cards of a won trick and its score'''
        from rlcard.games.tianwang.core import card_score
        self.captured.extend(cards)
        self.trick_score += sum(card_score(c.rank) for c in cards)

    def get_state(self, public_info, actions):
        '''Build the raw per-player state dict (used by Game.get_state)'''
        return {
            'seen_cards': public_info['seen_cards'],
            'trump_suit': public_info['trump_suit'],
            'dealer': public_info['dealer'],
            'trace': public_info['trace'].copy(),
            'played_cards': public_info['played_cards'],
            'kitty_size': public_info['kitty_size'],
            'self': self.player_id,
            'current_hand': self._current_hand[:],
            'num_cards_left': [n[0] for n in public_info['num_cards_left']],
            'scores': [n[1] for n in public_info['num_cards_left']],
            'actions': actions,
        }

    def __str__(self):
        return f'P{self.player_id}({"庄" if self.is_dealer else "闲"})'
