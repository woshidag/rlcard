# -*- coding: utf-8 -*-
'''Implement Tianwang Game class (RLCard 风格 Game，供 Env 调用)

对外 API 与 DoudizhuGame 对齐: init_game / step / step_back / get_state /
get_num_actions / get_player_id / get_players / is_over / get_payoffs。
动作空间说明: 单张/对子为静态 ID [0,68)，"两张不等牌"组合为惰性注册的
动态 ID (>=68)。Env 侧以固定向量长度 + legal action mask 处理。
'''
import numpy as np

from rlcard.games.tianwang import Player
from rlcard.games.tianwang import Round
from rlcard.games.tianwang import Judger
from rlcard.games.tianwang.core import (
    ACTION_SPACE, STATIC_SIZE, NUM_PLAYERS,
)


class TianwangGame:
    '''Provide game APIs for env to run Tianwang and get state info'''

    def __init__(self, allow_step_back=False):
        '''Initialize the game

        Args:
            allow_step_back (bool): whether to allow step back
        '''
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = NUM_PLAYERS

    def init_game(self):
        '''Initialize players and state.

        Returns:
            dict: first state in one game
            int: current player's id
        '''
        self.winner_id = None
        self.payoffs = None
        self.history = []

        # initialize players
        self.players = [Player(num, self.np_random)
                        for num in range(self.num_players)]

        # initialize players (RLCard 约定: game 对象复用, 每局必须重置全部动态状态)
        for player in self.players:
            player.reset_game_state()

        # initialize round: shuffle & deal (14*4 + 12 kitty), trump, bury
        self.round = Round(self.np_random)
        self.round.dealer.deal_cards(self.players)
        self.round.initiate(self.players)

        # initialize judger
        self.judger = Judger(self.np_random)

        player_id = self.round.current_player
        self.state = self.get_state(player_id)
        return self.state, player_id

    def step(self, action):
        '''Perform one move.

        Args:
            action (int): legal action id of the current player

        Returns:
            dict: next player's state
            int: next player's id
        '''
        player = self.players[self.round.current_player]
        next_id, done = self.round.proceed_round(self.players, int(action))
        self.history.append((player.player_id, int(action)))
        if done:
            self.payoffs = self.judger.judge_game(self.players)
            dealer = next(p for p in self.players if p.is_dealer)
            self.winner_id = dealer.player_id if \
                self.payoffs[dealer.player_id] > 0 else \
                max(self.payoffs, key=self.payoffs.get)
            next_id = 0
            state = self.get_state(0)
        else:
            state = self.get_state(next_id)
        self.state = state
        return state, next_id

    def step_back(self):
        '''Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        '''
        if not self.round.trace or not self.allow_step_back:
            return False
        self.payoffs = None
        self.winner_id = None
        pid = self.round.step_back(self.players)
        self.history.pop()
        self.state = self.get_state(pid)
        return True

    def get_state(self, player_id):
        '''Return the raw state of one player

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        '''
        player = self.players[player_id]
        st = self.round.get_state_for(self.players, player_id)
        if self.is_over():
            actions = []
        else:
            from rlcard.games.tianwang.core import get_legal_actions
            actions = get_legal_actions(st)
        state = player.get_state(self.round.public, actions)
        state['_core_state'] = st          # 供 Env 直接复用核心状态对象
        return state

    @staticmethod
    def get_num_static_actions():
        '''Number of static action ids (singles + pairs)'''
        return STATIC_SIZE

    @staticmethod
    def get_dynamic_base():
        '''First dynamic action id (two-different-card combos)'''
        return STATIC_SIZE

    def get_num_actions(self):
        '''Total registered actions so far (static + lazily registered).

        Note: the env uses a fixed-size vector; see TianwangEnv.
        '''
        return ACTION_SPACE.next_dynamic_id

    def get_num_players(self):
        '''Return the number of players in one game'''
        return self.num_players

    def get_player_id(self):
        '''Return current player's id'''
        return self.round.current_player

    def get_players(self):
        '''Return all the players'''
        return self.players

    def is_over(self):
        '''Check whether the game is over'''
        return self.round.is_over()

    def get_payoffs(self):
        '''Return the payoffs of a game

        Returns:
            dict: payoffs {player_id: float}
        '''
        return self.payoffs
