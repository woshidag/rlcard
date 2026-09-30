# -*- coding: utf-8 -*-
'''Tianwang Environment (RLCard Env 适配层)

状态编码(固定长度向量, 便于 DMC):
    [0,68)     手牌 one-hot 计数 (34 face x 2)
    [68,136)   已见牌计数 (34 x 2)
    [136,152)  主花色 one-hot(4) + 庄家 one-hot(4) + 自己座位 one-hot(4)
               + 剩余张数归一化(4)
    [152,280)  合法动作掩码前 128 位 (静态68 + 动态组合60)
    [280,288)  当前墩信息: 领出者 one-hot(4) + 领出类型(单/对)(2) + 垫2
state_shape = [288] x 4 人
动作空间: 固定 128 维向量。ID [0,68) 静态(单张/对子), ID [68,128) 为
"两张不等牌"组合的惰性注册槽位; 配合 legal action mask 使用。
'''
import numpy as np

from rlcard.envs import Env
from rlcard.games.tianwang.core import (
    ACTION_SPACE, ALL_FACES, FACE_IDX, STATIC_SIZE, NUM_PLAYERS, SUITS,
)
from rlcard.games.tianwang import Game

STATE_SHAPE = 288
ACTION_VEC_SIZE = 128          # 静态68 + 动态预留60


class TianwangEnv(Env):
    '''Tianwang Environment'''

    def __init__(self, config):
        '''Initialize the environment'''
        self.name = 'tianwang'
        self.game = Game()
        super().__init__(config)
        self.state_shape = [[STATE_SHAPE] for _ in range(self.num_players)]
        self.action_shape = [[ACTION_VEC_SIZE] for _ in range(self.num_players)]

    def _extract_state(self, state):
        '''Encode raw state into a fixed-size feature vector + mask.

        Args:
            state (dict): dict of original state produced by Game

        Returns:
            dict: {'processed_state': np.array[288], 'raw_state': dict,
                   'legal_actions': list[int]}
        '''
        core = state['_core_state']
        pid = state['self']

        feats = np.zeros(STATE_SHAPE, dtype=np.int32)

        # 手牌计数 one-hot (34*2)
        for c in core.hands[pid]:
            feats[FACE_IDX[c.key]] += 1
            feats[34 + FACE_IDX[c.key]] += 1
        # 已见牌计数 (34*2)
        seen = state['seen_cards']
        # seen_cards 是 repr 拼接串; 直接用 played_cards 重算更稳妥
        from collections import Counter
        cnt_seen = Counter()
        for p in self.game.players:
            for combo in p.played_cards:
                for c in combo.cards:
                    cnt_seen[FACE_IDX[c.key]] += 1
        for idx, n in cnt_seen.items():
            feats[68 + idx] = min(n, 2)
            if n > 2:
                feats[68 + 34 + idx] = 1
        # 主花色 / 庄家 / 自己
        ts = core.trump_suit
        if ts in SUITS:
            feats[136 + SUITS.index(ts)] = 1
        feats[140 + state['dealer']] = 1
        feats[144 + pid] = 1
        for i, n in enumerate(state['num_cards_left']):
            feats[148 + i] = n
        # 合法动作掩码 (前128位)
        mask = np.zeros(ACTION_VEC_SIZE, dtype=np.int32)
        legal = [a for a in state['actions'] if a < ACTION_VEC_SIZE]
        for a in state['actions']:
            if a >= ACTION_VEC_SIZE:
                raise RuntimeError(
                    f'action id {a} exceeds env action vector size '
                    f'{ACTION_VEC_SIZE}; increase ACTION_VEC_SIZE')
        mask[legal] = 1
        feats[152:152 + ACTION_VEC_SIZE] = mask
        # 当前墩信息
        if core.trick is not None:
            feats[280 + core.trick.leader] = 1
            feats[284 + core.trick.lead_combo.kind] = 1
        else:
            feats[284] = 1                       # 轮到领出

        processed = feats.astype(np.float32)
        return {'processed_state': processed,
                'raw_state': {k: v for k, v in state.items()
                              if k != '_core_state'},
                'legal_actions': list(state['actions'])}

    def get_state(self, player_id=None):
        '''Get one player's state (override base to pass core state)'''
        if player_id is None:
            player_id = self.game.get_player_id()
        return self._extract_state(self.game.get_state(player_id))

    def reset(self):
        '''Reset the game and return first state.

        与基类 Env.reset 语义一致(返回首玩家状态)，但直接走本类的
        _extract_state 通道，避免基类对 raw_state 做 get_raw_state。

        Returns:
            (tuple): (state dict of first player, first player id)
        '''
        self.action_recorder = []
        raw_state, player_id = self.game.init_game()
        self.timestep = 0
        return self._extract_state(raw_state), player_id

    def step(self, action, raw_action=False):
        '''Step forward with an action id (int).

        Args:
            action (int): the global action id chosen by the agent

        Returns:
            (tuple): (next state dict, next player id)
        '''
        cur = self.game.get_player_id()
        legal = self.game.get_state(cur)['actions']
        assert int(action) in legal, \
            f"Illegal action {action} for current player P{cur}"
        self.action_recorder.append(int(action))
        next_state, next_player = self.game.step(int(action))
        self.timestep += 1
        return self._extract_state(next_state), next_player

    def is_over(self):
        '''Check whether the game is over'''
        return self.game.is_over()

    def get_payoffs(self):
        '''Return payoffs as array ordered by player id'''
        payoffs = self.game.get_payoffs()
        return np.array([payoffs[i] for i in range(self.num_players)])

    def get_dealer_card(self):
        '''Get the kitty cards (string)'''
        return ''.join(repr(c) for c in self.game.round.dealer.kitty)

    def get_deck(self):
        '''Get the whole deck (string)'''
        return self.game.round.dealer.get_deck()
