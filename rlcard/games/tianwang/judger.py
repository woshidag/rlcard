# -*- coding: utf-8 -*-
'''Implement Tianwang Judger class (终局判定)

胜负条件: 闲家(3家)抓分总和 >= 庄分(叫分数, 原型默认75) => 闲家胜;
否则庄家胜。返回各玩家 payoff (零和, 归一化到 [-1, 1])。
'''
from rlcard.games.tianwang.core import NUM_PLAYERS

DEFAULT_BID = 75      # 原型阶段固定庄分(正式接入叫分状态机后取实际叫分)


class TianwangJudger:
    '''Determine the winner and payoffs at the end of the game'''

    def __init__(self, np_random):
        '''Initialize judger

        Args:
            np_random: numpy random state
        '''
        self.np_random = np_random

    @staticmethod
    def judge_game(players, bid=DEFAULT_BID):
        '''Judge the final result.

        Args:
            players (list): list of TianwangPlayer with trick_score set
            bid (int): 庄家定约分

        Returns:
            (dict): payoffs for each player, zero-sum, in [-1, 1]
        '''
        dealer = next(p for p in players if p.is_dealer)
        peasant_score = sum(p.trick_score for p in players
                            if not p.is_dealer)
        peasants_win = peasant_score >= bid
        margin = abs(peasant_score - bid) / 200.0     # 归一化幅度
        scale = min(1.0, 0.5 + margin)               # 至少输赢0.5
        payoffs = {}
        for p in players:
            if p.is_dealer:
                payoffs[p.player_id] = scale if not peasants_win else -scale
            else:
                share = (peasant_score - bid) / max(1, 3 * 200.0)
                payoffs[p.player_id] = (scale / 3.0) if peasants_win \
                    else -(scale / 3.0)
        # 保证严格零和
        total = sum(payoffs.values())
        payoffs[dealer.player_id] -= total
        return payoffs
