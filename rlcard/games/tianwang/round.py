# -*- coding: utf-8 -*-
'''Implement Tianwang Round class (RLCard 风格回合/墩循环)

原型阶段：一个 Round 即整局打牌阶段(类似升级的出牌吃墩全过程)。
流程: 领出(单张/对子) -> 依序跟牌 -> 4家全出后结算一墩 ->
      赢家领下一墩; 14 墩打完后结算抓分与扣底。
'''
import numpy as np

from rlcard.games.tianwang import Dealer
from rlcard.games.tianwang.core import (
    TianwangState, TrickState, Combo, ACTION_SPACE,
    trump_rank, is_trump, card_score, determine_trump, get_legal_actions,
    trick_winner, NUM_PLAYERS,
)


class TianwangRound:
    '''Round keeps the trick-taking loop running'''

    def __init__(self, np_random):
        '''Initialize round

        Args:
            np_random: numpy random state of the env
        '''
        self.np_random = np_random
        self.dealer = Dealer(self.np_random)
        self.trick = None                 # 当前墩 TrickState 或 None(轮到领出)
        self.trace = []                   # [(player_id, combo_str), ...]
        self.current_player = None
        self.num_tricks = 0               # 已完成墩数
        self.last_trick_winner = None     # 上一墩(终局时=最后一墩)赢家
        self.kitty = []                   # 庄家埋底后的最终底牌
        self.public = {}

    def initiate(self, players, kitty=None):
        '''Start the playing phase.

        原型简化: 定主由翻底决定(determine_trump), 叫分状态机后续接入;
        默认 P0 做庄(若 players 已带 bid 则由 Game 事先设定 is_dealer)。

        Args:
            players (list): list of TianwangPlayer objects (hands dealt)
            kitty (list): 12 张底牌; None 时从 dealer 取
        '''
        if kitty is None:
            kitty = self.dealer.kitty
        self.trump_suit = determine_trump(kitty, self.np_random)

        dealer_id = next(p.player_id for p in players if p.is_dealer) \
            if any(p.is_dealer for p in players) else 0
        for p in players:
            p.is_dealer = (p.player_id == dealer_id)

        # 庄家收底再埋底: 原型策略 = 埋掉手牌中分值最低(优先非分牌)的12张
        hand = players[dealer_id].current_hand
        full = sorted(hand + kitty, key=lambda c: (card_score(c.rank),
                                                   -trump_rank(c, self.trump_suit)))
        buried = full[:12]
        rest = full[12:]
        players[dealer_id].set_current_hand(rest)
        self.kitty = buried

        self.dealer_id = dealer_id
        self.current_player = dealer_id   # 庄家首攻
        self._build_public(players)

    def _build_public(self, players):
        self._players_ref = players
        self.public = {
            'deck': self.dealer.get_deck(),
            'seen_cards': self._seen_cards(players),
            'trump_suit': self.trump_suit,
            'dealer': self.dealer_id,
            'trace': self.trace,
            'played_cards': ['' for _ in range(NUM_PLAYERS)],
            'kitty_size': len(self.kitty),
            'num_cards_left': [],
        }
        self._update_num_cards(players)

    def _update_num_cards(self, players):
        self.public['num_cards_left'] = [
            (len(p.current_hand), p.trick_score) for p in players]
        self.public['seen_cards'] = self._seen_cards(players)

    @staticmethod
    def _seen_cards(players):
        played = []
        for p in players:
            for combo in p.played_cards:
                played.extend(combo.cards)
        return ''.join(sorted(repr(c) for c in played))

    def get_state_for(self, players, pid):
        '''Build TianwangState (core.py) for the acting player'''
        hands = [p.current_hand for p in players]
        return TianwangState(hands, self.trump_suit, self.trick)

    def proceed_round(self, players, action_id):
        '''One play step.

        Args:
            players (list): the players
            action_id (int): the chosen legal action id

        Returns:
            (next_player_id, done_flag)
        '''
        pid = self.current_player
        player = players[pid]
        combo = player.play(action_id)
        self.trace.append((pid, str(combo)))

        if self.trick is None:                       # 领出，开新墩
            self.trick = TrickState(pid, combo)
        else:                                        # 跟牌
            self.trick.plays.append((pid, combo))

        if len(self.trick.plays) == NUM_PLAYERS - 1:  # 一墩打完 -> 结算
            self._settle_trick(players)

        self._update_num_cards(players)
        if all(len(p.current_hand) == 0 for p in players):
            self._settle_kitty(players)              # 扣底
            self.current_player = None
            return None, True
        return self.current_player, False

    def _settle_trick(self, players):
        '''Resolve current trick: winner collects the cards'''
        winner_id = trick_winner(self.trick, self.trump_suit)
        cards = list(self.trick.lead_combo.cards)
        for _, combo in self.trick.plays:
            cards.extend(combo.cards)
        players[winner_id].win_trick(cards)
        self.last_trick_winner = winner_id
        self.last_trick_play = {pid: combo
                               for pid, combo in [(self.trick.leader,
                                                  self.trick.lead_combo)] +
                              list(self.trick.plays)}
        for pid, combo in self.last_trick_play.items():
            players[pid]._last_play = combo
        self.current_player = winner_id
        self.trick = None
        self.num_tricks += 1

    def _settle_kitty(self, players):
        '''扣底: 最后一墩赢家拿底分, 前提是其最后一手包含主牌'''
        w = self.last_trick_winner
        last_play = players[w]._last_play
        has_trump = any(is_trump(c, self.trump_suit) for c in last_play.cards)
        kitty_score = sum(card_score(c.rank) for c in self.kitty)
        if has_trump:
            players[w].trick_score += kitty_score
            players[w].captured.extend(self.kitty)
        # (正式版将加入翻倍倍数规则; 原型按"含主牌才得底分"实现)

    def step_back(self, players):
        '''Reverse the last play (allow_step_back support)'''
        pid, _ = self.trace.pop()
        players[pid].play_back()
        if self.trick is not None and len(self.trick.plays) >= 1 and \
                self.trick.plays[-1][0] == pid:
            self.trick.plays.pop()
        elif self.trick is not None and self.trick.leader == pid:
            self.trick = None
        self.current_player = pid
        self._update_num_cards(players)
        return pid

    def is_over(self):
        return self.current_player is None
