# -*- coding: utf-8 -*-
"""天王 Tianwang 环境规则符合性测试

覆盖:
  T1  牌库/发牌守恒 (68张, 每人14+底12, 总分200)
  T2  牌力序 (天王>王>主7>副7>主2>副2>主普通>副)
  T3  get_legal_actions 四级跟对子规则 + 跟单张 + 领出无甩牌
  T4  beats/trick_winner 结算 (主杀副/双毙赢副对/两单永输对子/垫必输)
  T5  env.make + 随机 rollout: 逐步校验动作合法性、手牌守恒、抓分守恒
  T6  扣底规则: 最后一墩赢家含主牌才得底分
  T7  step_back / get_state / payoff 零和
"""
import random
import unittest
from collections import Counter

import numpy as np

from rlcard.games.tianwang.core import (
    Card, Deck, Combo, ACTION_SPACE, STATIC_SIZE, ALL_FACES,
    trump_rank, is_trump, deal, TianwangState, TrickState,
    get_legal_actions, get_legal_action_mask, beats, trick_winner,
    single_action_id, pair_action_id, make_card,
)
from rlcard.envs import make


class TestDeckAndDeal(unittest.TestCase):
    def test_deck_size_and_excluded_ranks(self):
        d = Deck()
        self.assertEqual(len(d.cards), 68)
        ranks = {c.rank for c in d.cards if not c.is_joker}
        self.assertEqual(ranks, {'2','3','5','6','7','10','K','A'})
        cnt = Counter(c.key for c in d.cards)
        self.assertTrue(all(v == 2 for v in cnt.values()))

    def test_deal_conservation(self):
        for seed in range(20):
            players, kitty = deal(seed=seed)
            self.assertEqual([len(p.hand) for p in players], [14]*4)
            self.assertEqual(len(kitty), 12)
            total = sum(c.score for p in players for c in p.hand) \
                    + sum(c.score for c in kitty)
            self.assertEqual(total, 200)
            allc = Counter(c.key for p in players for c in p.hand) \
                 + Counter(c.key for c in kitty)
            self.assertTrue(all(v == 2 for v in allc.values()))


class TestCardStrength(unittest.TestCase):
    def test_ordering(self):
        ts = 'D'
        order = [('D','3'), ('dW','dW'), ('xW','xW'), ('D','7'), ('S','7'),
                 ('D','2'), ('S','2'), ('D','A'), ('S','A')]
        ranks = [trump_rank(make_card(f), ts) for f in order]
        self.assertEqual(ranks, sorted(ranks, reverse=True))
        # 严格跨层：主7 > 副A，主2 > 副K
        self.assertGreater(trump_rank(Card('S','7'), 'H'),
                           trump_rank(Card('S','A'), 'H'))
        self.assertGreater(trump_rank(Card('S','2'), 'H'),
                           trump_rank(Card('S','K'), 'H'))
        # 同层互等
        self.assertEqual(trump_rank(Card('H','7'), 'D'),
                         trump_rank(Card('S','7'), 'D'))


class TestLegalActions(unittest.TestCase):
    def setUp(self):
        H = lambda r: Card('H', r); S = lambda r: Card('S', r)
        D = lambda r: Card('D', r); C_ = lambda r: Card('C', r)
        self.ts = 'D'
        self.lead_pair = TrickState(0, Combo(1, (H('5'), H('5'))))
        self.cases = {
            'pair_required':   [H('A'), H('A'), S('K'), D('7')],
            'two_singles':     [H('A'), H('K'), S('K'), D('7')],
            'one_only':        [H('A'), S('K'), D('7'), C_('2')],
            'void':            [S('K'), D('7'), D('3'), C_('2')],
            'void_no_trump':   [S('K'), S('6'), C_('5'), C_('A')],
        }

    def _acts(self, hand, trick):
        st = TianwangState([hand, [], [], []], self.ts, trick, actor=1)
        return get_legal_actions(st)

    def test_lead_no_shoot(self):
        """领出只有单张与对子，绝无甩牌/三张以上。"""
        hand = [Card('H','A'), Card('H','A'), Card('S','7'), Card('S','7'),
                Card('S','7'), Card('D','3')]
        st = TianwangState([hand, [], [], []], self.ts, None, actor=0)
        acts = get_legal_actions(st)
        for a in acts:
            self.assertIn(ACTION_SPACE.decode(a).kind, (0, 1))
        # 3张黑桃7 -> 允许出对，不允许三张
        self.assertIn(pair_action_id(('S','7')), acts)
        self.assertTrue(all(len(ACTION_SPACE.decode(a).cards) <= 2 for a in acts))

    def test_level1_must_pair(self):
        acts = self._acts(self.cases['pair_required'], self.lead_pair)
        self.assertEqual(acts, [pair_action_id(('H','A'))])  # 不得拆对

    def test_level2_two_singles_registered(self):
        acts = self._acts(self.cases['two_singles'], self.lead_pair)
        combos = [ACTION_SPACE.decode(a) for a in acts]
        # 红桃A+红桃K 必须出现在合法动作中（两张单牌组合）
        target = Combo(2, (Card('H','A'), Card('H','K')))
        self.assertIn(target, combos)
        self.assertTrue(all(a >= STATIC_SIZE for a in acts))
        # ID 稳定且全局去重
        self.assertEqual(ACTION_SPACE.encode(target),
                         ACTION_SPACE.encode(Combo(2, (Card('H','K'), Card('H','A')))))

    def test_level3_must_include_single_suit(self):
        acts = self._acts(self.cases['one_only'], self.lead_pair)
        combos = [ACTION_SPACE.decode(a) for a in acts]
        for cb in combos:
            self.assertEqual(cb.kind, 2)
            self.assertTrue(any(c.key == ('H','A') for c in cb.cards))
        self.assertEqual(len(acts), 3)  # HA 带其余3张之一

    def test_level4_void_must_ruff_if_has_trump(self):
        acts = self._acts(self.cases['void'], self.lead_pair)
        combos = [ACTION_SPACE.decode(a) for a in acts]
        for cb in combos:  # 有主必毙：所有动作全主牌
            self.assertTrue(all(is_trump(c, self.ts) for c in cb.cards))

    def test_level4_no_trump_discard(self):
        acts = self._acts(self.cases['void_no_trump'], self.lead_pair)
        combos = [ACTION_SPACE.decode(a) for a in acts]
        self.assertEqual(len(combos), 6)  # C(4,2)=C4取2
        for cb in combos:
            self.assertFalse(any(is_trump(c, self.ts) for c in cb.cards))

    def test_follow_single(self):
        lead = TrickState(0, Combo(0, (Card('H','10'),)))
        hand = [Card('H','A'), Card('H','3'), Card('S','K'), Card('D','7')]
        acts = self._acts(hand, lead)
        self.assertEqual(sorted(acts), sorted([single_action_id(('H','A')),
                                               single_action_id(('H','3'))]))
        # 绝花色可毙可垫
        hand2 = [Card('S','K'), Card('D','7'), Card('C','2')]
        acts2 = self._acts(hand2, lead)
        self.assertEqual(len(acts2), 3)

    def test_mask_shape(self):
        st = TianwangState([self.cases['void'], [], [], []], self.ts,
                           self.lead_pair, actor=1)
        mask = get_legal_action_mask(st)
        self.assertTrue(mask.dtype == bool)
        acts = get_legal_actions(st)
        self.assertEqual(int(mask.sum()), len(set(acts)))
        self.assertTrue(all(mask[a] for a in acts))


class TestTrickResolution(unittest.TestCase):
    ts = 'D'
    def test_trump_beats_offsuit(self):
        self.assertTrue(beats(Combo(0,(Card('S','7'),)),
                              Combo(0,(Card('H','A'),)), self.ts))
        self.assertFalse(beats(Combo(0,(Card('H','A'),)),
                               Combo(0,(Card('S','7'),)), self.ts))

    def test_double_ruff_beats_off_pair(self):
        off_pair = Combo(1, (Card('H','A'), Card('H','A')))
        two_trumps = Combo(2, (Card('S','7'), Card('C','2')))
        self.assertTrue(beats(two_trumps, off_pair, self.ts))
        self.assertFalse(beats(off_pair, two_trumps, self.ts))

    def test_two_singles_always_lose_to_pair(self):
        one_trump_plus = Combo(2, (Card('D','3'), Card('H','K')))  # 一主带牌
        off_pair = Combo(1, (Card('S','6'), Card('S','6')))
        self.assertFalse(beats(one_trump_plus, off_pair, self.ts))
        self.assertTrue(beats(off_pair, one_trump_plus, self.ts))
        pure_discard = Combo(2, (Card('C','5'), Card('S','K')))
        self.assertFalse(beats(pure_discard, Combo(0,(Card('H','3'),)), self.ts))

    def test_tianwang_top(self):
        self.assertTrue(beats(Combo(0,(Card('D','3'),)),
                              Combo(0,(Card('dW','dW'),)), self.ts))
        self.assertTrue(beats(Combo(0,(Card('dW','dW'),)),
                              Combo(0,(Card('xW','xW'),)), self.ts))

    def test_winner_sequence(self):
        t = TrickState(0, Combo(1, (Card('H','5'), Card('H','5'))))
        t.plays = [(1, Combo(2, (Card('H','A'), Card('H','K')))),      # 两单跟对
                   (2, Combo(2, (Card('S','7'), Card('C','2')))),      # 双主毙
                   (3, Combo(1, (Card('D','7'), Card('D','7'))))]      # 主对压双毙?
        w = trick_winner(t, self.ts)
        # 主对(5) vs 双毙(6): 按逐位比较，天王级 D7=70,D7=70 vs S7=70,C2=50
        self.assertIn(w, (2, 3))


class TestEnvRollout(unittest.TestCase):
    def _rollout(self, seed):
        rng = random.Random(seed)
        env = make('tianwang')
        state, pid = env.reset()
        steps = 0
        while not env.is_over():
            # 行动者 = 游戏当前玩家（env.step 返回的 pid 是下一步行动者，
            # 与 RLCard 约定一致；此处以 game.get_player_id() 为准）
            actor = env.game.get_player_id()
            legal = state['legal_actions']
            self.assertGreater(len(legal), 0)
            # 每个合法动作的牌必须在行动者手牌内
            hand = list(env.game.players[actor].current_hand)
            for a in legal:
                cb = ACTION_SPACE.decode(a)
                h = hand[:]
                for c in cb.cards:
                    self.assertIn(c, h)
                    h.remove(c)
            action = legal[rng.randrange(len(legal))]
            state, pid = env.step(action)
            steps += 1
            self.assertLess(steps, 200)
        # 终局守恒: 全部68张要么已打出要么在手; 抓分总和(+底分归属)合理
        played = sum(len(p.played_cards) and sum(len(c.cards) for c in p.played_cards) or 0
                     for p in env.game.players)
        inhand = sum(len(p.current_hand) for p in env.game.players)
        self.assertEqual(played + inhand, 68)
        scores = [p.trick_score for p in env.game.players]
        payoffs = env.get_payoffs()
        self.assertAlmostEqual(float(sum(payoffs)), 0.0, places=6)
        return env, scores

    def test_random_rollouts(self):
        total_score_seen = []
        for seed in range(10):
            env, scores = self._rollout(seed)
            dealer = next(p for p in env.game.players if p.is_dealer)
            peasant = sum(s for p, s in zip(env.game.players, scores)
                          if not p.is_dealer)
            # 闲家抓分 >= 庄分(默认75) => 闲家胜
            peasants_win = peasant >= 75
            if peasants_win:
                self.assertLess(payoff_sign(dealer), 0)
            total_score_seen.append(peasant)
        # 抓分应在合理区间 [0,200]
        self.assertTrue(all(0 <= s <= 200 for s in total_score_seen))

    def test_step_back(self):
        from rlcard.games.tianwang.game import TianwangGame
        g = TianwangGame(allow_step_back=True)
        g.init_game()
        pid = g.get_player_id()
        acts = g.get_state(pid)['actions']
        before = list(g.players[pid].current_hand)
        g.step(acts[0])
        ok = g.step_back()
        self.assertTrue(ok)
        self.assertEqual(sorted(map(str, g.players[pid].current_hand)),
                         sorted(map(str, before)))

    def test_kitty_settle_rule(self):
        """扣底: 最后一墩赢家最后一手含主牌才拿底分。"""
        env = make('tianwang')
        env.reset()
        rnd = env.game.round
        from rlcard.games.tianwang.core import card_score
        # 构造: 赢家最后一手为主牌 -> 应获得底分
        class FakePlayer: pass
        for has_trump_last, expect in [(True, True), (False, False)]:
            for p in env.game.players:
                p.trick_score = 0
                p.captured = []
            last_combo = (Combo(0, (Card('S','7'),)) if has_trump_last
                          else Combo(0, (Card('H','A'),)))
            env.game.players[0]._last_play = last_combo
            rnd.last_trick_winner = 0
            rnd.kitty = [Card('H','5'), Card('S','10'), Card('C','K')]  # 25分
            rnd._settle_kitty(env.game.players)
            got = env.game.players[0].trick_score
            self.assertEqual(got > 0, expect)
            if not expect:
                self.assertEqual(got, 0)


def payoff_sign(p):
    j = __import__('rlcard.games.tianwang.judger', fromlist=['TianwangJudger'])
    return 1  # placeholder unused


if __name__ == '__main__':
    unittest.main(verbosity=2)
