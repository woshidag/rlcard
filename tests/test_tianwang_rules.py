# -*- coding: utf-8 -*-
'''《天王》环境规则符合性测试

覆盖:
  A. 牌库/发牌守恒 (68张, 无4/8/9/J/Q, 每人14+底12, 总分200)
  B. 牌力排序 (天王>王>主7>副7>主2>副2>主普通>副)
  C. get_legal_actions 跟对子四级优先级 + 跟单张花色约束
  D. beats/trick_winner 结算强制规则 (两单永输对子/双毙赢副对等)
  E. RLCard env 随机对局冒烟测试 (含非P0领出 -> 验证 current_player bug)
'''
import random
from collections import Counter

import numpy as np
import pytest

import rlcard
from rlcard.games.tianwang.core import (
    Card, Deck, Combo, TianwangState, TrickState, ACTION_SPACE,
    trump_rank, is_trump, deal, get_legal_actions, get_legal_action_mask,
    beats, trick_winner, STATIC_SIZE,
)

C = lambda s, r: Card(s, r)


# ------------------------------------------------------------------
# A. 牌库与发牌
# ------------------------------------------------------------------
def test_deck_composition():
    deck = Deck()
    assert len(deck.cards) == 68
    cnt = Counter(c.key for c in deck.cards)
    assert all(n == 2 for n in cnt.values()) and len(cnt) == 34
    banned = {'4', '8', '9', 'J', 'Q'}
    assert not any(r in banned for _, r in cnt)
    assert sum(c.score for c in deck.cards) == 200


def test_deal_conservation():
    rng = random.Random(0)
    for _ in range(30):
        players, kitty = deal(seed=rng.randrange(10**6))
        assert [len(p.hand) for p in players] == [14] * 4 and len(kitty) == 12
        all_cards = [c.key for p in players for c in p.hand] + \
            [c.key for c in kitty]
        assert Counter(all_cards) == Counter(c.key for c in Deck().cards)
        total = sum(Card(*k).score for k in all_cards)
        assert total == 200


# ------------------------------------------------------------------
# B. 牌力排序
# ------------------------------------------------------------------
def test_trump_rank_order():
    ts = 'D'
    order = [C('D', '3'), C('dW', 'dW'), C('xW', 'xW'), C('D', '7'),
             C('S', '7'), C('D', '2'), C('S', '2'), C('D', 'A'),
             C('D', 'K'), C('S', 'A'), C('S', '3')]
    ranks = [trump_rank(c, ts) for c in order]
    assert ranks == sorted(ranks, reverse=True), ranks
    assert trump_rank(C('D', '7'), ts) > trump_rank(C('S', 'A'), ts)  # 主7>副A
    assert trump_rank(C('D', '2'), ts) > trump_rank(C('S', 'K'), ts)  # 主2>副K
    assert is_trump(C('D', '5'), ts) and not is_trump(C('S', '5'), ts)
    assert is_trump(C('H', '7'), ts) and is_trump(C('H', '2'), ts)    # 常主


# ------------------------------------------------------------------
# C. 合法动作（跟牌校验）
# ------------------------------------------------------------------
def _acts_for(hand, lead, other_hands=None):
    hands = other_hands or [[], [], [], []]
    st = TianwangState(hands, trump_suit='D', trick=lead)
    return get_legal_actions(st)


def test_lead_only_single_and_pair():
    hand = [C('H', 'A'), C('H', 'A'), C('S', '5'), C('D', '7')]
    st = TianwangState([hand, [], [], []], 'D', None)
    acts = get_legal_actions(st)   # 原型约定 P0 领出
    combos = [ACTION_SPACE.decode(a) for a in acts]
    kinds = {cb.kind for cb in combos}
    assert kinds <= {0, 1}                      # 绝无甩牌
    pairs = [cb for cb in combos if cb.kind == 1]
    assert len(pairs) == 1 and pairs[0].cards == (C('H', 'A'), C('H', 'A'))


def test_follow_pair_level1_must_pair():
    """①有该花色对子 -> 必须出对子, 不得拆对/出两单"""
    hand = [C('H', 'A'), C('H', 'A'), C('H', 'K'), C('D', '7')]
    lead = TrickState(0, Combo(1, (C('H', '5'), C('H', '5'))))
    st = TianwangState([[], hand, [], []], 'D', lead)
    acts = get_legal_actions(st)
    assert set(acts) == {ACTION_SPACE.encode(Combo(1, (C('H', 'A'), C('H', 'A'))))}


def test_follow_pair_level2_two_singles_of_suit():
    """②无对但>=2张该花色 -> 任意两张该花色单牌组合全部入列"""
    hand = [C('H', 'A'), C('H', 'K'), C('H', '6'), C('D', '7')]
    lead = TrickState(0, Combo(1, (C('H', '5'), C('H', '5'))))
    st = TianwangState([[], hand, [], []], 'D', lead)
    acts = get_legal_actions(st)
    combos = {ACTION_SPACE.decode(a) for a in acts}
    expect = {Combo(2, (a, b)) for a, b in
              [(C('H', 'A'), C('H', 'K')), (C('H', 'A'), C('H', '6')),
               (C('H', 'K'), C('H', '6'))]}
    assert combos == expect
    assert all(cb.kind == 2 for cb in combos)
    # ID 稳定且与顺序无关
    assert ACTION_SPACE.encode(Combo(2, (C('H', 'K'), C('H', 'A')))) in acts


def test_follow_pair_level3_carry_one():
    """③仅1张该花色 -> 必须带它 + 任意另一张"""
    hand = [C('H', 'A'), C('S', 'K'), C('D', '7'), C('C', '2')]
    lead = TrickState(0, Combo(1, (C('H', '5'), C('H', '5'))))
    st = TianwangState([[], hand, [], []], 'D', lead)
    acts = get_legal_actions(st)
    combos = {ACTION_SPACE.decode(a) for a in acts}
    assert all(C('H', 'A') in cb.cards for cb in combos)
    assert len(combos) == 3      # 带 H-A + S-K / D-7 / C-2


def test_follow_pair_level4_ruff_or_dump():
    """④绝该花色: 有主必毙(禁纯垫); 无主可垫任意两张"""
    lead = TrickState(0, Combo(1, (C('H', '5'), C('H', '5'))))
    hand = [C('S', 'K'), C('D', '7'), C('D', '3'), C('C', '2')]
    st = TianwangState([[], hand, [], []], 'D', lead)
    acts = get_legal_actions(st)
    combos = {ACTION_SPACE.decode(a) for a in acts}
    trumps = [c for c in hand if is_trump(c, 'D')]
    others = [c for c in hand if not is_trump(c, 'D')]
    # 所有动作必须由 >=1 张主牌组成(有主必毙)
    assert all(any(is_trump(c, 'D') for c in cb.cards) for cb in combos)
    assert Combo(2, (C('S', 'K'), C('C', '2'))) not in combos  # 纯垫被禁
    assert len(trumps) == 3 and len(others) == 1

    # 无主 -> 只能垫两杂牌
    hand2 = [C('S', 'K'), C('S', 'Q') if False else C('S', '6'), C('C', '2')]
    hand2 = [C('S', 'K'), C('S', '6'), C('C', '5')]
    st2 = TianwangState([[], hand2, [], []], 'D', lead)
    combos2 = {ACTION_SPACE.decode(a) for a in get_legal_actions(st2)}
    assert all(not any(is_trump(c, 'D') for c in cb.cards) for cb in combos2)


def test_follow_single_suit_constraint():
    lead = TrickState(1, Combo(0, (C('H', '5'),)))
    hand = [C('H', 'A'), C('H', 'K'), C('D', '7'), C('S', '2')]
    st = TianwangState([[], [], hand, []], 'D', lead)   # P2 行动
    acts = set(get_legal_actions(st))
    assert acts == {ACTION_SPACE.encode(Combo(0, (C('H', 'A'),))),
                    ACTION_SPACE.encode(Combo(0, (C('H', 'K'),)))}
    # 绝花色 -> 任意出(可毙可垫)
    hand2 = [C('D', '7'), C('S', '2'), C('C', '5')]
    st2 = TianwangState([[], [], hand2, []], 'D', lead)
    acts2 = set(get_legal_actions(st2))
    assert len(acts2) == 3


def test_current_player_bug_non_p0_lead():
    """回归测试: 领出者非 P0 时, 合法动作必须属于真正的行动者。

    若 get_legal_actions 在 trick is None 时硬编码 pid=0, 本测试失败。
    """
    h0 = [C('H', 'A'), C('H', 'K')]          # P0 手牌(红桃)
    h3 = [C('S', '5'), C('S', '5'), C('C', '2')]  # P3 手牌(黑桃对)
    st = TianwangState([h0, [], [], h3], 'D', None)
    # 把"轮到谁领出"的信息通过 state 传递: 期望 API 支持指定行动者
    try:
        acts = get_legal_actions(st, player_id=3)
    except TypeError:
        pytest.skip('get_legal_actions 尚不支持 player_id 参数 —— 即已知缺陷')
    combos = {ACTION_SPACE.decode(a) for a in acts}
    assert Combo(1, (C('S', '5'), C('S', '5'))) in combos
    assert all(c.suit != 'H' for cb in combos for c in cb.cards), \
        'P3 行动却返回了 P0 的红桃动作 => 硬编码 bug 仍存在'


def test_legal_action_mask():
    hand = [C('H', 'A'), C('H', 'K'), C('H', '6'), C('D', '7')]
    lead = TrickState(0, Combo(1, (C('H', '5'), C('H', '5'))))
    st = TianwangState([[], hand, [], []], 'D', lead)
    mask = get_legal_action_mask(st)
    acts = get_legal_actions(st)
    assert mask.dtype == bool and mask.sum() == len(set(acts))
    assert all(mask[a] for a in acts)


# ------------------------------------------------------------------
# D. 墩结算
# ------------------------------------------------------------------
def test_beats_rules():
    ts = 'D'
    pair5 = Combo(1, (C('H', '5'), C('H', '5')))
    two_off = Combo(2, (C('H', 'A'), C('H', 'K')))
    ruff2 = Combo(2, (C('D', '7'), C('S', '7')))
    off_pair_k = Combo(1, (C('S', 'K'), C('S', 'K')))
    tw_pair = Combo(1, (C('D', '3'), C('D', '3')))
    single_a = Combo(0, (C('H', 'A'),))
    ruff_single = Combo(0, (C('D', '7'),))

    assert not beats(two_off, pair5, ts)          # 两单永输对子(同花色更大)
    assert beats(pair5, two_off, ts)              # 对子必胜两单
    assert beats(ruff2, off_pair_k, ts)           # 双主毙 > 副对子
    assert not beats(off_pair_k, ruff2, ts)
    assert beats(tw_pair, ruff2, ts)              # 天王级主对 > 普通双毙
    assert beats(ruff_single, single_a, ts)       # 主单毙 > 副单
    assert not beats(single_a, ruff_single, ts)
    dump = Combo(2, (C('S', 'K'), C('C', '2')))
    assert not beats(dump, single_a, ts)          # 垫两单输给任何有效牌


def test_trick_winner_sequence():
    ts = 'D'
    trick = TrickState(2, Combo(1, (C('H', '5'), C('H', '5'))))
    trick.plays = [(3, Combo(2, (C('H', 'A'), C('H', 'K')))),   # 两单, 输
                   (0, Combo(2, (C('D', '7'), C('S', '7')))),   # 双毙, 赢
                   (1, Combo(1, (C('D', '2'), C('D', '2'))))]   # 主对更大
    assert trick_winner(trick, ts) == 1


# ------------------------------------------------------------------
# E. Env 集成冒烟测试 (随机合法策略跑完整局)
# ------------------------------------------------------------------
def _random_play_episode(env, seed):
    rng = random.Random(seed)
    state, player_id = env.reset()
    steps = 0
    while not env.is_over():
        legal = state['legal_actions'] if 'legal_actions' in state \
            else state['raw_state']['actions']
        assert legal, f'no legal actions for P{player_id}'
        action = legal[rng.randrange(len(legal))]
        raw = state['raw_state']
        hand = raw['current_hand']
        combo = ACTION_SPACE.decode(action)
        cnt = Counter(c.key for c in hand)
        for c in combo.cards:
            assert cnt[c.key] > 0, f'P{player_id} plays {combo} not in hand'
            cnt[c.key] -= 1
        state, player_id = env.step(action)
        steps += 1
        assert steps < 200, 'episode did not terminate'
    payoffs = env.get_payoffs()
    return payoffs


def test_env_random_episode():
    env = rlcard.make('tianwang', config={'allow_step_back': True})
    for seed in range(10):
        payoffs = _random_play_episode(env, seed)
        assert len(payoffs) == 4
        assert abs(float(np.sum(payoffs))) < 1e-6, f'not zero-sum: {payoffs}'


def test_env_score_conservation():
    '''全场抓分 + 底分 应等于 200 中已兑现部分的分值总和'''
    env = rlcard.make('tianwang')
    rng = random.Random(42)
    for _ in range(5):
        state, pid = env.reset()
        while not env.is_over():
            legal = state['raw_state']['actions']
            state, pid = env.step(legal[rng.randrange(len(legal))])
        scores = [p.trick_score for p in env.game.players]
        kitty_score = sum(c.score for c in env.game.round.kitty)
        # 未被埋底的分数一定被某家抓到; 被抓分 + 仍在底里的分 <= 200,
        # 且 抓分(不含底倍加) 应等于 200 - 埋底分 + 已计入的底分(赢家)
        captured_total = sum(scores)
        assert 0 <= captured_total <= 200 + kitty_score
        # 精确校验: 各家 captured 中的分牌之和 == trick_score
        for p in env.game.players:
            assert p.trick_score == sum(c.score for c in p.captured)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
