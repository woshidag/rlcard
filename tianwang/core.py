# -*- coding: utf-8 -*-
"""
《天王 (Tianwang)》扑克游戏 —— 极简原型环境核心模块（第一阶段）
================================================================

任务一：核心数据结构 (Card / Deck / Player) + 基础发牌 (Deal)
任务二：全局动作映射 (Action Mapping) + get_legal_actions(state)

============================ 规则理解 ============================
【牌库】两副牌，去掉 4/8/9/J/Q，保留 2,3,5,6,7,10,K,A,小王,大王。
        每副 8*4+2 = 34 张，两副共 68 张。

【分牌】5=5分，10=10分，K=10分；全副总分 = 8*(5+10+10) = 200 分
        (缩减牌库每 rank 共 8 张：4 花色 × 2 副)。

【固定主牌(常主)】方块3(天王，全局最大)、大王、小王、所有的7、所有的2。
   定主后，主花色的其余普通牌(A,K,10,6,5,3)也升为主牌。

【牌力排序】(从大到小)
   天王(D3) > 大王 > 小王 > 主7 > 副7 > 主2 > 副2
   > 主花色普通牌(A>K>10>6>5>3) > 副牌(各花色内 A>K>10>6>5>3)
   注：主7 严格大于任何副牌(含副A)，主2 同理 —— "主牌中的7大于副牌中的7"。

【流程】4人局：68 张留 12 张底牌，其余 56 张平分给 4 家(每人 14 张)；
        定主(亮7/翻底，原型阶段简化为外部指定)；
        叫分 105 起叫、最低 40，叫分最低者做庄；
        庄家收 12 张底并重新埋 12 张底；
        胜负：闲家抓分总和 >= 庄分 => 闲家胜，否则庄家胜。

【出牌/吃墩】首家只能出 单张 或 对子（绝对没有甩牌！）。
   跟牌必须跟首家花色：
     - 首家出单张 x：有该花色必须出该花色单张；无该花色可主牌毙或垫牌(垫必输)。
     - 首家出对子 (s,r)：
         * 有 s 花色对子 -> 必须出对子；
         * 无 s 花色对子但有 >=2 张 s 花色单牌 -> 允许出"任意两张 s 花色单牌"
           （结算时比这两张中较大的一张）；
         * 只有 1 张 s 花色 -> 必须带上它 + 任意另一张牌；
         * 完全没有 s 花色 -> 主牌毙(一对主牌 或 两张不同主牌)，
           或垫任意两张杂牌(垫牌必输)。
   大小判定：同类可比则逐位比牌力；主杀副；"两张单牌"永远输给"对子"。
   扣底：最后一墩赢家获得底牌分数，前提是其最后一手出的牌中包含主牌。
=================================================================
"""
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

# ================================================================
# 常量
# ================================================================
NUM_PLAYERS = 4
HAND_SIZE = 14            # (68-12)/4
KITTY_SIZE = 12           # 底牌张数

RANKS = ['2', '3', '5', '6', '7', '10', 'K', 'A']   # 缩减牌库 rank
SUITS = ['H', 'S', 'D', 'C']                        # 红桃/黑桃/方块/梅花
JOKERS = ['xW', 'dW']                               # 小王 / 大王

TIANWANG = ('D', '3')     # 天王 = 方块3
SCORE_MAP = {'5': 5, '10': 10, 'K': 10}             # 分牌


def card_score(rank: str) -> int:
    """分牌分值；非分牌返回 0。"""
    return SCORE_MAP.get(rank, 0)


# ================================================================
# 1. Card —— 单张牌
# ================================================================
class Card:
    """一张扑克牌。两副牌同花色同点数视为同一"面"(face)，用 key 表示。"""

    __slots__ = ('suit', 'rank')

    def __init__(self, suit: str, rank: str):
        self.suit = suit    # 'H','S','D','C'；王牌的 suit 存 'xW'/'dW'
        self.rank = rank    # RANKS 之一；王牌的 rank 存 'xW'/'dW'

    @property
    def key(self) -> Tuple[str, str]:
        return (self.suit, self.rank)

    @property
    def is_joker(self) -> bool:
        return self.rank in JOKERS

    @property
    def score(self) -> int:
        return card_score(self.rank)

    def __eq__(self, other):
        return isinstance(other, Card) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        if self.is_joker:
            return '小王' if self.rank == 'xW' else '大王'
        suit_cn = {'H': '红桃', 'S': '黑桃', 'D': '方块', 'C': '梅花'}[self.suit]
        return f"{suit_cn}{self.rank}"

    def to_dict(self):
        return {'suit': self.suit, 'rank': self.rank}


def make_card(face: Tuple[str, str]) -> Card:
    return Card(face[0], face[1])


# ================================================================
# 2. 牌力等级（规则核心）
# ================================================================
_NORMAL_ORDER = {'A': 6, 'K': 5, '10': 4, '6': 3, '5': 2, '3': 1}


def trump_rank(card: Card, trump_suit: Optional[str]) -> int:
    """
    全局牌力等级，数值越大越强；同级互等(如 4 张主7)。
      100 天王(D3) | 90 大王 | 80 小王
      70 主7 / 60 副7 | 50 主2 / 40 副2
      10~15 主花色普通牌(A..3) | 1~6 副牌(A..3)
    """
    s, r = card.suit, card.rank
    if (s, r) == TIANWANG:
        return 100
    if r == 'dW':
        return 90
    if r == 'xW':
        return 80
    if r == '7':
        return 70 if s == trump_suit else 60
    if r == '2':
        return 50 if s == trump_suit else 40
    if s == trump_suit:                      # 主花色普通牌
        return 9 + _NORMAL_ORDER[r]
    return _NORMAL_ORDER[r]                  # 副牌


def is_trump(card: Card, trump_suit: Optional[str]) -> bool:
    """是否属于主牌集合(将牌)。"""
    return trump_rank(card, trump_suit) >= 10


# ================================================================
# 3. Combo —— 牌型组合（单张 / 对子 / 两张不等牌），绝无甩牌
#    kind: 0=单张  1=对子  2=两张不同牌(仅无奈跟牌/带牌/双主毙时使用)
# ================================================================
class Combo:
    __slots__ = ('kind', 'cards')

    def __init__(self, kind: int, cards: Tuple[Card, ...]):
        assert kind in (0, 1, 2) and len(cards) == (1 if kind == 0 else 2)
        self.kind = kind
        self.cards = tuple(cards)

    @property
    def lead_suit(self) -> str:
        """领出参照花色；两张不等牌以其中较大者的花色为准。"""
        if self.kind == 0:
            return self.cards[0].suit
        if self.kind == 1:
            return self.cards[0].suit
        hi = max(self.cards, key=lambda c: trump_rank(c, None))
        return hi.suit

    def sorted_strength(self, ts: str) -> List[int]:
        return sorted((trump_rank(c, ts) for c in self.cards), reverse=True)

    def __eq__(self, other):
        return (isinstance(other, Combo) and self.kind == other.kind
                and Counter(c.key for c in self.cards)
                == Counter(c.key for c in other.cards))

    def __hash__(self):
        return hash((self.kind, tuple(sorted(c.key for c in self.cards))))

    def __repr__(self):
        name = {0: '单', 1: '对', 2: '两单'}[self.kind]
        return f"<{name}:{'+'.join(map(repr, self.cards))}>"


# ================================================================
# 4. Deck —— 两副缩减牌库（68 张）
# ================================================================
class Deck:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.cards: List[Card] = []
        self._build()

    def _build(self):
        for _ in range(2):                       # 两副
            for s in SUITS:
                for r in RANKS:
                    self.cards.append(Card(s, r))
            for j in JOKERS:
                self.cards.append(Card(j, j))    # 王牌 suit/rank 同置
        assert len(self.cards) == 68

    def shuffle(self):
        self.rng.shuffle(self.cards)

    def draw(self, n: int) -> List[Card]:
        assert n <= len(self.cards)
        return [self.cards.pop() for _ in range(n)]

    @staticmethod
    def total_score() -> int:
        return 200          # 8张5(40) + 8张10(80) + 8张K(80)


# ================================================================
# 5. Player
# ================================================================
class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.hand: List[Card] = []
        self.bid: Optional[int] = None       # 叫分 40..105
        self.is_dealer: bool = False         # 庄家(叫分最低者)
        self.captured: List[Card] = []       # 吃到的墩牌
        self.trick_score: int = 0            # 抓分

    def add_cards(self, cards: List[Card]):
        self.hand.extend(cards)

    def remove_cards(self, cards: List[Card]):
        for c in cards:
            self.hand.remove(c)

    @property
    def hand_score(self) -> int:
        return sum(c.score for c in self.hand)

    def __repr__(self):
        return (f"Player(id={self.player_id}, cards={len(self.hand)}, "
                f"bid={self.bid}, dealer={self.is_dealer})")


# ================================================================
# 6. 发牌 Deal：68 -> 4*14 + 12 底
# ================================================================
def deal(seed: Optional[int] = None
         ) -> Tuple[List[Player], List[Card]]:
    deck = Deck(seed=seed)
    deck.shuffle()
    players = [Player(i) for i in range(NUM_PLAYERS)]
    for p in players:
        p.add_cards(deck.draw(HAND_SIZE))
    kitty = deck.draw(KITTY_SIZE)
    assert sum(len(p.hand) for p in players) == 56 and len(kitty) == 12
    return players, kitty


# ================================================================
# 7. 全局动作映射 ActionSpace
#    [0, 34)          单张: 每个 face 一个 ID
#    [34, 68)         对子: 每个 face 一个 ID
#    [68, +inf)       两张不等牌的组合: 惰性注册、全局去重、ID 永久稳定
# ================================================================
ALL_FACES: List[Tuple[str, str]] = (
    [(s, r) for s in SUITS for r in RANKS] + [(j, j) for j in JOKERS]
)                                  # 34 种牌面
FACE_IDX = {f: i for i, f in enumerate(ALL_FACES)}
PAIR_BASE = len(ALL_FACES)         # 34
STATIC_SIZE = PAIR_BASE * 2        # 68


def single_action_id(face: Tuple[str, str]) -> int:
    return FACE_IDX[face]


def pair_action_id(face: Tuple[str, str]) -> int:
    return PAIR_BASE + FACE_IDX[face]


def action_to_combo(action_id: int) -> Combo:
    return ACTION_SPACE.decode(action_id)


class ActionSpace:
    def __init__(self):
        self.id_to_combo: Dict[int, Combo] = {}
        self.combo_to_id: Dict[Combo, int] = {}
        for f in ALL_FACES:
            c1 = Combo(0, (make_card(f),))
            self.id_to_combo[single_action_id(f)] = c1
            self.combo_to_id[c1] = single_action_id(f)
            c2 = Combo(1, (make_card(f), make_card(f)))
            self.id_to_combo[pair_action_id(f)] = c2
            self.combo_to_id[c2] = pair_action_id(f)
        self.next_dynamic_id = STATIC_SIZE

    def register(self, combo: Combo) -> int:
        cid = self.combo_to_id.get(combo)
        if cid is not None:
            return cid
        cid = self.next_dynamic_id
        self.next_dynamic_id += 1
        self.id_to_combo[cid] = combo
        self.combo_to_id[combo] = cid
        return cid

    def decode(self, action_id: int) -> Combo:
        return self.id_to_combo[action_id]

    def encode(self, combo: Combo) -> int:
        return self.register(combo)


ACTION_SPACE = ActionSpace()       # 全局单例


# ================================================================
# 8. 状态容器
# ================================================================
class TrickState:
    """当前一墩：领出者、领出牌型、已跟牌序列。"""
    def __init__(self, leader: int, lead_combo: Combo):
        self.leader = leader
        self.lead_combo = lead_combo
        self.plays: List[Tuple[int, Combo]] = []   # [(player_id, combo), ...]

    def current_player(self) -> int:
        return (self.leader + len(self.plays)) % NUM_PLAYERS


class TianwangState:
    """原型状态：四家手牌 + 主花色 + 当前墩(None 表示轮到领出)。"""
    def __init__(self, hands: List[List[Card]], trump_suit: str,
                 trick: Optional[TrickState] = None):
        self.hands = hands
        self.trump_suit = trump_suit
        self.trick = trick


# ================================================================
# 9. 第二步核心：get_legal_actions(state)
# ================================================================
def get_legal_actions(state: TianwangState) -> List[int]:
    """
    返回当前行动玩家的合法动作 ID 列表（已排序去重）。
      - 领出：所有单张 + 所有对子（无甩牌）；
      - 跟单张：见 _follow_single；
      - 跟对子：见 _follow_pair —— 严格要求"无该花色对子时，
        把该花色的任意两张单牌组合加入合法动作"。
    """
    trick = state.trick
    if trick is None:
        pid = 0                     # 原型约定：开局由 P0 领出
        return _lead_actions(state.hands[pid])
    pid = trick.current_player()
    hand = state.hands[pid]
    ts = state.trump_suit
    lead = trick.lead_combo
    if lead.kind == 0:
        return sorted(set(_follow_single(hand, lead.cards[0], ts)))
    return sorted(set(_follow_pair(hand, lead.cards[0].suit, ts)))


def get_legal_action_mask(state: TianwangState,
                          extra_size: int = 256) -> np.ndarray:
    """布尔掩码版本：静态 68 位 + 预留动态组合位。"""
    size = max(ACTION_SPACE.next_dynamic_id, STATIC_SIZE + extra_size)
    mask = np.zeros(size, dtype=bool)
    for aid in get_legal_actions(state):
        mask[aid] = True
    return mask


def _lead_actions(hand: List[Card]) -> List[int]:
    """领出动作：单张 + 对子（绝无甩牌）。"""
    cnt = Counter(c.key for c in hand)
    ids = [single_action_id(f) for f in cnt]
    ids += [pair_action_id(f) for f, n in cnt.items() if n >= 2]
    return sorted(ids)


def _follow_single(hand: List[Card], lead_card: Card, ts: str) -> List[int]:
    """跟单张：有首家花色必须出该花色单张；否则毙/垫均合法(垫必输)。"""
    same = [c for c in hand if c.suit == lead_card.suit]
    if same:
        # 注意：同一 face 的两张复制只映射到同一个动作 ID
        return [single_action_id(c.key) for c in same]
    return [single_action_id(c.key) for c in hand]


def _follow_pair(hand: List[Card], lead_suit: str, ts: str) -> List[int]:
    """跟对子：四级优先级，严格校验花色。"""
    cnt = Counter(c.key for c in hand)
    suit_pairs = [f for f, n in cnt.items() if n >= 2 and f[0] == lead_suit]
    suit_cnt = sum(n for f, n in cnt.items() if f[0] == lead_suit)

    # ① 有该花色对子 -> 必须出对子
    if suit_pairs:
        return [pair_action_id(f) for f in suit_pairs]

    # ② 无对子但 >=2 张该花色单牌 -> "任意两张该花色单牌"组合入列
    if suit_cnt >= 2:
        faces = [f for f in cnt if f[0] == lead_suit]
        ids = []
        for i, fa in enumerate(faces):
            if cnt[fa] >= 2:                     # 同名两张(实为对子级)
                ids.append(ACTION_SPACE.register(
                    Combo(2, (make_card(fa), make_card(fa)))))
            for fb in faces[i + 1:]:
                ids.append(ACTION_SPACE.register(
                    Combo(2, (make_card(fa), make_card(fb)))))
        return ids

    # ③ 只剩 1 张该花色 -> 必须带它 + 任意另一张
    if suit_cnt == 1:
        only = next(make_card(f) for f in cnt if f[0] == lead_suit)
        ids = []
        for f, n in cnt.items():
            if f[0] != lead_suit:
                ids.append(ACTION_SPACE.register(Combo(2, (only, make_card(f)))))
            elif n >= 2:                          # 理论上不会到这里
                ids.append(ACTION_SPACE.register(Combo(2, (only, make_card(f)))))
        return ids

    # ④ 绝该花色 -> 主牌毙(对子级/两张不等) 或 垫任意两张杂牌
    trumps = [c for c in hand if is_trump(c, ts)]
    others = [c for c in hand if not is_trump(c, ts)]
    ids = []
    if trumps:
        # 有主必毙：只允许主牌组合（不允许pass，也不允许纯垫）
        for a, b in _two_combos(trumps):          # 毙：一对主牌 or 两单主
            k = 1 if a.key == b.key else 2
            ids.append(ACTION_SPACE.register(Combo(k, (a, b))))
        return ids
    # 无主可毙 -> 垫任意两张杂牌（垫牌必输但仍为合法动作）
    for a, b in _two_combos(others if others else hand):
        ids.append(ACTION_SPACE.register(Combo(2, (a, b))))
    return ids


def _two_combos(cards: List[Card]) -> List[Tuple[Card, Card]]:
    """枚举 cards 中所有不重复的"两张"组合(同 face 计对子)。"""
    seen: Dict[Tuple[str, str], Card] = {}
    counts: Counter = Counter()
    for c in cards:
        seen.setdefault(c.key, c)
        counts[c.key] += 1
    uniq = list(seen.values())
    out = []
    for i, ca in enumerate(uniq):
        if counts[ca.key] >= 2:
            out.append((ca, ca))
        for cb in uniq[i + 1:]:
            out.append((ca, cb))
    return out


# ================================================================
# 10. 墩结算辅助（第三步会用到，先给出正确实现）
#     规则：主杀副；两张单牌永输对子；垫牌必输；同级互等保持先出者
# ================================================================
def _play_class(combo: Combo, ts: str) -> int:
    """把一手牌归入比较类别(数值越大越强)。"""
    if combo.kind == 0:
        return 1 if is_trump(combo.cards[0], ts) else 0     # 副单 / 主单(毙)
    ranks = [trump_rank(c, ts) for c in combo.cards]
    if combo.kind == 1:                                     # 真对子
        return 5 if all(r >= 10 for r in ranks) else 4      # 主对 / 副对
    # kind==2 两张不等
    if all(r >= 10 for r in ranks):                         # 双主毙(两层)
        return 6
    if any(r >= 10 for r in ranks):                         # 单主毙+带牌(一层)
        return 2
    return -1                                               # 垫两单 => 必输


def beats(a: Combo, b: Combo, ts: str) -> bool:
    """a 能否压过当前最强牌 b。"""
    ca, cb = _play_class(a, ts), _play_class(b, ts)
    # —— 强制规则："两张单牌"类跟牌(纯垫 / 一主+带牌)永远输给对子；
    #    "双主毙"(两张不同主牌)属于两层毙牌，可以赢副牌对子。——
    if a.kind == 2 and b.kind == 1 and ca in (2, -1):
        return False
    if a.kind == 1 and b.kind == 2 and cb in (2, -1):
        return True                          # 对子必胜两单/一主带牌
    if ca != cb:
        return ca > cb                       # 类别序: 垫(-1)<副单<一主毙<副对<双毙/主对
    ra = sorted((trump_rank(c, ts) for c in a.cards), reverse=True)
    rb = sorted((trump_rank(c, ts) for c in b.cards), reverse=True)
    if len(ra) != len(rb):                   # 同为单张类
        return ra[0] > rb[0]
    return ra > rb                           # 同类逐位字典序比较


def trick_winner(trick: TrickState, ts: str) -> int:
    best_pid, best_combo = trick.leader, trick.lead_combo
    for pid, combo in trick.plays:
        if beats(combo, best_combo, ts):
            best_pid, best_combo = pid, combo
    return best_pid


# ================================================================
# demo & 自检
# ================================================================
def _demo():
    print("=" * 64)
    print("第一步：数据结构与发牌")
    print("=" * 64)
    players, kitty = deal(seed=2026)
    for p in players:
        print(f"{p}")
        print(f"   手牌: {sorted(map(repr, p.hand), key=str)}")
    print(f"底牌({len(kitty)}张): {sorted(map(repr, kitty), key=str)}")
    total = sum(c.score for p in players for c in p.hand) + \
        sum(c.score for c in kitty)
    print(f"分牌守恒校验: 全场总分 = {total} (应为 {Deck.total_score()})")

    print("\n" + "=" * 64)
    print("第二步：get_legal_actions 演示")
    print("=" * 64)
    ts = 'D'                                   # 假设本局定主：方块
    hands = [list(p.hand) for p in players]

    # -- 场景A: P0 领出(单张+对子) --
    st = TianwangState(hands, ts, trick=None)
    acts = get_legal_actions(st)
    print(f"[A] P0 领出: {len(acts)} 个合法动作, 例如:", end=' ')
    print([str(ACTION_SPACE.decode(a)) for a in acts[:6]], '...')

    # -- 场景B: P0 领出红桃5对, P1 跟牌 --
    trick = TrickState(0, Combo(1, (Card('H', '5'), Card('H', '5'))))
    st = TianwangState(hands, ts, trick)
    acts = get_legal_actions(st)
    print(f"[B] P{trick.current_player()} 跟 红桃5对:")
    for a in acts:
        print(f"    id={a:>3} -> {ACTION_SPACE.decode(a)}")

    # -- 场景C: 构造手牌验证全部跟牌分支 --
    print("-" * 64)
    print("[C] 手工构造场景逐一验证:")
    H = lambda r: Card('H', r)
    S = lambda r: Card('S', r)
    D = lambda r: Card('D', r)
    C_ = lambda r: Card('C', r)

    lead_pair = TrickState(0, Combo(1, (H('5'), H('5'))))

    cases = {
        "有红桃对子(必须出对)":      [H('A'), H('A'), S('K'), D('7')],
        "无对但2张红桃(两单组合)":   [H('A'), H('K'), S('K'), D('7')],
        "仅1张红桃(带牌)":           [H('A'), S('K'), D('7'), C_('2')],
        "绝红桃(可毙可垫)":          [S('K'), D('7'), D('3'), C_('2')],
    }
    for desc, hand in cases.items():
        st = TianwangState([hand, [], [], []], trump_suit='D',
                           trick=lead_pair)
        acts = get_legal_actions(st)
        combos = [str(ACTION_SPACE.decode(a)) for a in acts]
        print(f"  {desc:24s} -> {combos}")

    # -- 掩码输出 --
    mask = get_legal_action_mask(st)
    print(f"\nLegal Action Mask: shape={mask.shape}, "
          f"合法位数={int(mask.sum())}")

    # -- 牌力排序自检 --
    print("-" * 64)
    order = [('D','3'),('dW','dW'),('xW','xW'),('D','7'),('S','7'),
             ('D','2'),('S','2'),('D','A'),('S','A'),('S','3')]
    print("主=D 时牌力: ", {repr(make_card(f)): trump_rank(make_card(f), 'D')
                            for f in order})


if __name__ == '__main__':
    _demo()
