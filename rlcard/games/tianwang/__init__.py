# -*- coding: utf-8 -*-
'''Tianwang (天王) — 极简原型扑克游戏环境

第一阶段: 核心数据结构(Card/Deck/Player)、发牌、全局动作映射、
get_legal_actions / legal-action-mask；
本包同时提供 RLCard 标准五件套 Dealer/Player/Round/Judger/Game。
'''
from rlcard.games.tianwang.dealer import TianwangDealer as Dealer
from rlcard.games.tianwang.judger import TianwangJudger as Judger
from rlcard.games.tianwang.player import TianwangPlayer as Player
from rlcard.games.tianwang.round import TianwangRound as Round
from rlcard.games.tianwang.game import TianwangGame as Game
