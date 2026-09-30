# -*- coding: utf-8 -*-
'''Implement Tianwang Dealer class (RLCard 风格发牌器)

天王规则: 两副缩减牌库(去掉4/8/9/J/Q, 共68张), 留12张底牌,
其余56张平分给4人, 每人14张。
'''
import functools

from rlcard.games.tianwang.core import Deck, tianwang_sort_card


class TianwangDealer:
    '''Dealer will shuffle and deal the 68-card reduced double deck'''

    def __init__(self, np_random):
        '''Initialize the dealer with the reduced double deck

        Args:
            np_random: numpy random state used by the RLCard env
        '''
        self.np_random = np_random
        self.deck = Deck()
        self.kitty = []          # 12 张底牌
        self.dealer_id = None    # 庄家 id (叫分最低者), 由 Game 决定

    def shuffle(self):
        '''Randomly shuffle the deck using the env's np_random'''
        indices = self.np_random.permutation(len(self.deck.cards))
        self.deck.cards = [self.deck.cards[i] for i in indices]

    def deal_cards(self, players):
        '''Deal 14 cards to each player, keep 12 cards as kitty

        Args:
            players (list): list of TianwangPlayer objects
        '''
        self.shuffle()
        hand_num = (len(self.deck.cards) - 12) // len(players)   # 14
        for index, player in enumerate(players):
            current_hand = self.deck.cards[index * hand_num:(index + 1) * hand_num]
            current_hand.sort(key=functools.cmp_to_key(tianwang_sort_card))
            player.set_current_hand(current_hand)
            player.initial_hand = current_hand[:]
        self.kitty = self.deck.cards[len(players) * hand_num:]
        self.kitty.sort(key=functools.cmp_to_key(tianwang_sort_card))
        return self.kitty

    def determine_dealer(self, players):
        '''Determine the dealer (庄家) from bids.

        原型阶段简化: 叫分最低者做庄; 相同则座位号小者优先。
        (正式实现将替换为完整的 105起叫/40封底 叫分状态机)

        Args:
            players (list): list of TianwangPlayer objects with .bid set

        Returns:
            int: the dealer's player id
        '''
        self.dealer_id = min(
            players, key=lambda p: (p.bid if p.bid is not None else 999,
                                    p.player_id)).player_id
        return self.dealer_id

    def get_deck(self):
        '''Return string representation of the deck (RLCard API)'''
        return ''.join(repr(c) for c in self.deck.cards)

    def get_dealer_card(self):
        '''Return the kitty (底牌) as a string (RLCard API)'''
        return ''.join(repr(c) for c in self.kitty)
