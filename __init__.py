#!/usr/bin/env python
# -*- coding: utf-8 -*-

from traceback import print_exc
from pdb import set_trace
import ai
reload(ai)

# State:
#   0 1 2
#   3 4 5
#   6 7 8


def main(players):
    print '=== Restart ==='
    state = [None] * 9
    ret = evalS(state)
    turn = 0

    while ret is None:
        try:
            player = players[(turn % 2)]
            act = player.action(state)
            state = newstate(state, act, player.sgn)

            cnt0 = sum([(x == 'O') for x in state])
            cnt1 = sum([(x == 'X') for x in state])
            assert(cnt1 <= cnt0)

            ret = evalS(state)
            #show(state)
            turn += 1
        except:
            print_exc()
            set_trace()
    return ret, turn


keys = [
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 4, 8),
    (2, 4, 6),
    ]


def evalS(state):

    def evalK(ret, key):
        if ret is not None:
            return ret

        for sgn in ('O', 'X'):
            ret = sum([(state[k] == sgn) for k in key])
            if ret == 3:
                return sgn
        return None

    ret = reduce(evalK, keys, None)
    if ret:
        return ret
    if sum([(s is None) for s in state]) == 0:
        return 'draw'


def show(state):
    print 'Round table:'
    print state[:3]
    print state[3:6]
    print state[6:9]


def transform(state):
    state0 = [None] * 9
    for i, s in enumerate(state):
        if s != '0':
            state0[i] = s
    return state0


def newstate(state, a, sgn):
    state1 = list(state)
    state1[a] = sgn
    return state1


def actions(state):
    # Return possible actions
    for i, s in enumerate(state):
        if s is None:
            yield i


if __name__ == '__main__':
    # print evalS(transform('000000000'))
    # print evalS(transform('OOO000000'))
    # print evalS(transform('000XXX000'))
    # print evalS(transform('OOXXOOXXO'))

    players = [ai.player('O'), ai.playerA('X')]
    players[1].genstate()
    players[1].train()
    # [player.load() for player in players]

    cnt = {'O': 0., 'X': 0., 'draw': 0.}
    for i in xrange(1000):
        ret = main(players)
        cnt[ret[0]] += 1
    print cnt
