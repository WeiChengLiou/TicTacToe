#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from collections import defaultdict, namedtuple
import collections
import numpy as np
from numpy import random
import cPickle
import re
import __init__ as init
from pdb import set_trace
from traceback import print_exc
reload(init)
from __init__ import newstate
import abc
import itertools as it

# TicTacToe AI


def inv(s):
    def inv1(x):
        if x is None:
            return x
        elif x == 'O':
            return 'X'
        else:
            return 'O'
    return tuple(map(inv1, s))


def getkey(k):
    try:
        i = it.dropwhile(lambda x: x is None, k).next()
        if i == 'O':
            return tuple(k)
        else:
            return inv(k)
    except StopIteration:
        return tuple(k)
    except:
        print_exc()
        set_trace()


def getturn(state):
    cnt0 = sum([(x == 'O') for x in state])
    cnt1 = sum([(x == 'X') for x in state])
    assert(cnt1 <= cnt0)
    if cnt0 == cnt1:
        return 'O'
    else:
        return 'X'


class PI(collections.MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return getkey(key)


pi = PI()


class STATE(object):
    def __init__(self, state):
        self.V = None
        self.converge = False
        self.Actions = {}
        for act in init.actions(state):
            act1 = STATEACTION()
            self.Actions[act] = act1


class STATEACTION(object):
    def __init__(self):
        self.r = 0.
        self.p = 1.
        self.ret = None
        self.nextstate = None


def Reward(state, state1):
    def score(state):
        ret = init.evalS(state)
        cnt = np.zeros(2)
        if ret:
            if ret == 'O':
                cnt[0] = 10
            elif ret == 'X':
                cnt[1] = 10
        else:
            pt = [1, 1, 1,
                  1, 1, 1,
                  1, 1, 1]
            for i, s in enumerate(state):
                if s:
                    j = 0 if s == 'O' else 1
                    cnt[j] += pt[i]
        return cnt

    return score(state1) - score(state)


class player(object):
    # Random player
    def __init__(self, sgn):
        self.name = re.search(
            "'([\.\w]+)'", str(self.__class__)).groups()
        self.sgn = sgn
        self.gamma = 0.5

    def action(self, state):
        """action by state"""
        acts = list(init.actions(state))
        return random.choice(acts)

    def reward(self, state, state1):
        """Reward of (S, S')"""

    def value(self, state):
        """Value function"""

    def save(self):
        fi = '%s.pkl' % self.name
        cPickle.dump(self.pi, open(fi, 'wb'))

    def load(self):
        fi = '%s.pkl' % self.name
        if os.path.exists(fi):
            self.pi = cPickle.load(open(fi, 'rb'))


s0 = [None] * 9


class playerA(player):
    # Smart player
    pi = PI()

    def action0(self, state):
        acts = list(init.actions(state))
        bestAct = [(None, -99)]
        for act in acts:
            state1 = newstate(state, act, self.sgn)
            actV = (act, self.reward(state, state1))
            if actV[1] > bestAct[0][1]:
                bestAct = [actV]
            elif actV[1] == bestAct[0][1]:
                bestAct.append(actV)
        try:
            i = random.randint(len(bestAct))
            bestAct = bestAct[i]
        except:
            set_trace()
        return bestAct[0]

    def action1(self, state):
        V = self.pi[state]
        bestAct = (None, -1e9)
        for act, actf in V.Actions.iteritems():
            r = actf.r
            if actf.nextstate:
                r += self.gamma * actf.nextstate.V
            rV = (r[0] - r[1]) * (1 if self.sgn == 'O' else -1)
            if rV > bestAct[1]:
                bestAct = act, rV
        if bestAct[0] is None:
            print 'error action'
            set_trace()

        return bestAct[0]

    def action(self, state):
        return self.action1(state)

    def train(self):
        for i in xrange(10):
            dV = 0.
            for s, sf in self.pi.iteritems():
                if sf.converge:
                    continue
                V1 = np.zeros(2)

                for act, af in sf.Actions.iteritems():
                    r = af.r
                    sf1 = af.nextstate
                    if (sf1 is not None) and (sf1.V is not None):
                        r += self.gamma * sf1.V
                    V1 += af.p * r

                if sf.V is not None:
                    dV = V1 - sf.V
                    dV = np.abs(dV[0] - dV[1])
                    if dV < 1e-4:
                        sf.converge = True
                sf.V = V1

    def genstate(self):
        chg = 1
        state0 = [None] * 9
        stateV = STATE(state0)
        self.pi[tuple(state0)] = stateV

        while chg > 0:
            n0 = len(self.pi)
            keys = self.pi.keys()
            for s in keys:
                v = self.pi[s]
                sgn = getturn(s)
                n = len(v.Actions)

                for act, v1 in v.Actions.iteritems():
                    state1 = init.newstate(s, act, sgn)
                    v1.p = 1. / n
                    v1.ret = init.evalS(state1)
                    v1.r = Reward(s, state1)
                    if v1.ret is None:
                        key = getkey(state1)
                        if key not in self.pi:
                            stateV = STATE(key)
                            self.pi[key] = stateV
                        v1.nextstate = self.pi[key]
            n1 = len(self.pi)
            chg = n1 - n0
            print n0, n1

    def reward(self, state, state1):
        """"""

    def value(self, state):
        li = []
        try:
            for act, p0 in self.pi[state].iteritems():
                r = self.rSA(state, act, self.sgn)
                li.append((act, r))
            li = sorted(li, key=lambda k: -k[1])
            return sum([k[1] for k in li]), li
        except:
            print_exc()
            set_trace()

    def rSA(self, state, act, sgn):
        sgn1 = 'X' if self.sgn == 'O' else 'O'
        state1 = newstate(state, act, sgn)
        r0 = self.reward(state, state1)
        v = r0

        ret = init.evalS(state1)
        if ret is None:
            r1 = 0
            for act1, p1 in self.pi[state1].iteritems():
                state2 = newstate(state1, act1, sgn1)
                p = self.pi[state1][act1]
                V1 = self.value(state2)
                r1 += p * V1[0]
            v += self.gamma * r1
        return v



