from enum import Enum
import numpy as np

# this class is implemented in order to track the values that built superpositions can make
class Domain:
    def __init__(self, low, high):
        self.low = min(low, high)
        self.high = max(low, high)
    
    def contain(self, nDomain):
        return self.low <= nDomain.low and nDomain.high <= self.high

class DOMAINS:
    REAL = Domain(-np.inf, np.inf)
    POSITIVE = Domain(0, np.inf)
    NEGATIVE = Domain(-np.inf, 0)
    TWICE_REAL = (REAL, REAL)


class DomainException(Exception):
    pass

class Primitive:
    def __init__(self, func, valency, domain, codomain, string):
        self.func = func
        self.valency = valency
        self.nodes = []
        self.domain = domain
        self.codomain = codomain
        self.str = string
        
        self.inf = 10000


    def add_nodes(self, nodes):
        self.nodes = nodes
        return self

    def get_correct_value(self, value):
        if value == float('inf'):
            return self.inf
        if value == float('-inf'):
            return -self.inf
        if np.abs(value) > self.inf:
            return self.inf * np.sign(value)
        
        return value if value == value else 0

    def calc(self, x, y):
        if self.valency == 0:
            return self.get_correct_value(self.func(x, y))
        return self.get_correct_value(self.func(*[node.calc(x, y) for node in self.nodes]))


    def calc_domains(self):
        nodes_domains = [node.calc_domains() for node in self.nodes]

        if self.valency == 0:
            pass
        elif self.valency == 1:
            if not self.domain.contain(nodes_domains[0]):
                raise DomainException()
            self.domain = nodes_domains[0]
            self.codomain = Domain(self.func(self.domain.low), self.func(self.domain.high))

        elif self.valency == 2:
            
            # TODO remove/resolve switch
            fr, sc = nodes_domains

            if self.func == np.add:
                self.codomain = Domain(fr.low + sc.low, fr.high + sc.high)
            elif self.func == np.subtract:
                self.codomain = Domain(fr.low - sc.high, fr.high - sc.low)
            elif self.func == np.multiply:
                vars = np.array([fr.low * sc.low, fr.low * sc.high, fr.high * sc.low, fr.high * sc.high])
                vars = vars[vars == vars] # TODO bug
                self.codomain = Domain(np.min(vars), np.max(vars))

            elif self.func == np.divide:
                
                # TODO do true calculation

                vars = np.array([fr.low * sc.low, fr.low * sc.high, fr.high * sc.low, fr.high * sc.high])
                vars = vars[vars == vars] # TODO bug
                tmin, tmax = np.min(vars), np.max(vars)
                if tmin < 0 and tmax > 0:
                    self.codomain = DOMAINS.REAL
                elif tmax <= 0:
                    self.codomain = DOMAINS.NEGATIVE
                elif tmin >= 0:
                    self.codomain = DOMAINS.POSITIVE
                else:
                    raise Exception('Comparison failed')
            else:
                raise Exception('Undefined function')
        else:
            raise Exception('Undefined valency')

        return self.codomain


    def get_tokens(self):
        if self.valency == 0:
            return 1
        return 1 + np.sum([node.get_tokens() for node in self.nodes])
    

    def get_kth(self, n):
        i = 0
        while self.nodes[i].get_tokens() <= n:
            n -= self.nodes[i].get_tokens()
            i += 1
        if n == 0:
            return self, i
        return self.nodes[i].get_kth(n - 1)
        

    def get_random(self):
        rnd = np.random.randint(0, self.get_tokens() - 1)
        return self.get_kth(rnd)
    

    def __str__(self):
        nodes_names = [str(node) for node in self.nodes]
        if self.valency == 0:
            return self.str
        elif self.valency == 1:
            return self.str + '(' + nodes_names[0] + ')'
        elif self.valency == 2:
            return self.str + '(' + nodes_names[0] + ', ' + nodes_names[1] + ')'


    def print_as_tree(self, depth=0):
        print (" |" * depth, self.str)
        for node in self.nodes:
            node.print_as_tree(depth + 1)


    def get_str_representation(self):
        ans = str(self.str[0])

        for node in self.nodes:
            tmp = node.get_str_representation()
            ans += tmp

        return ans


    def get_height(self):
        max_ = 0
        for node in self.nodes:
            max_ = max(max_, node.get_height())
        return max_ + 1


    def get_number_of_leaves(self):
        if len(self.nodes) == 0:
            return 1
        return sum([node.get_number_of_leaves() for node in self.nodes])



class Primitives:
    TF = Primitive(lambda x, y: x, 0, DOMAINS.POSITIVE, DOMAINS.POSITIVE, 'tf')
    IDF = Primitive(lambda x, y: y, 0, DOMAINS.POSITIVE, DOMAINS.POSITIVE, 'idf')
    ADD = Primitive(np.add, 2, DOMAINS.TWICE_REAL, DOMAINS.TWICE_REAL, 'add')
    SUB = Primitive(np.subtract, 2, DOMAINS.TWICE_REAL, DOMAINS.TWICE_REAL, 'substract')
    MUL = Primitive(np.multiply, 2, DOMAINS.TWICE_REAL, DOMAINS.TWICE_REAL, 'multiply')
    DIV = Primitive(np.divide, 2, DOMAINS.TWICE_REAL, DOMAINS.TWICE_REAL, 'divide')
    LOG = Primitive(lambda x: np.log10(1+x), 1, DOMAINS.POSITIVE, DOMAINS.REAL, 'log')
    EXP = Primitive(np.exp, 1, DOMAINS.REAL, DOMAINS.POSITIVE, 'exp')
    # big S is actual in case that first letter is short name
    SQRT = Primitive(np.sqrt, 1, DOMAINS.POSITIVE, DOMAINS.POSITIVE, 'Sqrt')
    
PRIMITIVES = [ 
    Primitives.TF,
    Primitives.IDF,
    Primitives.ADD,
    Primitives.SUB,
    Primitives.MUL,
    Primitives.DIV,
    Primitives.LOG,
    Primitives.EXP,
    Primitives.SQRT,
]