from primitive import Primitive
from Levenshtein import distance as leven_dist

# consts for hashing
P = 10003
MOD = int(1e9 + 7)


def check_commutativity(primitive):
    if primitive.str in ["add", "multiply"]:
        if hash(primitive.nodes[1]) > hash(primitive.nodes[0]):
            primitive.nodes = primitive.nodes[::-1]


def list_nodes_with_hashes(primitive):
    ans = []
    hash_ = hash(primitive.str) % MOD

    check_commutativity(primitive)

    for node in primitive.nodes:
        list_nodes = list_nodes_with_hashes(node)
        hash_ *= P
        hash_ += list_nodes[-1][1]
        hash_ %= MOD
        ans += list_nodes

    ans.append((primitive, hash_))
    
    return ans


def compare(p1, p2):
    if p1.str != p2.str:
        return False

    check_commutativity(p1)
    check_commutativity(p2) 

    for l, r in zip(p1.nodes, p2.nodes):
        if not compare(l, r):
            return False

    return True


def get_first_structural_distance(p1, p2):

    nodes1 = list_nodes_with_hashes(p1)
    nodes2 = list_nodes_with_hashes(p2)
    
    max_same_part = 0
    
    for n1 in nodes1:
        for n2 in nodes2:
            if n1[1] == n2[1] and compare(n1[0], n2[0]):
                max_same_part = max(max_same_part, n1[0].get_tokens())

    return p1.get_tokens() + p2.get_tokens() - 2 * max_same_part


def get_second_structural_distance(p1, p2):
    w1 = p1.get_str_representation()
    w2 = p2.get_str_representation()
    return leven_dist(w1, w2)

