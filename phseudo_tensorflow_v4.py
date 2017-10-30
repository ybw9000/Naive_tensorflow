import numpy as np

class Worker(object):
    """
    Base class for Transformer and Operator class.

    Every Worker subclass has to provide an forwardprop and a backwardprop
    method.
    """

    def __init__(self):
        pass

    def forwardprop(self):
        raise NotImplementedError

    def backwardprop(self):
        raise NotImplementedError

class Transformer(Worker):
    """
    Transformer class to wrap a function class in order to supply forwardprop
    and backwardprop.
    """
    def __init__(self, function):
        super().__init__()
        self.f = function()

    def forwardprop(self, Z):
        #self.Z = Z
        self.A = self.f.function(Z)
        return self.A

    def backwardprop(self, Z, dA):
        self.dZ = self.f.diff(Z, dA)
        return self.dZ

class Operator(Worker):
    """
    Operator class to wrap a operation class in order to supply forwardprop
    and backwardprop.
    """

    def __init__(self, operation):
        super().__init__()
        self.op = operation()

    def forwardprop(self, A, B):
        self.Z = self.op.op(A, B)
        return self.Z

    def backwardprop(self, A, B, dZ):
        self.dA, self.dB = self.op.diff(A, B, dZ)
        return self.dA, self.dB

class sigmoid():
    """
    function class.
    """
    def function(self, Z):
        return (1/(1 + np.exp(-Z)))
    def diff(self, Z, dA):
        A = self.function(Z)
        return np.multiply(A*(1 - A), dA)

class multiply():
    """
    operation class.
    """
    def op(self, A, B):
        return np.multiply(A, B)
    def diff(self, A, B, dA):
        return np.multiply(B, dA), np.multiply(A, dA)

class dot():
    """
    operation class.
    """
    def op(self, A, B):
        return np.dot(A, B)
    def diff(self, A, B, dA):
        return np.dot(dA, B.T), np.dot(A.T, dA)

class identity():
    """
    function class.
    """
    def function(self, Z):
        return Z
    def diff(self, Z, dA):
        return 1

class Constant():
    """input node on a graph, e.g., an input Tensor."""
    def __init__(self, Z):
        self.visited = False
        self.Z = Z
        self.children = [] #outgoing edges
        self.children_dA = {} #gradient from a child

    def get_visited(self):
        """
        check if already visited in a graph.
        """
        return self.visited

    def set_visited(self):
        """
        set visited.
        """
        self.visited = True

    def add_children(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def add_children_dA(self, child, child_dA):
        self.children_dA[child] = child_dA

    def get_childrend_dA(self):
        return self.children_dA

    def set_output(self, Z):
        """
        set current output.
        """
        self.Z = Z

    def get_output(self):
        """
        return current output.
        """
        return self.Z

    def forwardprop(self):
        return self.Z

    def backwardprop(self):
        pass

class Parameter(Constant):
    """
    parameter node in the graph, e.g., a weight/bias Tensor.
    wraps a Constant class and have additional gradient method and dZ field.
    """
    def __init__(self, Z):
        super().__init__(Z)
        self.dZ = None #output gradient
        self.dA = None #input gradient

    def gradient(self):
        """
        gather gradients sent from children nodes.
        """
        if not self.children_dA:
            return np.ones(shape = self.Z.shape)
        self.dA = np.zeros(shape = self.Z.shape)
        for child_dA in self.children_dA.values():
            self.dA += child_dA
        return self.dA

    def get_gradient(self):
        return self.dZ

    def backwardprop(self):
        """overwrite backwardprop of Constant class."""
        self.dZ = self.gradient()
        return self.dZ

class Variable(Parameter):
    """
    intermediate node in a graph, e.g., Z = X.dot(M).
    wraps Parameter class and has additional parents and operator/transformer
    fields. Has additional local_gradient method.
    """
    def __init__(self, operator, transformer):
        super().__init__(None)
        self.operator = operator
        self.tranformer = transformer
        self.A = None
        self.parents = []
        self.parents_dA = {} #gradient sent back to parents.

    def set_output(self, A):
        """overwrite"""
        self.A = A

    def get_output(self):
        """overwrite"""
        return self.A

    def add_parents(self, parent):
        self.parents.append(parent)

    def get_parents(self):
        return self.parents

    def get_parents_dA(self):
        return self.parents_dA

    def forwardprop(self):
        """overwrite"""
        A, B = self.parents
        self.Z = self.operator.forwardprop(A.get_output(),
                                           B.get_output())
        self.A = self.tranformer.forwardprop(self.Z)
        return self.A

    def local_gradient(self):
        self.dA = self.gradient()
        self.dZ = self.tranformer.backwardprop(self.Z, self.dA)
        return self.dZ

    def backwardprop(self):
        "overwrite"
        self.dZ = self.local_gradient()
        p1, p2 = self.parents
        A, B = p1.get_output(), p2.get_output()
        p1_dA, p2_dA = self.operator.backwardprop(A, B, self.dZ)
        self.parents_dA[p1] = p1_dA
        self.parents_dA[p2] = p2_dA
        p1.add_children_dA(self, p1_dA)
        p2.add_children_dA(self, p2_dA)
        return self.parents_dA

class GraphBuilder():
    """
    build DAG
    """

    def __init__(self, operation, function = identity):
        self.operator = Operator(operation)
        self.tranformer = Transformer(function)

    def build(self, A, B):
        new_node = Variable(self.operator, self.tranformer)
        new_node.add_parents(A)
        new_node.add_parents(B)
        A.add_children(new_node)
        B.add_children(new_node)
        return new_node

def forwardprop(topo_order):
    for layer in topo_order:
        layer.forwardprop()

def backwardprop(topo_order):
    for layer in reversed(topo_order):
        layer.backwardprop()

def dfs(node, order):
    node.set_visited()
    for child in node.get_children():
        if not node.get_visited():
            dfs(node, order)
    order.append(node)

def topo_sort(nodes):
    order = []
    for node in nodes:
        dfs(node, order)
    return order

def gradient_check(topo_order, X, Y, epsilon):
    A = X.get_output()
    dummy_A = A.ravel()
    A_ = np.zeros(shape = dummy_A.shape)
    for i in range(len(dummy_A)):
        dummy_A[i] += epsilon
        X.set_output(dummy_A.reshape(*(A.shape)))
        forwardprop(topo_order)
        A_[i] += Y.get_output().sum()
        dummy_A[i] -= epsilon
    for i in range(len(dummy_A)):
        dummy_A[i] -= epsilon
        X.set_output(dummy_A.reshape(*(A.shape)))
        forwardprop(topo_order)
        A_[i] -= Y.get_output().sum()
        dummy_A[i] += epsilon
    X.set_output(A)
    backwardprop(topo_order)
    dA = 0.5/epsilon*A_.reshape(*(A.shape))
    return (dA - X.get_gradient()).sum()/(dA - X.get_gradient()).size

if __name__ == '__main__':
    np.random.seed(0)
    nodes = []
    dt = GraphBuilder(operation = dot, function = sigmoid)
    A = Parameter(np.random.rand(3,5))
    nodes.append(A)
    B = Constant(np.random.rand(5,6))
    nodes.append(B)
    C = dt.build(A, B)
    nodes.append(C)
    D = Parameter(np.random.rand(4, 3))
    nodes.append(D)
    E = dt.build(D, C)
    nodes.append(E)
    F = Constant(np.random.rand(4, 6))
    nodes.append(F)
    mul = GraphBuilder(multiply, function = sigmoid)
    G = mul.build(F, E)
    nodes.append(G)
    topo_order = topo_sort(nodes)
    forwardprop(topo_order)
    backwardprop(topo_order)
    print(gradient_check(topo_order, D, G, 1e-7))
