from typing import Optional

class AVLError(Exception):
    pass

"""A class represnting a node in an AVL tree"""

class AVLNode(object):
    _VIRTUAL_HEIGHT = -1
    key: Optional[int]
    left: "AVLNode"
    right: "AVLNode"
    parent: "AVLNode"
    height: int
    max: Optional["AVLNode"]
    
    """Constructor, you are allowed to add more fields. 
    
    @type key: int
    @param key: key of your node
    @type value: string
    @param value: data of your node
    """

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.parent = None 
        if key is None:
            self.height = -1 
            self.left = None
            self.right = None
            self.max = None 
        else:
            self.height = 0
            self.max = self        
            self.left = AVLNode(None, None) 
            self.left.parent = self
            self.right = AVLNode(None, None)
            self.right.parent = self

    def __repr__(self) -> str:
        return f"VIRTUAL" if self.virtual else f"{self.key=}, {self.value=}"
    
    """returns whether self is not a virtual node 

    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    """
    def is_real_node(self):
        return not self.virtual

    @property
    def virtual(self):
        return self.key is None

    def set_virtual(self):
        self.key = None
        self.height = -1

    def update_max_locally(self) -> None:
        if self.virtual:
            return
        if self.right.is_real_node():
            self.max = self.right.max
        else:
            self.max = self

    def update_height_locally(self) -> int:
        if self.virtual:
            return 0
        previous_height = self.height
        self.height = max(self.left.height, self.right.height) + 1
        if self.height > previous_height:
            return 1
        return 0

    def update_fields_locally(self) -> int:
        self.update_max_locally()
        return self.update_height_locally()

    def set_left(self, node: "AVLNode", update_height=True) -> int:
        if self.virtual:
            raise AVLError("Can not perform operations on virtual node")
        self.left = node
        node.parent = self
        if update_height:
            return self.update_fields_locally()
        return 0

    def set_right(self, node: "AVLNode", update_height=True) -> int:
        if self.virtual:
            raise AVLError("Can not perform operations on virtual node")
        self.right = node
        node.parent = self
        if update_height:
            return self.update_fields_locally()
        return 0

    @property
    def is_leaf(self):
        return self.right.virtual and self.left.virtual

    def successor(self) -> Optional["AVLNode"]:
        if self is self.max:
            return None
        if self.right.is_real_node():
            tmp = self.right
            while tmp.left.is_real_node():
                tmp = tmp.left
            return tmp
        x = self.right
        y = self.parent
        while y.is_real_node() and x is y.right:
            x = y
            y = x.parent
        return y

    def _replace_child(self, current: "AVLNode", replacement: "AVLNode") -> int:
        if current is self.left:
            return self.set_left(replacement)
        elif current is self.right:
            return self.set_right(replacement)
        else:
            raise AVLError("Attempt to replace a child that does not exist")

    """
    Replacing a child with an empty virtual node.
    """
    def remove_child(self, node: "AVLNode"):
        self._replace_child(node, AVLNode(None, ""))

    def _replace_with_succ_or_pre(self: "AVLNode", node: "AVLNode", typ: bool):
        if typ and node.right.is_real_node():
            node.parent._replace_child(node, node.right)
            node.remove_child(node.right)
        elif node.left.is_real_node():
            node.parent._replace_child(node, node.left)
            node.remove_child( node.left)
        else:
            node.parent.remove_child(node)

        if self.parent is not None and self.parent.is_real_node():
            self.parent._replace_child(self, node)
        else:
            node.parent = None

        if self.left is not node and self.left.is_real_node():
            node.set_left(self.left)
        if self.right is not node and self.right.is_real_node():
            node.set_right(self.right)

        self.parent = AVLNode(None, "")
        self.set_virtual()

    """
    Replaces `self` with `node`.
    @pre self: self is not root
    @pre node: `node` is a successor to `self`.
    """
    def replace_with_successor(self: "AVLNode"):
        successor = self.successor()
        assert successor
        self._replace_with_succ_or_pre(successor, True)

    @property
    def height_diff(self):
        if self.virtual:
            return 0
        return self.left.height - self.right.height

    @property
    def right_is_too_big(self):
        return self.height_diff < -1

    @property
    def left_is_bigger(self):
        return self.height_diff > 0

    def _roll_right(self: "AVLNode"):
        if self.virtual:
            raise AVLError("Can not perform operations on virtual node")

        sub_tree_parent = self.parent
        previous_left_node = self.left
        middle_node = self.left.right

        self.set_left(middle_node)
        previous_left_node.set_right(self)
        previous_left_node.parent = sub_tree_parent
        if sub_tree_parent is not None and sub_tree_parent.is_real_node():
            sub_tree_parent._replace_child(self, previous_left_node)
        self.update_fields_locally()

    def _roll_left(self):
        if self.virtual:
            raise AVLError("Can not perform operations on virtual node")

        sub_tree_parent = self.parent
        previous_right_node = self.right
        middle_node = self.right.left

        self.set_right(middle_node)
        previous_right_node.set_left(self)
        previous_right_node.parent = sub_tree_parent
        if sub_tree_parent is not None and sub_tree_parent.is_real_node():
            sub_tree_parent._replace_child(self, previous_right_node)
        self.update_fields_locally()

    def _double_roll_left(self):
        if self.virtual:
            raise AVLError("Can not perform operations on virtual node")
        self.right._roll_right()
        self._roll_left()

    @property
    def left_is_too_big(self):
        return self.height_diff > 1

    @property
    def right_is_bigger(self):
        return self.height_diff < 0

    def _double_roll_right(self):
        if self.virtual:
            raise AVLError("Can not perform operations on virtual node")
        self.left._roll_left()
        self._roll_right()

    """
    Returns the total amount of promotions made.
    """
    def rebalance(self) -> int:
        if self.virtual:
            return 0

        promotions_total = 0
        curr = self

        while curr is not None and curr.is_real_node():
            promotions_made = 0
            
            if curr.right_is_too_big:
                if curr.right.left_is_bigger:
                    curr._double_roll_left()
                else:
                    curr._roll_left()
            elif curr.left_is_too_big:
                if curr.left.right_is_bigger:
                    curr._double_roll_right()
                else:
                    curr._roll_right()
            else:
                promotions_made = curr.update_fields_locally()

            promotions_total += promotions_made
            curr = curr.parent

        return promotions_total

    def predecessor(self) -> Optional["AVLNode"]:
        if self.left.is_real_node():
            tmp = self.left
            while tmp.right.is_real_node():
                tmp = tmp.right
            return tmp

        x = self.left
        y = self.parent
        while y.is_real_node() and x is y.left:
            x = y
            y = x.parent
        return y

    def replace_with_predecessor(self: "AVLNode"):
        predecessor = self.predecessor()
        assert predecessor
        self._replace_with_succ_or_pre(predecessor, False)

    def in_order_keys(self):
        if self.virtual:
            return []
        return (
            self.left.in_order_keys()
            + [(self.key, self.value)]
            + self.right.in_order_keys()
        )
    
"""
A class implementing an AVL tree.
"""

class AVLTree(object):
    root: AVLNode
    t_size: int

    """
    Constructor, you are allowed to add more fields.
    """
    def __init__(self, root: Optional[AVLNode] = None):
        self.root = root if root else AVLNode(None, "")
        self.t_size = 0

    @property
    def height(self) -> int:
        return self.root.height

    @property
    def empty(self) -> bool:
        return self.root.virtual

    def _update_root(self):
        if self.root.virtual:
            return
        while self.root.parent is not None and self.root.parent.is_real_node():
            self.root = self.root.parent

    def _find_left_join_point(self, height: int) -> AVLNode:
        current = self.root
        while current.left.is_real_node() and current.left.height > height:
            current = current.left
        return current

    def _find_right_join_point(self, height: int) -> AVLNode:
        current = self.root
        while current.right.is_real_node() and current.right.height > height:
            current = current.right
        return current

    def _left_join(self, separator: AVLNode, smaller: "AVLTree", bigger: "AVLTree"):
        join_point = bigger._find_left_join_point(smaller.height)
        separator.set_left(smaller.root)   
        separator.set_right(join_point.left) 
        join_point.set_left(separator)      

    def _right_join(self, separator: AVLNode, smaller: "AVLTree", bigger: "AVLTree"):
        join_point = smaller._find_right_join_point(bigger.height)
        separator.set_right(bigger.root)
        separator.set_left(join_point.right)
        join_point.set_right(separator)  

    def _middle_join(self, separator: AVLNode, smaller: "AVLTree", bigger: "AVLTree"):
        separator.set_left(smaller.root)
        separator.set_right(bigger.root)
        self.root = separator

    def _search_path(self, key: int) -> tuple[AVLNode, int]:
        curr = self.root
        last_node = curr
        path_len = 0
        while curr.is_real_node():
            last_node = curr
            path_len += 1
            
            if key == curr.key:
                return curr, path_len
            if key < curr.key:
                curr = curr.left
            else:
                curr = curr.right
                
        return last_node, path_len

    """searches for a node in the dictionary corresponding to the key (starting at the root)
        
    @type key: int
    @param key: a key to be searched
    @rtype: (AVLNode,int)
    @returns: a tuple (x,e) where x is the node corresponding to key (or None if not found),
    and e is the number of edges on the path between the starting node and ending node+1.
    """
    def search(self, key: int) -> tuple[Optional[AVLNode], int]:
        if self.empty:
            return None, 0

        node, cost = self._search_path(key)
        
        if node.key == key:
            return node, cost
            
        return None, cost
    
    """searches for a node in the dictionary corresponding to the key, starting at the max
        
    @type key: int
    @param key: a key to be searched
    @rtype: (AVLNode,int)
    @returns: a tuple (x,e) where x is the node corresponding to key (or None if not found),
    and e is the number of edges on the path between the starting node and ending node+1.
    """
    def finger_search(self, key):
        if self.empty:
            return None, 1
        
        count = 1
        curr = self.max_node()

        if curr.key == key:
            return curr, count

        while curr.is_real_node() and curr.key > key:
            if curr.parent is None or not curr.parent.is_real_node():
                break
            if curr.parent.key < key:
                break
            if curr.parent.key == key:
                return (curr.parent, count + 1)
            curr = curr.parent
            count += 1

        while curr.is_real_node():
            if curr.key == key:
                return (curr, count)
            elif curr.key > key:
                if curr.left.is_real_node():
                    curr = curr.left
                    count += 1
                else:
                    return (None, count)
            else:
                if curr.right.is_real_node():
                    curr = curr.right
                    count += 1
                else:
                    return (None, count)

        return (None, count)

    """inserts a new node into the dictionary with corresponding key and value (starting at the root)

    @type key: int
    @pre: key currently does not appear in the dictionary
    @param key: key of item that is to be inserted to self
    @type val: string
    @param val: the value of the item
    @rtype: (AVLNode,int,int)
    @returns: a 3-tuple (x,e,h) where x is the new node,
    e is the number of edges on the path between the starting node and new node before rebalancing,
    and h is the number of PROMOTE cases during the AVL rebalancing
    """
    def insert(self, key: int, val: str) -> tuple[AVLNode, int, int]:
        if self.empty:
            self.root = AVLNode(key, val)
            self.t_size = 1
            return (self.root, 0, 0)

        curr = self.root
        depth = 0 
        while curr.is_real_node():
            if key == curr.key:
                return (curr, -1, -1)
            if key < curr.key:
                if not curr.left.is_real_node():
                    break
                curr = curr.left
            else:
                if not curr.right.is_real_node():
                    break
                curr = curr.right
            
            depth += 1 

        new_node = AVLNode(key, val)
        promotions = 0
        if key < curr.key:
            promotions += curr.set_left(new_node)
        else:
            promotions += curr.set_right(new_node)

        promotions += curr.rebalance()
        self._update_root()
        self.t_size += 1

        return (new_node, depth + 1, promotions)

    """inserts a new node into the dictionary with corresponding key and value, starting at the max

    @type key: int
    @pre: key currently does not appear in the dictionary
    @param key: key of item that is to be inserted to self
    @type val: string
    @param val: the value of the item
    @rtype: (AVLNode,int,int)
    @returns: a 3-tuple (x,e,h) where x is the new node,
    e is the number of edges on the path between the starting node and new node before rebalancing,
    and h is the number of PROMOTE cases during the AVL rebalancing
    """
    def finger_insert(self, key, val):
        if self.empty:
            self.root = AVLNode(key, val)
            self.t_size = 1
            return (self.root, 0, 0)
        
        count = 1
        curr = self.max_node()

        while curr.is_real_node() and curr.key > key:
            if curr.parent is None or not curr.parent.is_real_node():
                break
            curr = curr.parent
            count += 1

        while curr.is_real_node():
            if curr.key == key:
                return (curr, -1, -1)
            elif curr.key > key:
                if curr.left.is_real_node():
                    curr = curr.left
                    count += 1
                else:
                    break
            else: 
                if curr.right.is_real_node():
                    curr = curr.right
                    count += 1
                else:
                    break
                
        new_node = AVLNode(key, val)
        if curr.is_real_node():
            if curr.key > key:
                curr.set_left(new_node, update_height=False) 
            else:
                curr.set_right(new_node, update_height=False) 
        else:
            return None, -1, -1
        
        h = new_node.rebalance()
        self._update_root()
        self.t_size += 1

        return new_node, count, h
    
    """deletes node from the dictionary

    @type node: AVLNode
    @pre: node is a real pointer to a node in self
    """
    def delete(self, node: AVLNode) -> None:
        if node is None or node.virtual:
            return

        rebalance_start_node = None
        is_root = (node is self.root)

        if node.is_leaf:
            rebalance_start_node = node.parent
            node.set_virtual()
        else:
            succ = node.successor()
            if succ and succ.is_real_node():
                if succ.parent is node:
                    rebalance_start_node = succ 
                else:
                    rebalance_start_node = succ.parent
                
                node.replace_with_successor()
                if is_root:
                    self.root = succ
            else:
                pred = node.predecessor()
                if pred and pred.is_real_node():
                    if pred.parent is node:
                        rebalance_start_node = pred
                    else:
                        rebalance_start_node = pred.parent
                        
                    node.replace_with_predecessor()
                    if is_root:
                        self.root = pred

        self.t_size -= 1
        if rebalance_start_node and rebalance_start_node.is_real_node():
            rebalance_start_node.rebalance()

        self._update_root()

    """joins self with item and another AVLTree

    @type tree2: AVLTree 
    @param tree2: a dictionary to be joined with self
    @type key: int 
    @param key: the key separting self and tree2
    @type val: string
    @param val: the value corresponding to key
    @pre: all keys in self are smaller than key and all keys in tree2 are larger than key,
    or the opposite way
    """

    def join(self, tree2: "AVLTree", key: int, val: str) -> None:
        if self.empty and tree2.empty:
            self.insert(key, val)
            return

        if self.empty:
            tree2.insert(key, val)
            self.root = tree2.root
            self.t_size = tree2.t_size
            return

        if tree2.empty:
            self.insert(key, val)
            return

        separator = AVLNode(key, val)
       
        if self.root.key < key:
            left_tree = self
            right_tree = tree2
        else:
            left_tree = tree2
            right_tree = self

        if left_tree.height < right_tree.height:
            self._left_join(separator, left_tree, right_tree)
            self.root = right_tree.root 
            
        elif left_tree.height > right_tree.height:
            self._right_join(separator, left_tree, right_tree)
            self.root = left_tree.root
            
        else:
            self._middle_join(separator, left_tree, right_tree)
            self.root = separator

        separator.rebalance()        
        self._update_root()        
        self.t_size = self.t_size + tree2.t_size + 1
       
    """splits the dictionary at a given node

    @type node: AVLNode
    @pre: node is in self
    @param node: the node in the dictionary to be used for the split
    @rtype: (AVLTree, AVLTree)
    @returns: a tuple (left, right), where left is an AVLTree representing the keys in the 
    dictionary smaller than node.key, and right is an AVLTree representing the keys in the 
    dictionary larger than node.key.
    """

    def split(self, node: AVLNode) -> tuple["AVLTree", "AVLTree"]:
        if node is None or node.virtual:
            return (AVLTree(), AVLTree())
            
        T1 = AVLTree()
        if node.left.is_real_node():
            T1.root = node.left
            T1.root.parent = None
            
        T2 = AVLTree()
        if node.right.is_real_node():
            T2.root = node.right
            T2.root.parent = None

        curr = node
        while curr.parent is not None and curr.parent.is_real_node():
            parent = curr.parent
            p_key = parent.key
            p_val = parent.value
            if parent.right == curr:
                left_subtree = AVLTree()
                if parent.left.is_real_node():
                    left_subtree.root = parent.left
                    left_subtree.root.parent = AVLNode(None, "")
                T1.join(left_subtree, p_key, p_val)
                
            else:
                right_subtree = AVLTree()
                if parent.right.is_real_node():
                    right_subtree.root = parent.right
                    right_subtree.root.parent = AVLNode(None, "")
                T2.join(right_subtree, p_key, p_val)
                
            curr = parent

        return (T1, T2)

    """returns an array representing dictionary 

    @rtype: list
    @returns: a sorted list according to key of touples (key, value) representing the data structure
    """
    def avl_to_array(self) -> list[tuple[int, str]]:
        return self.root.in_order_keys()

    """returns the node with the maximal key in the dictionary

    @rtype: AVLNode
    @returns: the maximal node, None if the dictionary is empty
    """
    def max_node(self) -> Optional[AVLNode]:
        return self.root.max

    """returns the number of items in dictionary 

    @rtype: int
    @returns: the number of items in dictionary 
    """
    def size(self) -> int:
        return self.t_size

    """returns the root of the tree representing the dictionary

    @rtype: AVLNode
    @returns: the root, None if the dictionary is empty
    """
    def get_root(self) -> Optional[AVLNode]:
        if self.root.virtual:
            return None
        return self.root
    



    



