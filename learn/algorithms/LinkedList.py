

class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None
        self.prev = None
    
    def traverse(self):
        curr = self 
        output = ""
        while curr is not None:
            output += f"{curr.value}->"
            curr = curr.next
        print(output)


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dict = {}
        self.LRU = Node(0,0)
        self.MRU = Node(0,0)
        self.LRU.next = self.MRU.prev
        self.MRU.prev = self.LRU


    def insertRight(self, node:Node):
        prev = self.MRU.prev
        prev.next = node 
        node.prev = prev
        node.next = self.MRU
        self.MRU.prev = node
    
    def remove(self, node:Node):
        prev = node.prev 
        next = node.next
        prev.next = next 
        next.prev = prev
        node.next = node.prev = None

    def put(self, key:int, value:int) -> None:
        if key in self.dict:
            # variable exist: We should update and move to the right
            self.dict[key].value = value 
            self.remove(self.dict[key])    
            self.insertRight(self.dict[key])
        if key not in self.dict and len(self.dict) < self.capacity:
            # node not exist and there is capacity
            newNode = Node(key=key,value=value)
            self.dict[key] = newNode 
            self.insertRight(newNode)

        if key not in self.dict and len(self.dict) >= self.capacity:
            # node not exist and we need to remove LRU to allow storage
            newNode = Node(key=key,value=value)            
            lru = self.LRU.next 
            self.remove(lru)
            self.dict.pop(lru.key, 'None')
            self.dict[key] = newNode
            self.insertRight(newNode)
            


    def get(self, key: int) -> int:
        if key in self.dict:
            self.remove(self.dict[key])
            self.insertRight(self.dict[key])
            return self.dict[key].value
        return -1
            

  
lru = LRUCache(2)
lru.put(1,1)
lru.put(2,2)
print(lru.get(1))
lru.put(3,3)
print(lru.get(2))

lru.put(4,4)
print(lru.get(1))
print(lru.get(3))
print(lru.get(4))