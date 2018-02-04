class Node:
    # Initialized nodes to now have an extra attribute called priority and initialized it to be something large at first
    def __init__(self,content=None, priority = 999999, next=None):
        self.content = content
        self.next  = next
        self.priority = priority
class Queue:
    def __init__(self):
        self.length = 0
        self.head = None

    def is_empty(self):
        return self.length == 0

    def clear(self):
        self.length = 0
        self.head = None
        
    def insert(self, content, priority):
        node = Node(content, priority)
        if self.head is None:
             #If list is empty the new node goes first
            self.head = node
        else:
             #Find the last node in the list
            last = self.head
            while last.next:
                last = last.next
             #Append the new node
            last.next = node
        self.length = self.length + 1

    def remove(self):
        original_head = self.head
        node_with_lowest_priority = self.head
        pointer = self.head

        # This while-loop gets the node with the lowest priority
        while pointer != None:
            if pointer.priority < node_with_lowest_priority.priority:
                node_with_lowest_priority = pointer
            else:
                pointer = pointer.next


        pointer = original_head
        print(int(node_with_lowest_priority.content))

        # This is the special case for when the node with the lowest priority is originally at the head of the queue
        if pointer == node_with_lowest_priority:
            self.head = original_head.next
            self.length = self.length - 1
            return(self.head)

        ## This is for all other cases.
        else:
            # This iterates through the queue and stops when pointer.NEXT = node_with_lowest_priority
            while pointer.next != node_with_lowest_priority:
                pointer = pointer.next

            pointer.next = node_with_lowest_priority.next
            self.head = original_head
            self.length = self.length - 1
            return(self.head)

line = Queue()
while True:
    content_question = float(input("Please enter the content: "))
    if content_question != -1:
        priority_question = float(input("Now enter a priority: "))
        line.insert(content_question, priority_question)
        continue
    else:
        numbers_to_remove = int(input("How many numbers do you want to remove from the queue? :"))
        for i in range(1,numbers_to_remove + 1):
            line.remove()
            i += 1

        break



