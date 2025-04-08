def length_of_longest_substring(s: str) -> int:
    a = set()
    left = 0
    substring_count = 0
    for right in range(len(s)):
        while s[right] in a:
            a.remove(s[left])
            left = left + 1
        a.add(s[right])
        substring_count = max(substring_count, right -left +1)
    return substring_count
==================================================================================================

def contains_nearby_duplicate(nums: list[int], k: int) -> bool:
    b = {}
    for i, num in enumerate(nums):
        if num in b and i - b[num] <= k:
            return True
        b[num] = i
    return False
    

# 🔍 Test cases to validate your solution
if __name__ == "__main__":
    test_cases = [
        ([1, 2, 3, 1], 3, True),
        ([1, 0, 1, 1], 1, True),
        ([1, 2, 3, 1, 2, 3], 2, False),
        ([99, 99], 2, True),
        ([1, 2, 3, 4, 5, 6], 0, False),
        ([1], 1, False)
    ]

    for i, (nums, k, expected) in enumerate(test_cases):
        result = contains_nearby_duplicate(nums, k)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

===================================================================================================

def is_valid_parentheses(s: str) -> bool:
    p_stack = []
    for i in range(len(s)):
        if(s[i] == "(" or s[i] == "[" or s[i] == "{"):
            p_stack.append(s[i])
            
        else:
            if (p_stack and
            ((p_stack[-1] == "(" and s[i] == ")") or
            (p_stack[-1] == "[" and s[i] == "]") or
            (p_stack[-1] == "{" and s[i] == "}"))):
                p_stack.pop()
            else:
                return False
    return not p_stack

Another solution:
def is_valid_parentheses(s: str) -> bool:
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}
    
    for char in s:
        if char in mapping.values():
            stack.append(char)
        elif char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            return False  # invalid character
    return not stack


if __name__ == "__main__":
    test_cases = [
        ("()", True),
        ("()[]{}", True),
        ("(]", False),
        ("([)]", False),
        ("{[]}", True),
        ("", True),
        ("[", False),
        ("([])", True)
    ]

    for i, (inp, expected) in enumerate(test_cases):
        result = is_valid_parentheses(inp)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

=====================================================================================================


from collections import deque

def num_islands(grid: list[list[str]]) -> int:
    if not grid:
        return 0
        
    rows, cols = len(grid), len(grid[0])
    visit = set()
    islands = 0
    
    def bfs(r,c):
        q = deque()
        visit.add((r,c))
        q.append((r,c))
        
        while q:
            row, col = q.popleft()
            directions = [[1,0], [-1,0], [0,1], [0,-1]]
            
            for dr, dc in directions:
                new_r, new_c = row + dr, col + dc
                if (new_r in range(rows) and 
                new_c in range(cols) and grid[new_r][new_c] == "1"
                and (new_r,new_c) not in visit):
                    q.append((new_r, new_c))
                    visit.add((new_r, new_c))
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r,c) not in visit:
                bfs(r,c)
                islands = islands +1
    return islands
    
    
            

if __name__ == "__main__":
    test_grids = [
        (
            [["1","1","0","0","0"],
             ["1","1","0","0","0"],
             ["0","0","1","0","0"],
             ["0","0","0","1","1"]],
            3
        ),
        (
            [["1","1","1"],
             ["0","1","0"],
             ["1","1","1"]],
            1
        ),
        (
            [["0","0","0"],
             ["0","0","0"],
             ["0","0","0"]],
            0
        ),
        (
            [["1"]],
            1
        ),
        (
            [["0"]],
            0
        )
    ]

    for i, (grid, expected) in enumerate(test_grids):
        result = num_islands(grid)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

=============================================================================================

from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order(root: TreeNode) -> list[list[int]]:
    if not root:
        return 0
        
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
        
    return result

# Example tree:
#         3
#        / \
#       9  20
#          / \
#         15  7
root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)

print("Output:", level_order(root))  # Expected: [[3], [9, 20], [15, 7]]

====================================================================================

def majority_element(nums: list[int]) -> int:
    d = {}
    count_nums = []
    division = len(nums)//2
    nums_set = set(nums)
    for i in nums_set:
        count = nums.count(i)
        count_nums.append(count)
    res = dict(zip(nums_set, count_nums))
    result = max(res, key=res.get)
    
    return result
        
    

if __name__ == "__main__":
    test_cases = [
        ([3, 2, 3], 3),
        ([2, 2, 1, 1, 1, 2, 2], 2),
        ([1, 1, 1, 2, 3, 1], 1),
        ([4, 4, 5], 4)
    ]

    for i, (nums, expected) in enumerate(test_cases):
        result = majority_element(nums)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

===================================================================================

def backspace_compare(s: str, t: str) -> bool:
    def process(string):
        t1 = []
        for i in string:
            if(i != "#"):
                t1.append(i)
            elif t1:
                t1.pop()
        return t1
            
    return process(s) == process(t)

if __name__ == "__main__":
    test_cases = [
        ("ab#c", "ad#c", True),     # both become "ac"
        ("ab##", "c#d#", True),     # both become ""
        ("a#c", "b", False),
        ("xy#z", "xzz#", True),     # both become "xz"
        ("a##c", "#a#c", True)
    ]

    for i, (s, t, expected) in enumerate(test_cases):
        result = backspace_compare(s, t)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

=====================================================================================


